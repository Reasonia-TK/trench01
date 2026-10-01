#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トレンチ構造の表面帯電 2D シミュレーション
==========================================

プラズマから飛来する正イオン(異方的・高エネルギー)と電子(等方的・低エネルギー)が
誘電体トレンチの壁面に溜まり、電位を作る様子(electron shading 効果)を計算する。

モデル
------
* 2D (x: 横, y: 上向き)。奥行き方向は単位長さ(C/m などは「奥行き 1 m あたり」)。
* 構造: 接地Si基板(y=0) の上に誘電体(SiO2, eps_r=3.9)があり、その上にマスク(既定 300 nm)を載せる。
        マスクと誘電体を貫く開口(トレンチ)が中央にある。マスクの開口はトレンチと同じ幅。
        x 方向は周期境界(トレンチが周期的に並ぶ)。上端は真空領域。
* マスク: mask_type="conductor" (既定, doped carbon などの導電性マスク) は等電位の浮遊導体として扱う。
        導体に当たった電荷は導体内を自由に動き、合計の電荷で導体の電位が決まる。
        導体の緩和時間 (~数十 μs) は 1 バッチより短く、電荷を陽的に足すと電位が振動するため、
        バッチ中の電子電流がボルツマン因子で電位に応答するとして半陰的に更新する (conductor_step)。
        mask_type="dielectric" は比誘電率 mask_eps_r の誘電体。mask_t=0 でマスクなし。
* 粒子: Monte Carlo テスト粒子。電場は凍結して 1 バッチ分の粒子を追跡
        → 表面に当たった粒子の電荷を壁面セルに蓄積
        → Poisson 方程式を解き直して電場を更新、を繰り返す。
        (粒子の通過時間 ~ ps ≪ 帯電の時定数 ~ μs〜ms なので準静的近似が成り立つ)
* 入射条件(上端境界):
    - イオン  : 一定エネルギー Ei で垂直入射 + 横方向の熱速度 Ti
    - 電子    : 温度 Te のマクスウェル分布のフラックス分布(等方的)
    - 電子フラックス = イオンフラックス (フローティング表面の平均的な電流バランス)
* Poisson: div(eps_r grad phi) = -rho/eps0 を有限差分(セル中心, 面で調和平均)で解く。
          行列は形状固定なので LU 分解を 1 回だけ行う。
* 粒子が固体セルに入ったら完全吸収(付着確率 1, イオン反射なし)。
* 表面リーク: 誘電体の表面セル(真空に接するセル)を抵抗網でつなぎ、電位差に応じて
    表面に沿って電荷が移動する(シート伝導度 sigma_s [S], 0 で無効)。
    dQ/dt = G phi (G: 表面セル間のコンダクタンス行列, phi = P Q は Poisson の応答)
    を後退オイラーで解くので、sigma_s が大きくても無条件安定で総電荷も保存される。

使い方
------
依存パッケージは uv で管理している (pyproject.toml / uv.lock)。uv run が初回に .venv を作る。

    uv run trench_charging_2d.py                 # 既定値で実行 (約4分, 30 ms 分の帯電)
    uv run trench_charging_2d.py --n_batches 60  # 短時間で動作確認
    uv run trench_charging_2d.py --trench_d 120  # アスペクト比を変える
    uv run trench_charging_2d.py --sigma_s 1e-14 # 表面リークあり (シート伝導度 1e-14 S)
    uv run trench_charging_2d.py --until_steady  # 飽和帯電に達するまで継続 (上限 --max_batches)
    uv run trench_charging_2d.py --mask_t 40     # マスク厚 40 セル = 200 nm (0 でマスクなし)
    uv run trench_charging_2d.py --mask_type dielectric --mask_eps_r 3.0  # 絶縁性のマスク
    uv run trench_charging_2d.py --help          # 変更できるパラメータ一覧

飽和まで継続モード (--until_steady)
------------------------------------
steady_window バッチごとに「直近ブロックの平均電位」と「その前のブロックの平均電位」を
全観測点(マスク上面・側壁上/中/下・底)で比較し、変化が
    max(steady_rtol x 最大電位, 3 x 平均差の標準誤差)
以内という条件を steady_hold 回連続で満たしたら終了する(ノイズの範囲内の変動は無視)。
この場合 n_batches は使われず、max_batches が上限になる。

出力: <out>_fields.png, <out>_history.png, <out>.npz
"""
import argparse
import time
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 物理定数
E_CHARGE = 1.602176634e-19   # [C]
EPS0 = 8.8541878128e-12      # [F/m]
M_E = 9.1093837015e-31       # [kg]
AMU = 1.66053906660e-27      # [kg]

MASK_TYPES = ("conductor", "dielectric")  # mask_type の選択肢


@dataclass
class Params:
    # ---- 格子・形状 ----
    dx: float = 5e-9          # セルサイズ [m]
    nx: int = 60              # 横方向セル数 (周期境界)
    trench_w: int = 20        # トレンチ幅 [セル]  (20 -> 100 nm)
    trench_d: int = 80        # トレンチ深さ [セル] (80 -> 400 nm, AR=4)
    floor_t: int = 10         # トレンチ底の誘電体厚 [セル]
    n_vac: int = 50           # 構造の上面 (マスク上面) より上の真空領域 [セル]
    eps_r: float = 3.9        # 誘電体の比誘電率 (SiO2)
    top_bc: str = "dirichlet"  # 上端境界: "dirichlet"(phi=0, プラズマ電位) or "neumann"
    # ---- マスク (誘電体の上に載せる。開口はトレンチと同じ幅) ----
    mask_t: int = 60          # マスク厚 [セル] (60 -> 300 nm, 0 でマスクなし)
    mask_type: str = "conductor"  # "conductor"(導電性: doped carbon など, 浮遊導体) or "dielectric"
    mask_eps_r: float = 3.0   # マスクの比誘電率 (mask_type="dielectric" のときだけ使う)
    # ---- プラズマ ----
    flux: float = 1e20        # 粒子フラックス [m^-2 s^-1] (= 1e16 cm^-2 s^-1)
    ion_mass_amu: float = 40.0  # イオン質量 [amu] (Ar+)
    ion_energy_eV: float = 100.0  # イオン入射エネルギー(シース電圧) [eV]
    ion_temp_eV: float = 0.5  # イオンの横方向温度 [eV] -> 角度広がり
    electron_temp_eV: float = 3.0  # 電子温度 [eV]
    # ---- 表面リーク ----
    sigma_s: float = 0.0      # 表面シート伝導度 [S] (0=リークなし)。目安: 1e-16〜1e-13
    # ---- 飽和まで継続モード ----
    until_steady: bool = False  # True: 電位が飽和するまで継続 (n_batches は無視, max_batches が上限)
    max_batches: int = 3000   # 継続モードの上限バッチ数 (3000 -> 300 ms)
    steady_window: int = 30   # 飽和判定のブロック長 [バッチ]
    steady_rtol: float = 0.005  # ブロック間の電位変化の許容値 (最大電位に対する割合)
    steady_hold: int = 2      # 判定を連続で満たすべき回数
    # ---- 数値 ----
    n_batches: int = 300      # 固定長モードのバッチ数 (総時間 = n_batches * dt_batch = 30 ms)
    n_per_batch: int = 3000   # 1 バッチあたりの粒子数 (イオン, 電子 それぞれ)
    dt_batch: float = 1e-4    # 1 バッチが表す物理時間 [s] (大きすぎると電位が振動する)
    cfl: float = 0.35         # 1 ステップで進む距離の上限 [セル]
    max_steps: int = 3000     # 1 粒子あたりの最大ステップ数 (超えたら打ち切り=捕捉粒子)
    seed: int = 1


# ---------------------------------------------------------------- 形状
def build_geometry(p):
    """solid[ix, iy] (True=固体) と mask[ix, iy] (True=マスク, solid の一部) を作る。iy=0 が基板側。
    誘電体 (厚さ floor_t + trench_d) の上に厚さ mask_t のマスクを載せ、
    マスクと誘電体を貫く幅 trench_w の開口 (誘電体部分の深さ trench_d) を中央に掘る。"""
    if p.mask_t < 0:
        raise ValueError(f"mask_t は 0 以上にしてください: {p.mask_t}")
    if p.mask_type not in MASK_TYPES:
        raise ValueError(f"mask_type は {' / '.join(MASK_TYPES)} のどちらかにしてください: {p.mask_type!r}")
    top_ox = p.floor_t + p.trench_d           # 誘電体の上面 (= マスクの下面)
    ny = top_ox + p.mask_t + p.n_vac
    solid = np.zeros((p.nx, ny), dtype=bool)
    solid[:, : top_ox + p.mask_t] = True
    i0 = (p.nx - p.trench_w) // 2
    i1 = i0 + p.trench_w
    solid[i0:i1, p.floor_t:] = False          # トレンチ (とマスクの開口) を掘る
    mask = solid.copy()
    mask[:, :top_ox] = False
    return solid, mask, i0, i1


def permittivity(solid, mask, p):
    """セルごとの比誘電率 (真空 1, 誘電体 eps_r, 誘電体マスク mask_eps_r)。
    導体マスクのセルの値は使わない (build_poisson で導体として扱う)。"""
    eps = np.where(solid, p.eps_r, 1.0)
    return np.where(mask, p.mask_eps_r, eps) if p.mask_type == "dielectric" else eps


# ---------------------------------------------------------------- Poisson
def build_poisson(eps, cond, p):
    """div(eps_r grad phi) の疎行列を作る (x: 周期, y下端: 接地, y上端: 設定による)。
    eps: セルごとの比誘電率, cond: 導体のセル (True)。導体と接する面は、導体表面 (= 面の位置) が
    導体の電位になるよう 2 eps で結合する (調和平均で導体側の誘電率 -> ∞ とした極限)。
    導体どうしの面は結合しない (導体は等電位なので make_solver で 1 つの未知数にまとめる)。"""
    nx, ny = eps.shape
    dx2 = p.dx ** 2
    harm = lambda a, b: 2.0 * a * b / (a + b)  # 面の誘電率 (調和平均)

    def face(a, b, ca, cb):
        """面の誘電率。片側が導体なら 2 x もう片側、両側とも導体なら 0。"""
        h = np.where(ca, 2.0 * b, np.where(cb, 2.0 * a, harm(a, b)))
        return np.where(ca & cb, 0.0, h)

    ce = face(eps, np.roll(eps, -1, axis=0), cond, np.roll(cond, -1, axis=0)) / dx2  # 東(ix+1)側の面
    cw = np.roll(ce, 1, axis=0)                                                      # 西側の面

    c_int = face(eps[:, :-1], eps[:, 1:], cond[:, :-1], cond[:, 1:]) / dx2          # 上下の内部面
    cn = np.zeros((nx, ny)); cn[:, :-1] = c_int
    cs = np.zeros((nx, ny)); cs[:, 1:] = c_int

    diag_n = cn.copy()
    diag_s = cs.copy()
    diag_s[:, 0] += 2.0 * eps[:, 0] / dx2              # 下端: 接地 (0 V) まで半セル
    if p.top_bc == "dirichlet":
        diag_n[:, -1] += 2.0 * eps[:, -1] / dx2        # 上端: phi = 0 まで半セル
    diag = -(ce + cw + diag_n + diag_s)

    idx = np.arange(nx * ny).reshape(nx, ny)
    rows = [idx.ravel(), idx.ravel(), idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    cols = [idx.ravel(), np.roll(idx, -1, axis=0).ravel(), np.roll(idx, 1, axis=0).ravel(),
            idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    vals = [diag.ravel(), ce.ravel(), cw.ravel(), cn[:, :-1].ravel(), cs[:, 1:].ravel()]
    A = sps.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nx * ny, nx * ny)).tocsc()
    return A


def make_solver(A, cond):
    """Poisson の解法 solve(b) -> phi (全セル, 1 次元) を作る。行列は形状固定なので LU 分解は 1 回だけ。
    導体セル (cond) は 1 つの未知数 (浮遊導体の電位) にまとめる: 全セルの電位を E @ u と書いて
    (E^T A E) u = E^T b を解く。E^T b の導体成分は導体セルの電荷の合計なので、導体に入った電荷は
    導体内を自由に動き、合計だけが電位を決める。b は 2 次元 (列ごとに別の右辺) でもよい。"""
    if not cond.any():
        return splu(A).solve
    flat = cond.ravel()
    n_free = int((~flat).sum())
    col = np.where(flat, n_free, np.cumsum(~flat) - 1)    # 導体セルはすべて最後の未知数へ
    E = sps.csr_matrix((np.ones(flat.size), (np.arange(flat.size), col)), shape=(flat.size, n_free + 1))
    lu = splu((E.T @ A @ E).tocsc())
    return lambda b: E @ lu.solve(E.T @ b)


def _grad(c, m, pl, vm, vp, dx):
    """真空セルだけを使った差分で -dphi/dx を返す (誘電体界面をまたがない)。"""
    g = np.zeros_like(c)
    both = vm & vp
    g = np.where(both, (pl - m) / (2 * dx), g)
    g = np.where(vp & ~vm, (pl - c) / dx, g)
    g = np.where(vm & ~vp, (c - m) / dx, g)
    return -g


def compute_field(phi, vac, p):
    """電場 E = -grad(phi) をセル中心で計算 (真空セルのみ有効)。"""
    dx = p.dx
    Ex = _grad(phi, np.roll(phi, 1, 0), np.roll(phi, -1, 0),
               np.roll(vac, 1, 0), np.roll(vac, -1, 0), dx)

    ghost_top = -phi[:, -1] if p.top_bc == "dirichlet" else phi[:, -1]
    ghost_bot = -phi[:, 0]
    pD = np.concatenate([ghost_bot[:, None], phi[:, :-1]], axis=1)
    pU = np.concatenate([phi[:, 1:], ghost_top[:, None]], axis=1)
    f = np.zeros((phi.shape[0], 1), dtype=bool)
    vD = np.concatenate([f, vac[:, :-1]], axis=1)
    vU = np.concatenate([vac[:, 1:], ~f], axis=1)
    Ey = _grad(phi, pD, pU, vD, vU, dx)
    return Ex, Ey


# ---------------------------------------------------------------- 粒子
def inject(p, n, ny, rng):
    """上端境界からイオン n 個 + 電子 n 個を入射させる。"""
    dx = p.dx
    Lx, Ly = p.nx * dx, ny * dx
    m_i = p.ion_mass_amu * AMU

    # イオン: 垂直入射 + 横方向の熱速度
    xi = rng.uniform(0, Lx, n)
    vyi = -np.sqrt(2 * p.ion_energy_eV * E_CHARGE / m_i) * np.ones(n)
    vxi = rng.normal(0, np.sqrt(p.ion_temp_eV * E_CHARGE / m_i), n)

    # 電子: マクスウェル分布の「フラックス重み付き」サンプリング
    #   面に垂直な成分は Rayleigh 分布 (∝ v exp(-v^2/2s^2)), 平行成分は正規分布
    s = np.sqrt(p.electron_temp_eV * E_CHARGE / M_E)
    xe = rng.uniform(0, Lx, n)
    vxe = rng.normal(0, s, n)
    vye = -s * np.sqrt(-2.0 * np.log(1.0 - rng.uniform(size=n)))

    x = np.concatenate([xi, xe])
    y = np.full(2 * n, Ly - 1e-3 * dx)
    vx = np.concatenate([vxi, vxe])
    vy = np.concatenate([vyi, vye])
    qm = np.concatenate([np.full(n, E_CHARGE / m_i), np.full(n, -E_CHARGE / M_E)])
    spc = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])  # 0=イオン 1=電子
    return x, y, vx, vy, qm, spc


def interp_field(x, y, F, dx):
    """真空セルのみを重みに使った双線形補間 (誘電体内の電場を拾わないため)。
    F[..., 0]=Ex*vac, F[..., 1]=Ey*vac, F[..., 2]=vac"""
    nx, ny = F.shape[:2]
    fx = x / dx - 0.5
    fy = y / dx - 0.5
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx = (fx - i0)[:, None]
    ty = (fy - j0)[:, None]
    i1 = (i0 + 1) % nx
    i0 = i0 % nx
    j1 = np.minimum(np.maximum(j0 + 1, 0), ny - 1)
    j0 = np.minimum(np.maximum(j0, 0), ny - 1)
    acc = ((1 - tx) * (1 - ty) * F[i0, j0] + tx * (1 - ty) * F[i1, j0]
           + (1 - tx) * ty * F[i0, j1] + tx * ty * F[i1, j1])
    w = np.maximum(acc[:, 2], 1e-12)
    return acc[:, 0] / w, acc[:, 1] / w


def trace(x, y, vx, vy, qm, spc, Ex, Ey, vac, p):
    """凍結した電場中で粒子群を追跡。固体に当たった位置のセル数を種別ごとに返す。"""
    nx, ny = vac.shape
    dx = p.dx
    Lx, Ly = nx * dx, ny * dx
    F = np.stack([Ex * vac, Ey * vac, vac.astype(float)], axis=-1)
    hits = [[], []]
    n_escape = np.zeros(2, dtype=np.int64)

    for _ in range(p.max_steps):
        if x.size == 0:
            break
        ex, ey = interp_field(x, y, F, dx)
        ax, ay = qm * ex, qm * ey
        # 1 ステップの移動量が cfl セル以下になるように粒子ごとに dt を決める
        speed = np.hypot(vx, vy)
        dt = p.cfl * dx / (speed + np.sqrt(2.0 * dx * np.hypot(ax, ay)) + 1.0)
        vxn, vyn = vx + ax * dt, vy + ay * dt
        x = (x + 0.5 * (vx + vxn) * dt) % Lx
        y = y + 0.5 * (vy + vyn) * dt
        vx, vy = vxn, vyn

        escaped = y >= Ly
        ix = np.minimum((x / dx).astype(np.int64), nx - 1)
        iy = np.minimum(np.maximum((y / dx).astype(np.int64), 0), ny - 1)
        hit = (~escaped) & (~vac[ix, iy])

        if hit.any():
            flat, s = ix[hit] * ny + iy[hit], spc[hit]
            hits[0].append(flat[s == 0])
            hits[1].append(flat[s == 1])
        if escaped.any():
            n_escape += np.bincount(spc[escaped], minlength=2)

        keep = ~(hit | escaped)
        x, y, vx, vy, qm, spc = x[keep], y[keep], vx[keep], vy[keep], qm[keep], spc[keep]

    n_lost = x.size  # max_steps で打ち切った粒子 (電位の谷に捕まった電子など)
    counts = []
    for k in (0, 1):
        h = np.concatenate(hits[k]) if hits[k] else np.zeros(0, dtype=np.int64)
        counts.append(np.bincount(h, minlength=nx * ny).reshape(nx, ny))
    return counts[0], counts[1], n_escape, n_lost


# ---------------------------------------------------------------- 表面リーク
def surface_cells(solid):
    """8近傍のどこかが真空である固体セル(=表面セル)の mask。x は周期, y 範囲外は固体扱い。"""
    nx, ny = solid.shape
    vp = np.pad(~solid, ((0, 0), (1, 1)), constant_values=False)
    near = np.zeros_like(solid)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            if sx == 0 and sy == 0:
                continue
            near |= np.roll(vp, -sx, axis=0)[:, 1 + sy: 1 + sy + ny]
    return solid & near


def build_leakage(solid, solve, i0, i1, p):
    """表面シート伝導度 sigma_s による表面電荷の移動(陰解法)の行列を作る。
        dQ/dt = G phi,  phi = P Q   ->   Q_new = (I - dt G P)^-1 Q_old
    G: 表面セルを 4 近傍でつないだ抵抗網のコンダクタンス行列 (g = sigma_s/dx [S/m])
    P: 表面セルの単位電荷に対する表面セル電位の応答 (Poisson を表面セル数ぶん解いて作る。solve は make_solver)
    G は半負定値・P は半正定値なので無条件安定。G の列和が 0 なので総電荷は保存される。
    マスク表面も同じ sigma_s でつなぐ。導体マスクの表面セルどうしは等電位なので電流は流れず、
    誘電体表面との境目で電荷が導体に出入りする。"""
    nx, ny = solid.shape
    dx = p.dx
    ix, iy = np.nonzero(surface_cells(solid))
    m = ix.size
    smap = -np.ones((nx, ny), dtype=np.int64)
    smap[ix, iy] = np.arange(m)

    g = p.sigma_s / dx
    G = np.zeros((m, m))
    for a, b in ((smap, np.roll(smap, -1, axis=0)),      # 東隣 (x 周期)
                 (smap[:, :-1], smap[:, 1:])):           # 北隣
        sel = (a >= 0) & (b >= 0)
        a, b = a[sel], b[sel]
        np.add.at(G, (a, b), g)
        np.add.at(G, (b, a), g)
        np.add.at(G, (a, a), -g)
        np.add.at(G, (b, b), -g)

    flat = ix * ny + iy
    B = np.zeros((nx * ny, m))
    B[flat, np.arange(m)] = -1.0 / (dx ** 2 * EPS0)      # 単位電荷 1 C/m を置いた右辺
    P = solve(B)[flat, :]
    M = np.linalg.inv(np.eye(m) - p.dt_batch * (G @ P))

    # 電流モニタ用: トレンチ中腹より下の表面セル(底 + 両側壁の下半分)
    cut = p.floor_t + p.trench_d // 2
    region = (ix >= i0 - 1) & (ix <= i1) & (iy < cut)
    return dict(flat=flat, M=M, region=region)


# ---------------------------------------------------------------- 導電性マスク (浮遊導体)
def conductor_response(solve, cond, dx):
    """浮遊導体に 1 C/m を置いたときの全セルの電位 g [V/(C/m)] (nx x ny)。
    相反定理より g[j] は「セル j に置いた 1 C/m が導体に作る電位」でもある。導体の自己容量は 1 / g[導体]。"""
    b = np.zeros(cond.size)
    b[np.flatnonzero(cond)[0]] = -1.0 / (dx ** 2 * EPS0)   # 導体に 1 C/m を置いた右辺
    return solve(b).reshape(cond.shape)


def conductor_step(n_i, n_e, q, C, Te, drift=0.0):
    """浮遊導体の 1 バッチでの電位変化 [V] を半陰的に求める。
    導体の緩和時間 C / (dI_e/dV) (~数十 μs) は 1 バッチより短いので、当たった電荷をそのまま足す
    (陽解法) と電位が振動する。そこでバッチ中の電位変化 v に対して、イオン電流と drift (周りの誘電体の
    電荷変化が導体の電位を動かす分) は一定、電子電流はボルツマン因子 exp(v/Te) で応答するとして
        dv/dt = (q n_i / C + drift - q n_e / C exp(v/Te)) / dt_batch
    をバッチの終わりまで厳密に積分する (w = exp(-v/Te) について線形な ODE になる)。
    n_i, n_e: このバッチで導体に当たったイオン・電子の数 (マクロ粒子), q: マクロ粒子の電荷 [C/m],
    C: 導体の自己容量 [F/m], Te: 電子温度 [eV], drift [V]。戻り値は drift を含む電位変化。
    変化が 0 になる (定常) のは q (n_i - n_e) / C + drift = 0 のときで、陽解法の定常条件と同じ。"""
    x_i, x_e = q * n_i / C + drift, q * n_e / C      # 電位を上げる/下げる 1 バッチ分の量 [V]
    if n_e == 0:
        return x_i
    k = x_i / Te
    frac = -np.expm1(-k) / k if k != 0 else 1.0      # (1 - e^-k) / k
    return -Te * np.logaddexp(-k, np.log(x_e / Te * frac))


def conductor_surface_charge(rho, phi, A, cond):
    """表示・保存用の電荷密度を返す (rho のコピー)。導体セルの電荷を、等電位の条件から決まる実際の
    分布 (導体表面の誘導電荷) に置き換える。計算中の導体の電荷は合計だけが意味を持ち、1 セルにまとめて
    置いているため。"""
    out = rho.copy()
    if cond.any():
        out[cond] = (-EPS0 * (A @ phi.ravel())).reshape(rho.shape)[cond]
    return out


# ---------------------------------------------------------------- 飽和判定
def _mean_and_se(x):
    """ブロックの平均と、その標準誤差を返す。
    ブロック内の線形トレンドを除いた残差から分散を求め、表面電荷ノイズの時間相関
    (AR(1) 近似の lag-1 自己相関 r1) を有効サンプル数 n*(1-r1)/(1+r1) で補正する。"""
    n = x.size
    t = np.arange(n)
    resid = x - np.polyval(np.polyfit(t, x, 1), t)
    var = resid.var(ddof=2)
    r1 = 0.0
    if var > 0:
        r1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
        r1 = float(np.clip(r1, 0.0, 0.95)) if np.isfinite(r1) else 0.0
    n_eff = max(n * (1 - r1) / (1 + r1), 1.0)
    return x.mean(), np.sqrt(var / n_eff)


def check_steady(hist, probes, p):
    """直近 W バッチ と その前の W バッチ の平均電位を観測点ごとに比較する。
    |平均差| <= max(rtol x 最大電位, 3 x 平均差の標準誤差) を全観測点で満たせば OK。
    (ノイズの大きい観測点が、ノイズ範囲内の変動のせいで判定を妨げないようにするため)
    戻り値: (ok, 最も飽和から遠い観測点の名前, その平均差 [V], その許容値 [V])"""
    W = p.steady_window
    stats = {k: (_mean_and_se(np.asarray(hist[k][-W:])),
                 _mean_and_se(np.asarray(hist[k][-2 * W:-W]))) for k in probes}
    scale = max(1.0, max(abs(c[0]) for c, _ in stats.values()))
    ok, worst = True, (None, 0.0, 1.0, -1.0)
    for k, ((m_c, se_c), (m_p, se_p)) in stats.items():
        d = m_c - m_p
        tol = max(p.steady_rtol * scale, 3.0 * np.hypot(se_c, se_p))
        ok &= abs(d) <= tol
        if abs(d) / tol > worst[3]:
            worst = (k, d, tol, abs(d) / tol)
    return ok, worst[0], worst[1], worst[2]


# ---------------------------------------------------------------- メイン計算
def run(p, log=print, progress=None, stop=None):
    """シミュレーション本体。
    log(str)       : ログ出力先 (既定は print)
    progress(dict) : 毎バッチ後に呼ばれる(GUI の経過表示用)。phi, rho, Ex, Ey, hist などのコピーを渡す
    stop           : is_set() が True になったら、現在のバッチ終了後に中断する (threading.Event など)
    中断した場合も、それまでの結果を返す (1 バッチも終わっていなければ None)。"""
    rng = np.random.default_rng(p.seed)
    solid, mask, i0, i1 = build_geometry(p)
    vac = ~solid
    nx, ny = solid.shape
    dx = p.dx
    conductor = p.mask_type == "conductor" and mask.any()
    cond = mask if conductor else np.zeros_like(mask)     # 導体のセル (導電性マスク)
    A = build_poisson(permittivity(solid, mask, p), cond, p)
    solve = make_solver(A, cond)
    leak = build_leakage(solid, solve, i0, i1, p) if p.sigma_s > 0 else None
    if conductor:
        g_cond = conductor_response(solve, cond, dx)
        C_cond = 1.0 / g_cond[cond][0]                           # 導体の自己容量 [F/m]
        c_cell = np.flatnonzero(surface_cells(solid) & cond)[0]  # 導体の電荷をまとめて置くセル

    # 1 マクロ粒子が運ぶ電荷 [C/m] (奥行き 1 m あたり)
    w_real = p.flux * (nx * dx) * p.dt_batch / p.n_per_batch
    q_macro = E_CHARGE * w_real
    log(f"格子 {nx} x {ny}  (トレンチ幅 {p.trench_w*dx*1e9:.0f} nm, 深さ {p.trench_d*dx*1e9:.0f} nm, "
          f"AR={p.trench_d/p.trench_w:.1f})")
    if p.mask_t > 0:
        kind = f"導体 (浮遊電位, 自己容量 {C_cond*1e12:.1f} pF/m)" if conductor else f"誘電体 eps_r={p.mask_eps_r:g}"
        log(f"マスク: 厚さ {p.mask_t*dx*1e9:.0f} nm, {kind},  開口全体の AR={(p.trench_d + p.mask_t)/p.trench_w:.1f}")
    if conductor and leak is None:
        log("注意: 導電性マスクで表面リークなし (sigma_s=0) だと、マスクと誘電体の境目のすぐ下の誘電体表面に"
            "正電荷が溜まり続け、長時間では非物理的に強い電場になります。sigma_s > 0 との併用を推奨します")
    if leak is not None:
        log(f"表面リーク ON: sigma_s={p.sigma_s:g} S, 表面セル数 {leak['flat'].size}")
    log(f"1 マクロ粒子 = 実粒子 {w_real:.1f} 個/m,  1 バッチ = {p.dt_batch*1e6:.2f} μs")

    # 観測点 (固体の表面セル)。側壁の上/中/下・底は誘電体部分
    iw = i0 - 1                       # 左側壁の表面セル列
    top_ox = p.floor_t + p.trench_d   # 誘電体の上面 (= マスクの下面)
    probes = {"mask top": (i0 - 10, top_ox + p.mask_t - 1)}   # マスクなしなら誘電体の上面
    if p.mask_t > 0 and not conductor:                        # 導体マスクは等電位なので上面だけ見る
        probes["mask sidewall"] = (iw, top_ox + p.mask_t // 2)
    probes.update({
        "sidewall upper":  (iw, top_ox - 4),
        "sidewall middle": (iw, p.floor_t + p.trench_d // 2),
        "sidewall lower":  (iw, p.floor_t + 3),
        "bottom center":   ((i0 + i1) // 2, p.floor_t - 1),
    })
    hist = {"t": [], "ratio_i_bottom": [], "ratio_e_bottom": [], "esc_e": [], "lost": [], "leak": []}
    for k in probes:
        hist[k] = []

    rho = np.zeros((nx, ny))          # 電荷密度 [C/m^3]
    phi = np.zeros((nx, ny))
    Ex = np.zeros((nx, ny)); Ey = np.zeros((nx, ny))
    frac_open = p.trench_w / nx
    t0 = time.time()
    n_max = p.max_batches if p.until_steady else p.n_batches
    n_ok, converged_at, stopped = 0, None, False

    for b in range(n_max):
        x, y, vx, vy, qm, spc = inject(p, p.n_per_batch, ny, rng)
        cnt_i, cnt_e, esc, n_lost = trace(x, y, vx, vy, qm, spc, Ex, Ey, vac, p)

        # 電荷の蓄積 -> Poisson
        rho_prev = rho.copy() if conductor else None
        drho = (cnt_i - cnt_e) * q_macro / dx ** 2
        if conductor:
            drho[cond] = 0.0                # 導体に当たった分は、下でまとめて半陰的に加える
        rho += drho

        # 表面リーク: 表面セルの電荷を表面に沿って移動させる
        leak_ratio = 0.0
        if leak is not None:
            Qs = rho.flat[leak["flat"]] * dx ** 2           # 表面セルの電荷 [C/m]
            Qs_new = leak["M"] @ Qs
            rho.flat[leak["flat"]] = Qs_new / dx ** 2
            dQ = Qs[leak["region"]].sum() - Qs_new[leak["region"]].sum()  # 下半分から上へ流出した正電荷
            leak_ratio = dQ / p.dt_batch / (E_CHARGE * p.flux * p.trench_w * dx)

        if conductor:
            # 導体の電位変化を半陰的に求め、それに対応する導体自身の電荷を 1 セルにまとめて置く
            # (導体の電荷は合計だけが意味を持つ。当たった電荷との差は、電位変化で追い返された/引き込まれた電子の分)
            drift = ((rho - rho_prev) * g_cond).sum() * dx ** 2   # ここまでの電荷変化による導体の電位変化 [V]
            dV = conductor_step(cnt_i[cond].sum(), cnt_e[cond].sum(), q_macro, C_cond,
                                p.electron_temp_eV, drift)
            rho.flat[c_cell] += C_cond * (dV - drift) / dx ** 2

        phi = solve((-rho / EPS0).ravel()).reshape(nx, ny)
        Ex, Ey = compute_field(phi, vac, p)

        # 記録
        hist["t"].append((b + 1) * p.dt_batch)
        for k, (ix, iy) in probes.items():
            hist[k].append(phi[ix, iy])
        norm = p.n_per_batch * frac_open        # トレンチ開口に入射する粒子数
        hist["ratio_i_bottom"].append(cnt_i[i0:i1, p.floor_t - 1].sum() / norm)
        hist["ratio_e_bottom"].append(cnt_e[i0:i1, p.floor_t - 1].sum() / norm)
        hist["esc_e"].append(esc[1] / p.n_per_batch)
        hist["lost"].append(n_lost / (2 * p.n_per_batch))
        hist["leak"].append(leak_ratio)

        if (b + 1) % 10 == 0 or b == 0:
            log(f"[{b+1:4d}/{n_max}] t={hist['t'][-1]*1e6:7.1f} μs  "
                  f"phi(bottom)={hist['bottom center'][-1]:7.2f} V  "
                  f"phi(side mid)={hist['sidewall middle'][-1]:7.2f} V  "
                  f"phi(mask top)={hist['mask top'][-1]:7.2f} V  "
                  f"e/i@bottom={np.mean(hist['ratio_e_bottom'][-10:]) / max(np.mean(hist['ratio_i_bottom'][-10:]), 1e-9):.2f}  "
                  f"lost={np.mean(hist['lost'][-10:])*100:.2f}%  "
                  + (f"leak/Iion={np.mean(hist['leak'][-10:]):.3f}  " if leak is not None else "")
                  + f"({time.time()-t0:.0f}s)")

        # 飽和判定 (継続モード)
        W = p.steady_window
        check_info = None
        if p.until_steady and (b + 1) % W == 0 and (b + 1) >= 2 * W:
            ok, name, d, tol = check_steady(hist, probes, p)
            n_ok = n_ok + 1 if ok else 0
            log(f"  [飽和判定] 最大変化: {name} {d:+.2f} V / {W * p.dt_batch * 1e3:.1f} ms "
                  f"(許容 {tol:.2f} V) -> {'OK' if ok else 'NG'} ({n_ok}/{p.steady_hold})")
            check_info = dict(ok=ok, name=name, d=d, tol=tol, n_ok=n_ok)
            if n_ok >= p.steady_hold:
                converged_at = b + 1

        if progress is not None:
            progress(dict(batch=b + 1, n_max=n_max, elapsed=time.time() - t0,
                          phi=phi.copy(), rho=conductor_surface_charge(rho, phi, A, cond),
                          Ex=Ex.copy(), Ey=Ey.copy(),
                          hist={k: np.array(v) for k, v in hist.items()},
                          check=check_info, converged=converged_at is not None,
                          probes=list(probes), t_conv=np.nan))
        if converged_at is not None:
            break
        if stop is not None and stop.is_set():
            stopped = True
            log("ユーザー操作により中断しました")
            break

    if not hist["t"]:
        return None
    hist = {k: np.array(v) for k, v in hist.items()}

    # 飽和値: 直近 W バッチの平均
    Wl = min(p.steady_window, len(hist["t"]))
    steady = {k: float(hist[k][-Wl:].mean()) for k in probes}
    if p.until_steady:
        if converged_at:
            log(f"飽和に到達: t = {converged_at * p.dt_batch * 1e3:.1f} ms ({converged_at} バッチ)")
        elif not stopped:
            log(f"警告: 上限 {n_max} バッチ ({n_max * p.dt_batch * 1e3:.0f} ms) までに飽和しませんでした "
                  f"(--max_batches を増やしてください)")
    log(f"電位 (直近 {Wl} バッチ平均): " + ",  ".join(f"{k} {v:.1f} V" for k, v in steady.items()))
    t_conv = converged_at * p.dt_batch if converged_at else np.nan
    return dict(phi=phi, rho=conductor_surface_charge(rho, phi, A, cond), Ex=Ex, Ey=Ey,
                solid=solid, mask=mask, i0=i0, i1=i1, hist=hist, probes=probes,
                steady=steady, t_conv=t_conv, stopped=stopped)


# ---------------------------------------------------------------- 可視化
def _subplots(ncols, figsize, pyplot):
    """pyplot=True: plt.subplots (plt.show() で表示可能)。
    pyplot=False: Figure を直接作る (GUI アプリ内から呼んでもウィンドウが増えず、メモリも溜まらない)。"""
    if pyplot:
        return plt.subplots(1, ncols, figsize=figsize)
    from matplotlib.figure import Figure
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(1, ncols)


def draw_outline(ax, xc, yc, solid, mask, color):
    """固体の輪郭 (実線) とマスクの範囲 (破線) を描く。"""
    ax.contour(xc, yc, solid.T.astype(float), levels=[0.5], colors=color, linewidths=0.8)
    if mask.any():
        ax.contour(xc, yc, mask.T.astype(float), levels=[0.5], colors=color, linewidths=0.8,
                   linestyles="--")


def draw_mask_level(ax, y):
    """側壁プロファイルの図に、マスクの下面 (誘電体との境目) の高さ y [nm] を示す。"""
    ax.axhline(y, color="0.5", ls="--", lw=0.8)
    ax.text(0.02, y, " mask", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8, color="0.4")


def mask_label(p):
    """図のタイトル用のマスクの説明。"""
    if p.mask_t == 0:
        return "no mask"
    kind = "conductor" if p.mask_type == "conductor" else f"dielectric, eps_r={p.mask_eps_r:g}"
    return f"mask {p.mask_t * p.dx * 1e9:.0f} nm ({kind})"


def plot_results(res, p, out, pyplot=True):
    phi, rho, Ex, Ey, solid = res["phi"], res["rho"], res["Ex"], res["Ey"], res["solid"]
    mask, i0, hist = res["mask"], res["i0"], res["hist"]
    nx, ny = solid.shape
    dxn = p.dx * 1e9
    xc = (np.arange(nx) + 0.5) * dxn
    yc = (np.arange(ny) + 0.5) * dxn
    extent = [0, nx * dxn, 0, ny * dxn]

    # ---- 場の図 ----
    fig, axs = _subplots(3, (14, 7.5), pyplot)
    vmax = max(np.abs(phi).max(), 1e-9)
    im = axs[0].imshow(phi.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    draw_outline(axs[0], xc, yc, solid, mask, "k")
    axs[0].set_title("Potential [V]")
    fig.colorbar(im, ax=axs[0], shrink=0.6)

    vac = ~solid
    Emag = np.ma.masked_where(solid, np.hypot(Ex, Ey) / 1e6)
    im = axs[1].imshow(Emag.T, origin="lower", extent=extent, cmap="viridis")
    draw_outline(axs[1], xc, yc, solid, mask, "w")
    s = 3
    X, Y = np.meshgrid(xc[::s], yc[::s], indexing="ij")
    ex, ey, m = Ex[::s, ::s], Ey[::s, ::s], vac[::s, ::s]
    mag = np.hypot(ex, ey) + 1e-30
    axs[1].quiver(X[m], Y[m], (ex / mag)[m], (ey / mag)[m], color="w", scale=45, width=0.003)
    axs[1].set_title("|E| [MV/m] and direction")
    fig.colorbar(im, ax=axs[1], shrink=0.6)

    # 左側壁 (誘電体 + マスク) の表面電位・表面電荷密度 (高さ方向)
    top_ox = p.floor_t + p.trench_d
    iy = np.arange(p.floor_t, top_ox + p.mask_t)
    iw = i0 - 1
    ax = axs[2]
    ax.plot(phi[iw, iy], (iy + 0.5) * dxn, "r-", label="surface potential [V]")
    ax.set_xlabel("Potential [V]", color="r")
    ax.set_ylabel("y [nm]")
    ax.set_title("Left sidewall profile")
    ax2 = ax.twiny()
    ax2.plot(rho[iw, iy] * p.dx * 1e3, (iy + 0.5) * dxn, "b--", label="surface charge")
    ax2.set_xlabel("Surface charge [mC/m$^2$]", color="b")
    if p.mask_t > 0:
        draw_mask_level(ax, top_ox * dxn)
    ax.set_ylim(extent[2], extent[3])
    ax.grid(alpha=0.3)
    for a in axs[:2]:
        a.set_xlabel("x [nm]"); a.set_ylabel("y [nm]")
    leak_label = f"surface sheet conductance = {p.sigma_s:g} S" if p.sigma_s > 0 else "no surface leakage"
    fig.suptitle(f"{mask_label(p)},  {leak_label}")
    fig.tight_layout()
    fig.savefig(f"{out}_fields.png", dpi=130)

    # ---- 時間履歴 ----
    t = hist["t"] * 1e6
    ncol = 3 if p.sigma_s > 0 else 2
    fig, axs = _subplots(ncol, (6.5 * ncol, 4.5), pyplot)
    for k in res["probes"]:
        axs[0].plot(t, hist[k], label=k)
    if np.isfinite(res["t_conv"]):
        for a in axs:
            a.axvline(res["t_conv"] * 1e6, color="k", ls=":", lw=1)
        axs[0].text(res["t_conv"] * 1e6, axs[0].get_ylim()[1], " steady", va="top", ha="right", fontsize=8)
    axs[0].set_xlabel("time [μs]"); axs[0].set_ylabel("surface potential [V]")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    k = min(10, max(1, len(t) // 5))
    ma = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    axs[1].plot(t[k - 1:], ma(hist["ratio_i_bottom"]), label="ion  (bottom)")
    axs[1].plot(t[k - 1:], ma(hist["ratio_e_bottom"]), label="electron (bottom)")
    axs[1].set_xlabel("time [μs]")
    axs[1].set_ylabel("flux at trench bottom / incident flux")
    axs[1].legend(); axs[1].grid(alpha=0.3)
    if p.sigma_s > 0:
        axs[2].plot(t[k - 1:], ma(hist["leak"]))
        axs[2].set_xlabel("time [μs]")
        axs[2].set_ylabel("surface leakage current\n/ ion current into opening")
        axs[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}_history.png", dpi=130)


def save_results(res, p, out, pyplot=False):
    """図 2 枚 (<out>_fields.png, <out>_history.png) と <out>.npz を保存する。"""
    plot_results(res, p, out, pyplot=pyplot)
    np.savez(f"{out}.npz", phi=res["phi"], rho=res["rho"], Ex=res["Ex"], Ey=res["Ey"],
             solid=res["solid"], mask=res["mask"], t_conv=res["t_conv"],
             **{f"hist_{k}": v for k, v in res["hist"].items()},
             **{f"steady_{k}": v for k, v in res["steady"].items()})
    return [f"{out}_fields.png", f"{out}_history.png", f"{out}.npz"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = {"mask_type": MASK_TYPES}
    for k, v in asdict(Params()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v, choices=choices.get(k))
    ap.add_argument("--out", default="trench_charging")
    ap.add_argument("--show", action="store_true", help="図をウィンドウにも表示する")
    args = vars(ap.parse_args())
    out, show = args.pop("out"), args.pop("show")
    p = Params(**args)

    if not show:
        matplotlib.use("Agg")
    res = run(p)
    files = save_results(res, p, out, pyplot=show)
    print("保存: " + ", ".join(files))
    if show:
        plt.show()


if __name__ == "__main__":
    main()
