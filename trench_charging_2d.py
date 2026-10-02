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
* 構造 (既定): 接地Si基板(y=0) の上に誘電体(SiO2, eps_r=3.9)があり、その上にマスク(既定 300 nm)を載せる。
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
    - イオン  : 垂直入射 + 横方向の熱速度 Ti。エネルギーは bias="rf" (既定) なら RF バイアスの
                プラズマ等価回路 (plasma_circuit.py: RF 電源 → ブロッキングコンデンサ → ウェハ → SiO2 →
                マスク → シース → プラズマ → シース → 接地) を RF 周期定常まで解き、そのシース電圧で
                1 次元のシースを時間発展させて求めた IED (sheath_ied.py, ied_model="sheath") から選ぶ。
                電源の波形は正弦波 (既定) か、負に凸の矩形パルス (rf_wave="pulse", 幅 rf_duty) 。
                bias="dc" なら全イオンが ion_energy_eV。
                既定モードは IED だけを使う (一方向の連成)。field_mode="circuit" は指定位相の回路電位を
                基板・マスクへ適用し、Te/2 で入射したイオンを領域内で加速する瞬時電場の近似。
                field_mode="circuit_rf" は全周期の入射位相をサンプリングし、飛行中も回路電位を更新する。
                表面電荷はバッチごとに周期平均電流で更新し、電位履歴・飽和判定も周期平均で行う。
    - 電子    : 温度 Te のマクスウェル分布のフラックス分布(等方的)
    - 電子フラックス = イオンフラックス (フローティング表面の平均的な電流バランス。RF でも周期平均で成り立つ)
* Poisson: div(eps_r grad phi) = -rho/eps0 を有限差分(セル中心, 面で調和平均)で解く。
          行列は形状固定なので LU 分解を 1 回だけ行う。
* 粒子が固体セルに入ったら完全吸収(付着確率 1, イオン反射なし)。
* 表面リーク: 誘電体の表面セル(真空に接するセル)を抵抗網でつなぎ、電位差に応じて
    表面に沿って電荷が移動する(シート伝導度 sigma_s [S], 既定 1e-15 S, 0 で無効)。
    dQ/dt = G phi (G: 表面セル間のコンダクタンス行列, phi = P Q は Poisson の応答)
    を後退オイラーで解くので、sigma_s が大きくても無条件安定で総電荷も保存される。

使い方
------
依存パッケージは uv で管理している (pyproject.toml / uv.lock)。uv run が初回に .venv を作る。

    uv run trench_charging_2d.py                 # 既定値で実行 (約4分, 30 ms 分の帯電)
    uv run trench_charging_2d.py --n_batches 60  # 短時間で動作確認
    uv run trench_charging_2d.py --trench_d 120  # アスペクト比を変える
    uv run trench_charging_2d.py --sigma_s 1e-14 # 表面リークを強くする (既定 1e-15 S, 0 でリークなし)
    uv run trench_charging_2d.py --until_steady  # 飽和帯電に達するまで継続 (上限 --max_batches)
    uv run trench_charging_2d.py --resume_result charged.npz --n_batches 100  # 100 バッチ追加
    uv run trench_charging_2d.py --resume_result charged.npz --until_steady --max_batches 5000  # 通算上限5000
    uv run trench_charging_2d.py --mask_t 40     # マスク厚 40 セル = 200 nm (0 でマスクなし)
    uv run trench_charging_2d.py --trench_taper_deg 2 --mask_taper_deg 5  # 鉛直からのテーパー角 [deg]
    uv run trench_charging_2d.py --mask_type dielectric --mask_eps_r 3.0  # 絶縁性のマスク
    uv run trench_charging_2d.py --rf_freq 2e6 --rf_volt 150  # RF バイアスの周波数 [Hz] と振幅 [V]
    uv run trench_charging_2d.py --rf_wave pulse --rf_freq 1e6 --rf_duty 0.1  # 負の矩形パルス (幅 10%)
    uv run trench_charging_2d.py --bias dc       # RF 回路を使わず、全イオンを ion_energy_eV にする
    uv run trench_charging_2d.py --help          # 変更できるパラメータ一覧

飽和まで継続モード (--until_steady)
------------------------------------
steady_window バッチごとに全観測点の電位と、左右側壁・底・マスク・全表面の蓄積電荷を比較する。
電荷はブロック平均が安定し、正味の充電のドリフトがノイズ以下になることも必要。
電位と電荷の両方が steady_hold 回連続で条件を満たしたときに終了する。
この場合 n_batches は使われず、max_batches が上限になる。

出力: <out>_fields.png, <out>_history.png, <out>_charge.png, <out>.npz
      RF なら <out>_rf.png、--vertical_probe で飽和後の軌道比較 <out>_ions.png も。
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib
import matplotlib.pyplot as plt

import plasma_circuit as PC
import trench_rf as RF

# ---------------------------------------------------------------- 物理定数
E_CHARGE = 1.602176634e-19   # [C]
EPS0 = 8.8541878128e-12      # [F/m]
M_E = 9.1093837015e-31       # [kg]
AMU = 1.66053906660e-27      # [kg]

MASK_TYPES = ("conductor", "dielectric")  # mask_type の選択肢
WALL_MODELS = ("barrier", "absorb")       # wall_model の選択肢
FIELD_MODES = ("floating", "fixed", "circuit", "circuit_rf")
CIRCUIT_FIELDS = ("circuit", "circuit_rf")
SIDE_BCS = ("periodic", "pillar", "fixed")
BACKENDS = ("cpu", "cuda", "auto")


@dataclass
class Params:
    # ---- 格子・形状 ----
    dx: float = 5e-9          # セルサイズ [m]
    nx: int = 60              # 横方向セル数 (周期境界)
    trench_w: int = 20        # SiO2 上面でのトレンチ幅 [セル] (20 -> 100 nm)
    trench_offset: int = 0    # 中央からの水平移動 [セル] (正=右)
    trench_taper_deg: float = 0.0  # 鉛直からの側壁角 [deg]、正=底ほど狭い
    trench_d: int = 80        # トレンチ深さ [セル] (80 -> 400 nm, AR=4)
    floor_t: int = 10         # トレンチ底の誘電体厚 [セル]
    n_vac: int = 50           # 構造の上面 (マスク上面) より上の真空領域 [セル]
    eps_r: float = 3.9        # 誘電体の比誘電率 (SiO2)
    top_bc: str = "dirichlet"  # 上端境界: "dirichlet"(phi=0, プラズマ電位) or "neumann"
    # ---- マスク (下面の開口を SiO2 上面の開口に接続する) ----
    mask_t: int = 60          # マスク厚 [セル] (60 -> 300 nm, 0 でマスクなし)
    mask_taper_deg: float = 0.0  # 鉛直からの側壁角 [deg]、正=上面ほど広い
    mask_type: str = "conductor"  # "conductor"(導電性: doped carbon など, 浮遊導体) or "dielectric"
    mask_eps_r: float = 3.0   # マスクの比誘電率 (mask_type="dielectric" のときだけ使う)
    # ---- 外部電位 (circuit: 位相固定 / circuit_rf: 全周期、飛行中も時間発展) ----
    field_mode: str = "floating"  # floating / fixed / circuit / circuit_rf
    rf_phase: float = 0.0     # circuit: 固定位相 / circuit_rf: 結果の表示位相 [deg]
    rf_time_steps: int = 256  # circuit_rf: 粒子の dt 上限を RF 周期 / この値とする
    substrate_voltage: float = 0.0  # fixed: 基板電位 [V]
    mask_voltage: float = 0.0       # fixed: 導電性マスク電位 [V]
    top_voltage: float = 0.0        # fixed/floating: 上端電位 [V]
    side_bc: str = "periodic" # pillar: SiO2 両端を基板→マスク電位で線形補間
    left_voltage: float = 0.0 # side_bc=fixed: SiO2 左端電位 [V]
    right_voltage: float = 0.0 # side_bc=fixed: SiO2 右端電位 [V]
    # ---- 飽和後の垂直イオン診断 (電荷は更新しない) ----
    probe_ions: int = 1000    # 開口幅に一様な垂直イオン数
    probe_energy_eV: float = 100.0  # 診断の入射エネルギー [eV]
    probe_trajectories: int = 40     # 保存・表示する軌道数の上限
    # ---- プラズマ ----
    flux: float = 1e20        # 粒子フラックス [m^-2 s^-1] (= 1e16 cm^-2 s^-1)
    ion_mass_amu: float = 40.0  # イオン質量 [amu] (Ar+)
    ion_energy_eV: float = 100.0  # イオン入射エネルギー(シース電圧) [eV] (bias="dc" のとき)
    ion_temp_eV: float = 0.5  # イオンの横方向温度 [eV] -> 角度広がり
    electron_temp_eV: float = 3.0  # 電子温度 [eV]
    # ---- RF バイアス (プラズマ等価回路, plasma_circuit.py) ----
    bias: str = "rf"          # "rf": 等価回路のシース電圧から IED を作る / "dc": 全イオンが ion_energy_eV
    rf_freq: float = 13.56e6  # RF 周波数 [Hz]
    rf_volt: float = 100.0    # RF 電源の振幅 [V] (pulse ではパルスの高さ)
    rf_wave: str = "sine"     # RF 電源の波形: "sine" (正弦波) / "pulse" (0 V から -rf_volt へ下がる負の矩形パルス)
    rf_duty: float = 0.1      # pulse: パルス幅 (半分の高さで測る) の周期に対する割合
    rf_rise: float = 1e-9     # pulse: 立ち下がり・立ち上がりの時間 [s]
    c_block: float = 4900e-12  # ブロッキングコンデンサ [F]
    wafer_d: float = 0.3      # ウェハ (電極) の直径 [m]
    wall_ratio: float = 5.0   # 接地側 (壁) の面積 / ウェハの面積
    ied_model: str = "sheath"  # IED: "sheath" (1 次元シースの時間発展) / "transit" (通過時間で平均) / "instant"
    gas_pressure: float = 1.0  # 背景ガス (イオンと同じ原子, Ar) の圧力 [Pa]。1 次元シースの衝突 (南部–北谷モデル), 0 で衝突なし
    gas_temp: float = 300.0    # 背景ガスの温度 [K]
    # ---- 表面リーク ----
    sigma_s: float = 1e-15    # 表面シート伝導度 [S] (0=リークなし)。目安: 1e-16〜1e-13
    #                           導電性マスクでは、マスク境目の電荷を逃がすため 0 より大きくする
    # ---- 飽和まで継続モード ----
    until_steady: bool = False  # True: 電位・表面電荷の両方が飽和するまで継続
    max_batches: int = 3000   # 継続モードの通算上限バッチ数 (3000 -> 300 ms)
    steady_window: int = 30   # 飽和判定のブロック長 [バッチ]
    steady_rtol: float = 0.005  # ブロック間の電位変化の許容値 (最大電位に対する割合)
    steady_charge_rtol: float = 0.005  # 各部位のブロック平均電荷の相対許容値
    steady_charge_atol: float = 1e-15  # 電荷の絶対許容値 [C/m] (0.001 pC/m)
    steady_hold: int = 2      # 判定を連続で満たすべき回数
    # ---- 数値 ----
    backend: str = "cpu"      # cpu: 従来 / cuda: CUDA 必須 / auto: 利用できれば CUDA
    cuda_device: int = 0      # NVIDIA GPU のデバイス番号
    n_batches: int = 300      # 固定長のバッチ数。再開時は追加で計算するバッチ数
    n_per_batch: int = 3000   # 1 バッチあたりの粒子数 (イオン, 電子 それぞれ)
    dt_batch: float = 1e-4    # 1 バッチが表す物理時間 [s] (大きすぎると電位が振動する)
    cfl: float = 0.35         # 1 ステップで進む距離の上限 [セル]
    max_steps: int = 3000     # 1 粒子あたりの最大ステップ数 (超えたら打ち切り=捕捉粒子)
    wall_model: str = "barrier"  # 壁際: "barrier" (壁面までの電位の山を越えられない粒子は反射) / "absorb" (すべて吸収)
    seed: int = 1


# ---------------------------------------------------------------- 形状
def opening_widths(p):
    """連続形状の開口幅 [セル]。幅の基準面は SiO2 上面 (マスク下面)。"""
    return dict(bottom=p.trench_w - 2*p.trench_d*np.tan(np.radians(p.trench_taper_deg)),
                interface=float(p.trench_w),
                top=p.trench_w + 2*p.mask_t*np.tan(np.radians(p.mask_taper_deg)))


def build_geometry(p):
    """solid[ix, iy] (True=固体) と mask[ix, iy] (True=マスク, solid の一部) を作る。iy=0 が基板側。
    SiO2 上面の開口幅 trench_w を基準に、鉛直からの角度で両側壁を傾ける。
    正の角度は下ほど狭い。セル中心が開口内なら真空とする階段状の格子近似。
    i0, i1 は基準面の開口座標であり、テーパー時の底・入口は opening_span から得る。"""
    if (any(not isinstance(getattr(p, k), (int, np.integer))
            for k in ("nx", "trench_w", "trench_offset", "trench_d", "floor_t", "n_vac", "mask_t"))
            or p.nx < 3
            or not 1 <= p.trench_w <= p.nx - 2 or min(p.floor_t, p.trench_d, p.n_vac) < 1):
        raise ValueError("形状のセル数は正の整数、トレンチ幅は nx-2 以下にしてください")
    i0 = (p.nx - p.trench_w) // 2 + p.trench_offset
    i1 = i0 + p.trench_w
    if i0 < 1 or i1 > p.nx - 1:
        raise ValueError("トレンチ移動後も左右に少なくとも 1 セルの固体を残してください")
    if p.mask_t < 0:
        raise ValueError(f"mask_t は 0 以上にしてください: {p.mask_t}")
    if p.mask_type not in MASK_TYPES:
        raise ValueError(f"mask_type は {' / '.join(MASK_TYPES)} のどちらかにしてください: {p.mask_type!r}")
    for key in ("trench_taper_deg", "mask_taper_deg"):
        angle = getattr(p, key)
        if not np.isfinite(angle) or not -90 < angle < 90:
            raise ValueError(f"{key}: テーパー角は -90° より大きく 90° 未満の有限値にしてください")
    center = (i0+i1)/2
    for level, width in opening_widths(p).items():
        label = {"bottom": "トレンチの底", "interface": "SiO2 上面", "top": "マスク上面"}[level]
        if width < 1:
            raise ValueError(f"{label}の開口幅が 1 セル未満です。幅・深さ・テーパー角を調整してください")
        if center-width/2 < 1 or center+width/2 > p.nx-1:
            raise ValueError(f"{label}のテーパーが計算領域を超えます。左右に 1 セル以上の固体を残してください")
    top_ox = p.floor_t + p.trench_d           # 誘電体の上面 (= マスクの下面)
    ny = top_ox + p.mask_t + p.n_vac
    solid = np.zeros((p.nx, ny), dtype=bool)
    solid[:, : top_ox + p.mask_t] = True
    y = np.arange(p.floor_t, top_ox+p.mask_t) + 0.5
    slope = np.tan(np.radians(np.where(y < top_ox, p.trench_taper_deg, p.mask_taper_deg)))
    half_width = p.trench_w/2 + (y-top_ox)*slope
    x = (np.arange(p.nx)+0.5)[:, None]
    opening = (x >= center-half_width) & (x < center+half_width)
    solid[:, p.floor_t:top_ox+p.mask_t] = ~opening
    mask = solid.copy()
    mask[:, :top_ox] = False
    return solid, mask, i0, i1


def opening_span(solid, row):
    """指定高さにおける実際の格子開口 [左端, 右端)。"""
    cells = np.flatnonzero(~solid[:, row])
    return int(cells[0]), int(cells[-1]+1)


def wall_bounds(solid, p):
    """各高さで真空開口に隣接する左・右の固体セル列。"""
    iy = np.arange(p.floor_t, p.floor_t+p.trench_d+p.mask_t)
    vac = ~solid[:, iy]
    left = np.argmax(vac, axis=0)-1
    right = solid.shape[0]-np.argmax(vac[::-1], axis=0)
    return left, right, iy


def permittivity(solid, mask, p):
    """セルごとの比誘電率 (真空 1, 誘電体 eps_r, 誘電体マスク mask_eps_r)。
    導体マスクのセルの値は使わない (build_poisson で導体として扱う)。"""
    eps = np.where(solid, p.eps_r, 1.0)
    return np.where(mask, p.mask_eps_r, eps) if p.mask_type == "dielectric" else eps


# ---------------------------------------------------------------- Poisson
def build_poisson(eps, cond, p):
    """div(eps_r grad phi) の疎行列を作る (外部電位の右辺は setup_fields で与える)。
    x は周期、side_bc が periodic 以外なら SiO2 両端のみ Dirichlet。上下端は設定による。
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
    top_ox = p.floor_t + p.trench_d
    if p.side_bc != "periodic":
        ce[-1, :top_ox] = 0.0                        # SiO2 内では周期結合を外す
    cw = np.roll(ce, 1, axis=0)                                                      # 西側の面

    c_int = face(eps[:, :-1], eps[:, 1:], cond[:, :-1], cond[:, 1:]) / dx2          # 上下の内部面
    cn = np.zeros((nx, ny)); cn[:, :-1] = c_int
    cs = np.zeros((nx, ny)); cs[:, 1:] = c_int

    diag_n = cn.copy()
    diag_s = cs.copy()
    diag_s[:, 0] += 2.0 * eps[:, 0] / dx2              # 下端: 基板面まで半セル
    if p.top_bc == "dirichlet":
        diag_n[:, -1] += 2.0 * eps[:, -1] / dx2        # 上端: phi = 0 まで半セル
    diag = -(ce + cw + diag_n + diag_s)
    if p.side_bc != "periodic":
        diag[0, :top_ox] -= 2.0 * eps[0, :top_ox] / dx2
        diag[-1, :top_ox] -= 2.0 * eps[-1, :top_ox] / dx2

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


def circuit_voltages(c, phase):
    """周期補間した瞬時電位。プラズマ電位を 0 V とするゲージで返す。"""
    t = np.asarray(c["t"])
    period = (t[1] - t[0]) * t.size
    at = (phase % 360.0) / 360.0 * period
    values = {k: float(np.interp(at, np.append(t, period), np.append(c[k], c[k][0])))
              for k in ("uw", "um", "up", "V1")}
    return dict(substrate=values["uw"] - values["up"], mask=values["um"] - values["up"],
                top=0.0, sheath=values["V1"])


def _boundary_rhs(p, eps, volts, fixed_sides=True):
    rhs = np.zeros_like(eps)
    rhs[:, 0] -= 2.0 * eps[:, 0] / p.dx ** 2 * volts["substrate"]
    if p.top_bc == "dirichlet":
        rhs[:, -1] -= 2.0 * eps[:, -1] / p.dx ** 2 * volts["top"]
    top_ox = p.floor_t + p.trench_d
    if p.side_bc == "pillar":
        yfrac = (np.arange(top_ox) + 0.5) / top_ox
        vl = vr = volts["substrate"] + (volts["mask"] - volts["substrate"]) * yfrac
    else:
        vl, vr = (p.left_voltage, p.right_voltage) if fixed_sides else (0.0, 0.0)
    if p.side_bc != "periodic":
        rhs[0, :top_ox] -= 2.0 * eps[0, :top_ox] / p.dx ** 2 * vl
        rhs[-1, :top_ox] -= 2.0 * eps[-1, :top_ox] / p.dx ** 2 * vr
    return rhs


def setup_fields(p, solid, mask, circuit=None):
    """電荷に対する線形応答と、指定境界電位による電位分布を分離して作る。"""
    if p.field_mode not in FIELD_MODES or p.side_bc not in SIDE_BCS:
        raise ValueError("電場モードまたは左右境界条件が正しくありません")
    if p.top_bc not in ("dirichlet", "neumann"):
        raise ValueError("top_bc は dirichlet / neumann のどちらかにしてください")
    numeric = (p.rf_phase, p.substrate_voltage, p.mask_voltage, p.top_voltage,
               p.left_voltage, p.right_voltage)
    if not np.isfinite(numeric).all():
        raise ValueError("電位と RF 位相には有限の値を指定してください")
    driven = p.field_mode != "floating"
    if driven and mask.any() and p.mask_type != "conductor":
        raise ValueError("マスク電位を指定するには mask_type=conductor にしてください")
    if p.side_bc == "pillar" and not driven:
        raise ValueError("ピラー境界には field_mode=fixed / circuit / circuit_rf が必要です")
    if p.field_mode in CIRCUIT_FIELDS:
        if p.bias != "rf" or circuit is None or p.top_bc != "dirichlet":
            raise ValueError("回路電位モードには bias=rf と top_bc=dirichlet が必要です")
        volts = circuit_voltages(circuit, p.rf_phase)
    else:
        volts = dict(substrate=p.substrate_voltage, mask=p.mask_voltage, top=p.top_voltage)
    if p.field_mode == "circuit_rf":
        if not isinstance(p.rf_time_steps, (int, np.integer)) or p.rf_time_steps < 16:
            raise ValueError("RF 周期の時間分割数は 16 以上の整数にしてください")
        t = np.asarray(circuit["t"])
        period = (t[1] - t[0]) * t.size
        if (not np.isfinite(period) or period <= 0 or not np.isclose(t[0], 0, atol=period*1e-12)
                or not np.allclose(np.diff(t), period / t.size, rtol=1e-10, atol=period*1e-12)):
            raise ValueError("RF 回路の波形には 0 から始まる等間隔の 1 周期を使ってください")
        if not np.isfinite(p.dt_batch) or p.dt_batch < period * (1 - 1e-12):
            raise ValueError("RF 周期平均モードでは dt_batch を RF 1 周期以上にしてください")
        volt_wave = np.column_stack((circuit["uw"] - circuit["up"], circuit["um"] - circuit["up"]))
        mean = volt_wave.mean(axis=0)
        volts = dict(substrate=float(mean[0]), mask=float(mean[1]), top=0.0,
                     sheath=float(np.mean(circuit["V1"])))
    cond = mask if p.mask_type == "conductor" else np.zeros_like(mask)
    eps = permittivity(solid, mask, p)
    A = build_poisson(eps, cond, p)
    if driven and cond.any():
        free = ~cond.ravel()
        lu = splu(A[free][:, free].tocsc())

        def solve(b):
            out = np.zeros_like(b, dtype=float)
            out[free] = lu.solve(b[free])
            return out

        phi_cond = cond.ravel().astype(float) * volts["mask"]
        mask_rhs = -(A @ phi_cond)
    else:
        solve = make_solver(A, cond)
        phi_cond = np.zeros(cond.size)
        mask_rhs = np.zeros(cond.size)
    rhs = _boundary_rhs(p, eps, volts)
    external = (solve(rhs.ravel() + mask_rhs) + phi_cond).reshape(eps.shape)
    fields = dict(A=A, solve=solve, cond=cond, eps=eps, external=external,
                  boundary_rhs=rhs, volts=volts, floating=bool(cond.any() and not driven), rf=None)
    if p.field_mode == "circuit_rf":
        # 電荷と幾何形状が一定なら、瞬時場は基板・マスクの単位電圧応答の線形結合。
        basis, rhs_basis = [], []
        for vs, vm in ((1.0, 0.0), (0.0, 1.0)):
            brhs = _boundary_rhs(p, eps, dict(substrate=vs, mask=vm, top=0.0), fixed_sides=False)
            bc = cond.ravel().astype(float) * vm
            basis.append((solve(brhs.ravel() - A @ bc) + bc).reshape(eps.shape))
            rhs_basis.append(brhs)
        basis = np.array(basis)
        electric = [compute_field(v, ~solid, p, 0.0) for v in basis]
        dt_max = period / p.rf_time_steps
        if p.rf_wave == "pulse":
            dt_max = min(dt_max, p.rf_rise / 8)
        fields["rf"] = dict(period=period, wave=volt_wave - mean, phi=basis,
                            Ex=np.array([v[0] for v in electric]), Ey=np.array([v[1] for v in electric]),
                            dt_max=dt_max, boundary_rhs=np.array(rhs_basis))
    return fields


def field_snapshot(fields, mean_phi, p):
    """RF 全周期モードの表示だけを指定位相へ切り替える。履歴・追跡の基準は周期平均。"""
    if fields["rf"] is None:
        return mean_phi, fields["external"], fields["boundary_rhs"], fields["volts"]
    rf = fields["rf"]
    delta = RF.coefficients(rf, (p.rf_phase % 360) / 360 * rf["period"])
    shift = np.einsum("i,ijk->jk", delta, rf["phi"])
    rhs = fields["boundary_rhs"] + np.einsum("i,ijk->jk", delta, rf["boundary_rhs"])
    volts = {**fields["volts"], "substrate": fields["volts"]["substrate"] + float(delta[0]),
             "mask": fields["volts"]["mask"] + float(delta[1])}
    return mean_phi + shift, fields["external"] + shift, rhs, volts


def _grad(c, m, pl, vm, vp, dx):
    """真空セルだけを使った差分で -dphi/dx を返す (誘電体界面をまたがない)。"""
    g = np.zeros_like(c)
    both = vm & vp
    g = np.where(both, (pl - m) / (2 * dx), g)
    g = np.where(vp & ~vm, (pl - c) / dx, g)
    g = np.where(vm & ~vp, (c - m) / dx, g)
    return -g


def compute_field(phi, vac, p, top_voltage=None):
    """電場 E = -grad(phi) をセル中心で計算 (真空セルのみ有効)。"""
    dx = p.dx
    Ex = _grad(phi, np.roll(phi, 1, 0), np.roll(phi, -1, 0),
               np.roll(vac, 1, 0), np.roll(vac, -1, 0), dx)

    vtop = p.top_voltage if top_voltage is None else top_voltage
    ghost_top = 2 * vtop - phi[:, -1] if p.top_bc == "dirichlet" else phi[:, -1]
    ghost_bot = 2 * p.substrate_voltage - phi[:, 0]
    pD = np.concatenate([ghost_bot[:, None], phi[:, :-1]], axis=1)
    pU = np.concatenate([phi[:, 1:], ghost_top[:, None]], axis=1)
    f = np.zeros((phi.shape[0], 1), dtype=bool)
    vD = np.concatenate([f, vac[:, :-1]], axis=1)
    vU = np.concatenate([vac[:, 1:], ~f], axis=1)
    Ey = _grad(phi, pD, pU, vD, vU, dx)
    return Ex, Ey


# ---------------------------------------------------------------- 粒子
def inject(p, n, ny, rng, ion_energy=None, ion_velocity=None):
    """上端境界からイオン n 個 + 電子 n 個を入射させる。
    ion_velocity(rng, n) -> (vx, vy): イオンの速度を選ぶ関数 (1 次元シースで電極に着いたイオンの組。衝突による
        角度の広がりも含む)。ion_energy(rng, n): イオンエネルギー [eV] を選ぶ関数 (transit / instant の IED)。
    どちらもなければ全イオンが ion_energy_eV。エネルギーで選ぶときは、横方向の熱速度 (ion_temp_eV) を足す。"""
    dx = p.dx
    Lx, Ly = p.nx * dx, ny * dx
    m_i = p.ion_mass_amu * AMU

    # イオン: 垂直入射 + 横方向の熱速度
    xi = rng.uniform(0, Lx, n)
    if ion_velocity is not None:
        vxi, vyi = ion_velocity(rng, n)
    else:
        Ei = p.ion_energy_eV * np.ones(n) if ion_energy is None else ion_energy(rng, n)
        vyi = -np.sqrt(2 * Ei * E_CHARGE / m_i)
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


def _wall_entry(x0, y0, sx, sy, vac, dx):
    """固体セルに入った粒子 (移動前の位置 x0, y0 と移動量 sx, sy) について、最初に入った固体セル (cx, cy)、
    そこへ入る直前の真空セル (vx_, vy_)、横切った面が x 向きか (face_x) を返す。
    1 ステップの移動は 1 セル未満なので、斜めの移動では x と y の境界のどちらを先に横切ったかで決める。"""
    nx, ny = vac.shape
    ix0 = np.minimum((x0 / dx).astype(np.int64), nx - 1)
    iy0 = np.minimum(np.maximum((y0 / dx).astype(np.int64), 0), ny - 1)
    xu, yu = x0 + sx, y0 + sy                            # 周期で折り返さない移動後の位置
    dix = np.floor(xu / dx).astype(np.int64) - ix0
    diy = np.minimum(np.maximum(np.floor(yu / dx).astype(np.int64), 0), ny - 1) - iy0
    with np.errstate(divide="ignore", invalid="ignore"):
        fx = np.where(dix != 0, ((ix0 + (dix > 0)) * dx - x0) / sx, np.inf)   # x の境界を横切る時刻 (割合)
        fy = np.where(diy != 0, ((iy0 + (diy > 0)) * dx - y0) / sy, np.inf)
    x_first = fx < fy
    # 先に横切った境界の向こうのセル
    ax_ = np.where(x_first, (ix0 + dix) % nx, ix0)
    ay_ = np.where(x_first, iy0, iy0 + diy)
    diag = (dix != 0) & (diy != 0)
    first_solid = ~vac[ax_, ay_]
    # 斜めの移動で、先に入ったセルが真空なら、もう一方の境界を横切って最後のセルに入った
    last_x, last_y = (ix0 + dix) % nx, iy0 + diy
    use_last = diag & ~first_solid
    cx = np.where(use_last, last_x, ax_)
    cy = np.where(use_last, last_y, ay_)
    vx_ = np.where(use_last, ax_, ix0)
    vy_ = np.where(use_last, ay_, iy0)
    face_x = np.where(use_last, ~x_first, x_first)
    return cx, cy, vx_, vy_, face_x


def trace_cpu(x, y, vx, vy, qm, spc, Ex, Ey, vac, p, barrier=None, record=None, stop=None, rf=None):
    """粒子群を追跡。rf があれば飛行中にも周期的な電極電圧を加える。
    barrier=None (wall_model="absorb"): 固体セルに入った粒子はすべて吸収する。
    barrier=(phi, wface) (wall_model="barrier"): 壁面の電位 φ_面 = φ_真空 + wface (φ_固体 - φ_真空) までの
        電位の山を、壁に垂直な運動エネルギーで越えられない粒子 (m v_n^2 / 2 < q (φ_面 - φ_真空)) は鏡面反射させる。
        電場は真空セルだけから作るので、真空セルの中心から壁面までの半セル分の電位差をここで扱う。
        wface = ε/(1+ε) (誘電体: 電束の連続から), 導体は 1 (導体表面 = 面の位置)。
    record=dict を渡すと、診断用の間引いた軌道・全粒子の到達点・速度・状態を格納する。
    stop は診断用の中断イベント。既定の帯電計算では軌道を保存しない。"""
    nx, ny = vac.shape
    dx = p.dx
    Lx, Ly = nx * dx, ny * dx
    tapered = bool(p.trench_taper_deg or (p.mask_t > 0 and p.mask_taper_deg))
    F = np.stack([Ex * vac, Ey * vac, vac.astype(float)], axis=-1)
    if rf is not None:
        RF.validate(rf, len(x), vac.shape)
        clock = np.asarray(rf["initial_time"]).copy() % rf["period"]
        elapsed = np.zeros(len(x))
        rf_F = [np.stack([a * vac, b * vac, vac.astype(float)], axis=-1)
                for a, b in zip(rf["Ex"], rf["Ey"])]
        voltage_bound = np.abs(rf["wave"]).max(axis=0)
    hits = [[], []]
    n_escape = np.zeros(2, dtype=np.int64)
    if record is not None:
        ids = np.arange(x.size)
        selected = np.unique(np.linspace(0, x.size - 1, min(p.probe_trajectories, x.size), dtype=int))
        slots = np.full(x.size, -1, dtype=int)
        slots[selected] = np.arange(selected.size)
        paths = [[(x[k], y[k])] for k in selected]
        record.update(selected=selected, status=np.full(x.size, 3, dtype=np.int8),
                      hit_cells=np.full((x.size, 2), -1, dtype=int),
                      initial_x=x.copy(), endpoints=np.column_stack((x, y)),
                      final_velocity=np.column_stack((vx, vy)))
        if rf is not None:
            record.update(launch_phase=clock / rf["period"] * 360, final_phase=clock / rf["period"] * 360,
                          flight_time=elapsed.copy())

    for _ in range(p.max_steps):
        if x.size == 0:
            break
        if stop is not None and stop.is_set():
            break
        ex, ey = interp_field(x, y, F, dx)
        ax, ay = qm * ex, qm * ey
        # 1 ステップの移動量が cfl セル以下になるように粒子ごとに dt を決める
        speed = np.hypot(vx, vy)
        if rf is None:
            dt = p.cfl * dx / (speed + np.sqrt(2.0 * dx * np.hypot(ax, ay)) + 1.0)
        else:
            basis_e = np.array([interp_field(x, y, f, dx) for f in rf_F])
            # 電圧が急変しても CFL を守るよう、1 周期内の加速度の上界を使う。
            bound_x = np.abs(ex) + voltage_bound @ np.abs(basis_e[:, 0])
            bound_y = np.abs(ey) + voltage_bound @ np.abs(basis_e[:, 1])
            accel_bound = np.abs(qm) * np.hypot(bound_x, bound_y)
            dt = np.minimum(rf["dt_max"], p.cfl * dx / (speed + np.sqrt(2 * dx * accel_bound) + 1))
            coeff = RF.coefficients(rf, clock + dt / 2)
            ex = ex + np.sum(coeff.T * basis_e[:, 0], axis=0)
            ey = ey + np.sum(coeff.T * basis_e[:, 1], axis=0)
            ax, ay = qm * ex, qm * ey
            clock = (clock + dt) % rf["period"]
            elapsed = elapsed + dt
        vxn, vyn = vx + ax * dt, vy + ay * dt
        x0, y0 = x, y
        sx, sy = 0.5 * (vx + vxn) * dt, 0.5 * (vy + vyn) * dt
        x = (x + sx) % Lx
        y = y + sy
        vx, vy = vxn, vyn

        escaped = y >= Ly
        ix = np.minimum((x / dx).astype(np.int64), nx - 1)
        iy = np.minimum(np.maximum((y / dx).astype(np.int64), 0), ny - 1)
        hit = (~escaped) & (~vac[ix, iy])
        if tapered:
            # 段差の角では、真空セル間の斜めの移動でも途中で固体面を横切り得る。
            ix0 = np.minimum((x0 / dx).astype(np.int64), nx-1)
            iy0 = np.minimum(np.maximum((y0 / dx).astype(np.int64), 0), ny-1)
            h = np.flatnonzero(~hit & ~escaped & (ix != ix0) & (iy != iy0))
            if h.size:
                cx, cy, _, _, _ = _wall_entry(x0[h], y0[h], sx[h], sy[h], vac, dx)
                hit[h] = ~vac[cx, cy]

        if barrier is not None and hit.any():
            phi, wface = barrier
            h = np.flatnonzero(hit)
            cx, cy, vcx, vcy, face_x = _wall_entry(x0[h], y0[h], sx[h], sy[h], vac, dx)
            vn = np.where(face_x, vx[h], vy[h])
            d_phi = wface[cx, cy] * (phi[cx, cy] - phi[vcx, vcy])       # φ_面 - φ_真空
            if rf is not None:
                coeff = RF.coefficients(rf, clock[h])
                delta_phi = rf["phi"][:, cx, cy] - rf["phi"][:, vcx, vcy]
                d_phi += wface[cx, cy] * np.sum(coeff.T * delta_phi, axis=0)
            refl = 0.5 * vn ** 2 < qm[h] * d_phi                         # 電位の山を越えられない
            r = h[refl]
            rx = face_x[refl]
            vx[r] = np.where(rx, -vx[r], vx[r])
            vy[r] = np.where(rx, vy[r], -vy[r])
            x[r], y[r] = x0[r], y0[r]                                    # 移動前の位置 (真空) に戻す
            hit[r] = False
            ix[h], iy[h] = cx, cy                                        # 吸収は最初に入った固体セルに
        if (record is not None or tapered) and barrier is None and hit.any():
            h = np.flatnonzero(hit)
            cx, cy, _, _, _ = _wall_entry(x0[h], y0[h], sx[h], sy[h], vac, dx)
            ix[h], iy[h] = cx, cy
        if hit.any():
            flat, s = ix[hit] * ny + iy[hit], spc[hit]
            hits[0].append(flat[s == 0])
            hits[1].append(flat[s == 1])
        if escaped.any():
            n_escape += np.bincount(spc[escaped], minlength=2)

        if record is not None:
            xp, yp = x.copy(), y.copy()
            time_fraction = np.ones(x.size) if rf is not None else None
            if hit.any():
                h = np.flatnonzero(hit)
                cx, cy, _, _, fx = _wall_entry(x0[h], y0[h], sx[h], sy[h], vac, dx)
                with np.errstate(divide="ignore", invalid="ignore"):
                    xb = (cx + (sx[h] < 0)) * dx
                    xb += np.round((x0[h] - xb) / Lx) * Lx
                    tx = (xb - x0[h]) / sx[h]
                    ty = ((cy + (sy[h] < 0)) * dx - y0[h]) / sy[h]
                frac = np.clip(np.where(fx, tx, ty), 0, 1)
                if rf is not None:
                    time_fraction[h] = frac
                xp[h] = (x0[h] + frac * sx[h]) % Lx
                yp[h] = y0[h] + frac * sy[h]
                record["status"][ids[h]] = 1
                record["hit_cells"][ids[h]] = np.column_stack((cx, cy))
            record["status"][ids[escaped]] = 2
            if escaped.any():
                h = np.flatnonzero(escaped)
                frac = np.clip((Ly-y0[h])/sy[h], 0, 1)
                if rf is not None:
                    time_fraction[h] = frac
                xp[h] = (x0[h] + frac*sx[h]) % Lx
                yp[h] = Ly
            record["endpoints"][ids] = np.column_stack((xp, yp))
            record["final_velocity"][ids] = np.column_stack((vx, vy))
            if rf is not None:
                record["flight_time"][ids] = elapsed - (1-time_fraction) * dt
                record["final_phase"][ids] = ((clock - (1-time_fraction) * dt) % rf["period"]) / rf["period"] * 360
            tracked = np.flatnonzero(slots[ids] >= 0)
            for k in tracked:
                paths[slots[ids[k]]].append((xp[k], yp[k]))

        keep = ~(hit | escaped)
        if record is not None:
            ids = ids[keep]
        x, y, vx, vy, qm, spc = x[keep], y[keep], vx[keep], vy[keep], qm[keep], spc[keep]
        if rf is not None:
            clock, elapsed = clock[keep], elapsed[keep]

    n_lost = x.size  # max_steps で打ち切った粒子 (電位の谷に捕まった電子など)
    counts = []
    for k in (0, 1):
        h = np.concatenate(hits[k]) if hits[k] else np.zeros(0, dtype=np.int64)
        counts.append(np.bincount(h, minlength=nx * ny).reshape(nx, ny))
    if record is not None:
        record["paths"] = [np.asarray(v) for v in paths]
        record["cancelled"] = stop is not None and stop.is_set()
    return counts[0], counts[1], n_escape, n_lost


def make_particle_tracer(p, vac, log=None):
    """実行開始時にバックエンドを選ぶ。CUDA 指定の失敗は明示し、auto のみ CPU へ戻る。"""
    if p.backend not in BACKENDS:
        raise ValueError(f"backend は {' / '.join(BACKENDS)} から選んでください")
    if p.backend != "cpu":
        try:
            from trench_cuda import CudaTracer
            tracer = CudaTracer(vac, p.cuda_device)
        except Exception as exc:
            if p.backend == "cuda":
                raise RuntimeError(f"CUDA を初期化できませんでした: {exc}") from exc
            if log is not None:
                log(f"CUDA を利用できないため CPU で計算します: {exc}")
        else:
            if log is not None:
                log(f"粒子追跡: CUDA ({tracer.name}, device {p.cuda_device}, float64)")
            return tracer, "cuda"
    if log is not None:
        log("粒子追跡: CPU (NumPy, float64)")
    return trace_cpu, "cpu"


def trace(x, y, vx, vy, qm, spc, Ex, Ey, vac, p, barrier=None, record=None, stop=None, rf=None):
    """CPU/CUDA 共通の追跡 API。連続したバッチには make_particle_tracer で追跡器を再利用する。"""
    tracer, _ = make_particle_tracer(p, vac)
    return tracer(x, y, vx, vy, qm, spc, Ex, Ey, vac, p, barrier, record, stop, rf=rf)


# ---------------------------------------------------------------- 表面リーク
def surface_cells(solid, vacuum=None):
    """8近傍のどこかが真空である固体セル(=表面セル)の mask。x は周期, y 範囲外は固体扱い。"""
    nx, ny = solid.shape
    vp = np.pad(~solid if vacuum is None else vacuum, ((0, 0), (1, 1)), constant_values=False)
    near = np.zeros_like(solid)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            if sx == 0 and sy == 0:
                continue
            near |= np.roll(vp, -sx, axis=0)[:, 1 + sy: 1 + sy + ny]
    return solid & near


def inner_surface_cells(solid, p):
    """開口内部に露出した固体セル。階段の水平面も含み、外側の上面は除く。"""
    inner_vac = ~solid.copy()
    inner_vac[:, :p.floor_t] = False
    inner_vac[:, p.floor_t+p.trench_d+p.mask_t:] = False
    return surface_cells(solid, inner_vac)


def wall_profiles(solid, p):
    """左右の内壁を高さ順にたどる表面セル座標 (SiO2 とマスク、水平段差を含む)。"""
    ix, iy = np.indices(solid.shape)
    surface = inner_surface_cells(solid, p) & (iy >= p.floor_t)
    center = (p.nx-p.trench_w)//2 + p.trench_offset + p.trench_w/2
    profiles = {}
    for side, half in (("left", ix+0.5 < center), ("right", ix+0.5 >= center)):
        x, y = np.nonzero(surface & half)
        order = np.lexsort((x if side == "left" else -x, y))
        profiles[side] = (x[order], y[order])
    return profiles


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
    region = inner_surface_cells(solid, p)[ix, iy] & (iy < cut)
    return dict(flat=flat, M=M, G=G, region=region)


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


def conductor_surface_charge(rho, phi, A, cond, boundary_rhs=None):
    """表示・保存用の電荷密度を返す (rho のコピー)。導体セルの電荷を、等電位の条件から決まる実際の
    分布 (導体表面の誘導電荷) に置き換える。計算中の導体の電荷は合計だけが意味を持ち、1 セルにまとめて
    置いているため。"""
    out = rho.copy()
    if cond.any():
        induced = (A @ phi.ravel()).reshape(rho.shape)
        if boundary_rhs is not None:
            induced = induced - boundary_rhs
        out[cond] = (-EPS0 * induced)[cond]
    return out


# ---------------------------------------------------------------- 飽和判定
STEADY_CRITERION_VERSION = 2


def charge_monitors(solid, mask, i0, i1, p):
    """生の蓄積電荷を観測するセル群。RF 駆動導体の瞬時誘導電荷は含めない。"""
    exposed = surface_cells(solid)
    dielectric = exposed & ~(mask if p.mask_type == "conductor" else np.zeros_like(mask))
    ix, iy = np.indices(solid.shape)
    wall = (iy >= p.floor_t) & (iy < p.floor_t + p.trench_d)
    inner = inner_surface_cells(solid, p)
    center = (i0+i1)/2
    b0, b1 = opening_span(solid, p.floor_t)
    regions = dict(left=(ix+0.5 < center) & wall & inner,
                   right=(ix+0.5 >= center) & wall & inner,
                   bottom=(ix >= b0) & (ix < b1) & (iy == p.floor_t-1), surface=dielectric)
    for side in ("left", "right"):
        for band, cells in enumerate(np.array_split(np.arange(p.floor_t, p.floor_t+p.trench_d), 3)):
            if len(cells):
                regions[f"{side} band {band+1}"] = regions[side] & np.isin(iy, cells)
    if mask.any():
        regions["mask"] = mask  # 浮遊導体は1セルにまとめた総電荷、駆動導体は常に0
    return {key: np.flatnonzero(region) for key, region in regions.items()}


def sample_charge_monitors(rho, regions, dx):
    values = {key: float(rho.ravel()[cells].sum() * dx**2) for key, cells in regions.items()}
    values["surface absolute"] = float(np.abs(rho.ravel()[regions["surface"]]).sum() * dx**2)
    return values


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


def check_steady(hist, probes, p, reference=None):
    """直近 W バッチ と その前の W バッチ の平均電位を観測点ごとに比較する。
    |平均差| <= max(rtol x 最大電位, 3 x 平均差の標準誤差) を全観測点で満たせば OK。
    (ノイズの大きい観測点が、ノイズ範囲内の変動のせいで判定を妨げないようにするため)
    戻り値: (ok, 最も飽和から遠い観測点の名前, その平均差 [V], その許容値 [V])"""
    W = p.steady_window
    stats = {k: (_mean_and_se(np.asarray(hist[k][-W:])),
                 _mean_and_se(np.asarray(hist[k][-2 * W:-W]))) for k in probes}
    scale = max(1.0, max(abs(c[0] - (reference[k] if reference is not None else 0.0))
                         for k, (c, _) in stats.items()))
    ok, worst = True, (None, 0.0, 1.0, -1.0)
    for k, ((m_c, se_c), (m_p, se_p)) in stats.items():
        d = m_c - m_p
        tol = max(p.steady_rtol * scale, 3.0 * np.hypot(se_c, se_p))
        ok &= abs(d) <= tol
        if abs(d) / tol > worst[3]:
            worst = (k, d, tol, abs(d) / tol)
    return ok, worst[0], worst[1], worst[2]


def check_charge_steady(charge_hist, p):
    """電荷の平均差と、差分電荷から求めた正味の充電ドリフトを両方確認する。

    相対変化だけでは Q(t)=a*t の変化率が 1/t になり、増加中にも飽和と判定される。
    直近2Wバッチの ΔQ の平均を W バッチ分へ換算し、3標準誤差または絶対許容値以下
    になることも要求する。ドリフトの許容値は蓄積電荷の大きさでは緩めない。
    """
    W = p.steady_window
    if (not np.isfinite(p.steady_charge_rtol) or p.steady_charge_rtol <= 0
            or not np.isfinite(p.steady_charge_atol) or p.steady_charge_atol < 0
            or W < 5 or not np.isfinite(p.dt_batch) or p.dt_batch <= 0):
        raise ValueError("電荷の相対許容値とバッチ時間は有限の正の値、絶対許容値は有限の0以上、ブロック長は5以上にしてください")
    if not charge_hist or any(len(q) < 2*W for q in charge_hist.values()):
        raise ValueError("電荷の飽和判定には2ブロック分の履歴が必要です")
    ok, worst = True, None
    for key, values in charge_hist.items():
        q = np.asarray(values[-2*W:], dtype=float)
        if q.ndim != 1 or not np.isfinite(q).all():
            raise ValueError("電荷履歴には有限の値が必要です")
        cur, se_cur = _mean_and_se(q[-W:])
        prev, se_prev = _mean_and_se(q[:W])
        difference = float(cur-prev)
        mean_tol = max(p.steady_charge_atol, p.steady_charge_rtol*max(abs(cur), abs(prev)),
                       3*np.hypot(se_cur, se_prev))
        current, se_current = _mean_and_se(np.diff(q))
        drift = float(current*W)
        drift_tol = max(p.steady_charge_atol, float(3*se_current*W))
        for kind, d, tol in (("mean", difference, mean_tol), ("drift", drift, drift_tol)):
            passed = abs(d) <= tol
            ratio = abs(d)/tol if tol > 0 else (0.0 if d == 0 else np.inf)
            ok &= passed
            if worst is None or ratio > worst["ratio"]:
                worst = dict(name=key, kind=kind, d=d, tol=float(tol), ratio=float(ratio),
                             rate=float(current/p.dt_batch), unit="C/m")
    return {**worst, "ok": bool(ok)}


# ---------------------------------------------------------------- 再開状態
CHECKPOINT_VERSION = 1
RESUME_EDITABLE = {"until_steady", "n_batches", "max_batches", "steady_window", "steady_rtol",
                   "steady_charge_rtol", "steady_charge_atol", "steady_hold", "backend", "cuda_device",
                   "probe_ions", "probe_energy_eV", "probe_trajectories"}
STEADY_SETTINGS = ("until_steady", "steady_window", "steady_rtol", "steady_charge_rtol",
                   "steady_charge_atol", "steady_hold")


def resume_params(res, p):
    """物理条件の変更を確認し、表示単位の往復による丸めは保存値に戻す。"""
    from dataclasses import replace
    source = Params(**res["params"])
    editable = RESUME_EDITABLE | ({"rf_phase"} if source.field_mode == "circuit_rf" else set())
    changed = []
    for k, v in asdict(source).items():
        entered = getattr(p, k)
        equal = np.isclose(entered, v, rtol=1e-9, atol=0) if isinstance(v, float) else entered == v
        if k not in editable and not equal:
            changed.append(k)
    if changed:
        raise ValueError("続きから計算する場合、形状・物理条件・粒子条件は保存結果と同じにしてください: "
                         + ", ".join(changed))
    return replace(source, **{k: getattr(p, k) for k in editable})


def validate_resume(res, p):
    """保存結果の形状・物理条件・バッチ境界を確認し、既計算バッチ数を返す。"""
    p = resume_params(res, p)
    source = Params(**res["params"])
    solid, mask, _, _ = build_geometry(p)
    if not np.array_equal(res["solid"], solid) or not np.array_equal(res["mask"], mask):
        raise ValueError("再開する結果の形状と設定が一致しません")
    rho = np.asarray(res["rho_raw"])
    if rho.shape != solid.shape or not np.isfinite(rho).all():
        raise ValueError("再開する結果の蓄積電荷の形状または数値が不正です")
    t = np.asarray(res.get("hist", {}).get("t", []))
    n = t.size if t.ndim == 1 else 0
    if (n == 0 or not np.isfinite(t).all()
            or not np.allclose(t, np.arange(1, n+1)*source.dt_batch, rtol=1e-10, atol=source.dt_batch*1e-8)):
        raise ValueError("再開には、バッチ境界の時刻を含む連続した計算履歴が必要です")
    for family in ("hist", "charge_hist", "charge_monitor_hist"):
        for name, values in res.get(family, {}).items():
            a = np.asarray(values)
            if a.ndim != 1 or len(a) != n or np.isinf(a).any():
                raise ValueError(f"再開する履歴の長さまたは数値が不正です: {family}/{name}")
    checkpoint = res.get("checkpoint")
    if checkpoint and (checkpoint.get("version") != CHECKPOINT_VERSION
                       or checkpoint.get("batches_completed") != n):
        raise ValueError("保存された再開状態のバージョンまたはバッチ数が一致しません")
    if checkpoint and (not isinstance(checkpoint.get("steady_n_ok"), int)
                       or not 0 <= checkpoint["steady_n_ok"] <= source.steady_hold):
        raise ValueError("保存された飽和判定の連続 OK 回数が不正です")
    if source.bias == "rf" and res.get("circuit") is None:
        raise ValueError("RF 計算の再開には保存済みの回路波形・入射分布が必要です")
    if p.until_steady and p.max_batches <= n:
        raise ValueError(f"上限バッチ数は通算です。既計算の {n} バッチより大きくしてください")
    if not p.until_steady and p.n_batches < 1:
        raise ValueError("再開時の追加バッチ数は1以上にしてください")
    return n


def restore_rngs(res, p, ny, rng, phase_rng, ion_energy, ion_velocity, log):
    """旧 NPZ は入射の抽出だけを再実行し、粒子追跡をせずに乱数の続きへ進める。"""
    checkpoint = res.get("checkpoint")
    if checkpoint:
        try:
            rng.bit_generator.state = checkpoint["rng_state"]
            if phase_rng is not None:
                phase_rng.bit_generator.state = checkpoint["phase_rng_state"]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("保存された乱数状態を復元できません") from e
        log("保存された乱数状態から再開します")
        return
    n = len(res["hist"]["t"])
    log(f"旧保存結果の乱数状態を {n} バッチ分の入射抽出から復元します (粒子追跡は再実行しません)")
    for _ in range(n):
        inject(p, p.n_per_batch, ny, rng, ion_energy, ion_velocity)
        if phase_rng is not None:
            for _ in range(2):
                RF.launch_times(phase_rng, p.n_per_batch, 1/p.rf_freq)
    log("乱数状態の復元が完了しました")


def steady_history_start(hist, probes, charge_hist):
    """旧結果にない観測履歴は NaN とし、全観測値が揃う区間だけで判定する。"""
    first = 0
    for values in [*(hist[k] for k in probes), *charge_hist.values()]:
        missing = np.flatnonzero(~np.isfinite(values))
        if missing.size:
            first = max(first, int(missing[-1])+1)
    return first


# ---------------------------------------------------------------- メイン計算
def run(p, log=print, progress=None, stop=None, use_cache=True, resume=None):
    """シミュレーション本体。
    log(str)       : ログ出力先 (既定は print)
    progress(dict) : 毎バッチ後に呼ばれる(GUI の経過表示用)。phi, rho, Ex, Ey, hist などのコピーを渡す
    stop           : is_set() が True になったら、現在のバッチ終了後に中断する (threading.Event など)
    use_cache      : RF バイアスの等価回路と IED の結果を .cache/circuit に保存・再利用する
    resume         : run / load_results の返す結果。電荷・履歴・乱数を引き継ぐ。
                     固定長では n_batches を追加、継続モードでは max_batches を通算上限とする。
    中断した場合も、それまでの結果を返す (1 バッチも終わっていなければ None)。"""
    if resume is not None:
        p = resume_params(resume, p)
    if p.bias not in PC.BIAS_TYPES:
        raise ValueError(f"bias は {' / '.join(PC.BIAS_TYPES)} のどちらかにしてください: {p.bias!r}")
    if p.until_steady and (p.steady_window < 5 or p.steady_hold < 1 or p.steady_rtol <= 0
                          or not np.isfinite(p.steady_charge_rtol) or p.steady_charge_rtol <= 0
                          or not np.isfinite(p.steady_charge_atol) or p.steady_charge_atol < 0):
        raise ValueError("飽和判定のブロック長は5以上、連続回数・相対許容値は正、電荷絶対許容値は0以上にしてください")
    start_batch = validate_resume(resume, p) if resume is not None else 0
    rng = np.random.default_rng(p.seed)
    solid, mask, i0, i1 = build_geometry(p)
    vac = ~solid
    particle_tracer, backend_used = make_particle_tracer(p, vac, log)
    nx, ny = solid.shape
    dx = p.dx
    if p.field_mode in CIRCUIT_FIELDS and p.bias != "rf":
        raise ValueError("回路電位モードには bias=rf が必要です")
    if p.bias == "rf" and resume is None:
        log("RF 等価回路の電位波形を準備しています…" if p.field_mode in CIRCUIT_FIELDS else "RF 等価回路と入射分布を準備しています…")
    circuit = (resume.get("circuit") if resume is not None else
               PC.solve_circuit_cached(p, use_cache, log, include_ied=p.field_mode not in CIRCUIT_FIELDS)
               if p.bias == "rf" else None)
    if resume is not None and circuit is not None:
        log("保存済みの RF 回路波形・入射分布を再利用します")
    fields = setup_fields(p, solid, mask, circuit)
    phase_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 0x5246])) if fields["rf"] else None
    cond, eps, A, solve = (fields[k] for k in ("cond", "eps", "A", "solve"))
    conductor = fields["floating"]
    driven = p.field_mode != "floating"
    if p.wall_model not in WALL_MODELS:
        raise ValueError(f"wall_model は {' / '.join(WALL_MODELS)} のどちらかにしてください: {p.wall_model!r}")
    wface = np.where(cond, 1.0, eps / (1.0 + eps))     # 壁面の電位 = φ_真空 + wface (φ_固体 - φ_真空)
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
    log(f"トレンチ中心 {(i0+i1)/2*dx*1e9:.1f} nm (移動 {p.trench_offset:+d} セル), "
        f"電場モード {p.field_mode}, SiO2 左右境界 {p.side_bc}")
    b0, b1 = opening_span(solid, p.floor_t)
    e0, e1 = opening_span(solid, p.floor_t+p.trench_d+p.mask_t-1)
    if p.trench_taper_deg or (p.mask_t > 0 and p.mask_taper_deg):
        widths = opening_widths(p)
        log(f"テーパー (鉛直から): SiO2 {p.trench_taper_deg:g} deg / マスク {p.mask_taper_deg:g} deg; "
            f"設計開口 底 {widths['bottom']*dx*1e9:.2f} nm / SiO2 上面 {p.trench_w*dx*1e9:g} nm / "
            f"入口 {widths['top']*dx*1e9:.2f} nm (格子上の底 {(b1-b0)*dx*1e9:g} nm / 入口 {(e1-e0)*dx*1e9:g} nm)")
    if driven:
        log(f"{'周期平均' if fields['rf'] else '境界'}電位: 基板 {fields['volts']['substrate']:.3f} V, "
            f"マスク {fields['volts']['mask']:.3f} V, 上端 {fields['volts']['top']:.3f} V"
            + (f", RF 位相 {p.rf_phase % 360:g} deg (瞬時値を凍結)" if p.field_mode == "circuit" else ""))
    if p.mask_t > 0:
        kind = (f"導体 (浮遊電位, 自己容量 {C_cond*1e12:.1f} pF/m)" if conductor else
                "導体 (RF 電位で駆動)" if cond.any() and fields["rf"] else
                "導体 (外部電位固定)" if cond.any() else f"誘電体 eps_r={p.mask_eps_r:g}")
        log(f"マスク: 厚さ {p.mask_t*dx*1e9:.0f} nm, {kind},  開口全体の AR={(p.trench_d + p.mask_t)/p.trench_w:.1f}")
    if conductor and leak is None:
        log("注意: 導電性マスクで表面リークなし (sigma_s=0) だと、マスクと誘電体の境目のすぐ下の誘電体表面に"
            "正電荷が集中し、絶縁破壊を超える非物理的に強い電場になります (wall_model=absorb では溜まり続けます)。"
            "sigma_s > 0 との併用を推奨します")
    if leak is not None:
        log(f"表面リーク ON: sigma_s={p.sigma_s:g} S, 表面セル数 {leak['flat'].size}")
    ion_energy, ion_velocity = None, None
    if p.bias == "rf":
        if circuit["cached"]:
            log("等価回路の結果はキャッシュから読み込みました")
        if p.field_mode not in CIRCUIT_FIELDS:
            if "vn" in circuit:                     # 1 次元シース: 速度の組 (角度の広がりを含む) から選ぶ
                ion_velocity = PC.ion_velocity_sampler(circuit)
            else:
                ion_energy = PC.ion_energy_sampler(circuit)
        log(f"RF バイアスの電源: {PC.source_label(p)}, C_b {p.c_block*1e12:g} pF, "
            f"ウェハ {p.wafer_d*1e3:g} mm, 壁/ウェハ面積比 {p.wall_ratio:g}")
        log(PC.summary(circuit))
        if p.field_mode not in CIRCUIT_FIELDS and p.ied_model != "sheath" and p.gas_pressure > 0:
            log("注意: ガスとの衝突 (gas_pressure) は ied_model=sheath のときだけ扱います")
    if p.field_mode in CIRCUIT_FIELDS:
        # 上端をプラズマ電位に置いたので、シース加速後の IED を再び入射させない。
        # 領域内で Bohm エネルギー Te/2 から加速する。
        ion_velocity = None
        ion_energy = lambda rng, n: np.full(n, p.electron_temp_eV / 2)
        log("回路電場: 上端から Te/2 のイオンを入射し、領域内で加速します (IED の二重加速を防止)。"
            "シース長は上部真空領域で近似しています")
    if fields["rf"] is not None:
        log(f"RF 全周期: 各バッチの入射を 0–360 deg に層化し、飛行中も電位を更新します。"
            f"粒子 dt 上限 {fields['rf']['dt_max']*1e9:.4g} ns、表示位相 {p.rf_phase%360:g} deg")
        log("表面電荷はバッチ内で凍結し、周期平均の付着電流とリークで更新します。"
            "電位履歴・飽和判定は RF 周期平均です (ガス衝突・シース空間電荷は 2D 領域では扱いません)")
    log(f"1 マクロ粒子 = 実粒子 {w_real:.1f} 個/m,  1 バッチ = {p.dt_batch*1e6:.2f} μs")
    if resume is not None:
        restore_rngs(resume, p, ny, rng, phase_rng, ion_energy, ion_velocity, log)
        log(f"続きから計算: {start_batch} バッチ、t={start_batch*p.dt_batch*1e3:g} ms の蓄積電荷を引き継ぎます")

    # 観測点 (固体の表面セル)。側壁の上/中/下・底は誘電体部分
    top_ox = p.floor_t + p.trench_d   # 誘電体の上面 (= マスクの下面)
    left, right, _ = wall_bounds(solid, p)
    probes = {"mask top": (max(0, e0 - 10), top_ox + p.mask_t - 1)}
    if p.mask_t > 0 and not cond.any():
        y_mask = top_ox + p.mask_t // 2
        probes["mask sidewall"] = (int(left[y_mask-p.floor_t]), y_mask)
    for level, y_wall in (("upper", max(p.floor_t, top_ox-4)),
                          ("middle", p.floor_t+p.trench_d//2),
                          ("lower", min(top_ox-1, p.floor_t+3))):
        probes[f"sidewall {level}"] = (int(left[y_wall-p.floor_t]), y_wall)
    probes["bottom center"] = ((b0+b1)//2, p.floor_t-1)
    for level in ("upper", "middle", "lower"):
        y_wall = probes[f"sidewall {level}"][1]
        probes[f"right sidewall {level}"] = (int(right[y_wall-p.floor_t]), y_wall)
    steady_reference = ({k: float(fields["external"][ix, iy]) for k, (ix, iy) in probes.items()}
                        if fields["rf"] is not None else None)
    hist = {"t": [], "ratio_i_bottom": [], "ratio_e_bottom": [], "esc_e": [], "lost": [], "leak": []}
    for k in probes:
        hist[k] = []
    charge_hist = {"left": [], "right": []}  # SiO2 側壁の積分電荷 [C/m]
    q_regions = charge_monitors(solid, mask, i0, i1, p)
    charge_monitor_hist = {k: [] for k in (*q_regions, "surface absolute")}

    rho = np.zeros((nx, ny))          # 電荷密度 [C/m^3]
    phi = fields["external"].copy()
    if resume is not None:
        rho = np.array(resume["rho_raw"], dtype=float, copy=True)
        # RF の表示位相の瞬時場ではなく、蓄積電荷と周期平均の外部電位から再構成する。
        phi = solve((-rho/EPS0).ravel()).reshape(nx, ny) + fields["external"]
        for key in hist:
            hist[key] = np.asarray(resume["hist"].get(key, np.full(start_batch, np.nan))).tolist()
            if key in probes and key not in resume["hist"]:
                ix, iy = probes[key]
                hist[key][-1] = float(phi[ix, iy])
        initial_charge = sample_charge_monitors(rho, q_regions, dx)
        for key in charge_hist:
            charge_hist[key] = np.asarray(resume.get("charge_hist", {}).get(key, np.full(start_batch, np.nan))).tolist()
            if key not in resume.get("charge_hist", {}):
                charge_hist[key][-1] = initial_charge[key]
        for key in charge_monitor_hist:
            saved_q = resume.get("charge_monitor_hist", {}).get(key)
            if saved_q is None:
                saved_q = charge_hist[key] if key in charge_hist else np.full(start_batch, np.nan)
            charge_monitor_hist[key] = np.asarray(saved_q).tolist()
            charge_monitor_hist[key][-1] = initial_charge[key]
    Ex, Ey = compute_field(phi, vac, p, fields["volts"]["top"])
    frac_open = (b1-b0) / nx  # 底面の平均フラックスを入射フラックスで規格化
    t0 = time.time()
    n_max = p.max_batches if p.until_steady else start_batch+p.n_batches
    n_ok, converged_at, stopped = 0, None, False
    history_start = steady_history_start(hist, probes, charge_monitor_hist)
    if resume is not None and p.until_steady:
        source = Params(**resume["params"])
        same_criterion = (resume.get("steady_criterion_version") == STEADY_CRITERION_VERSION
                          and not resume.get("convergence_note")
                          and all(getattr(source, k) == getattr(p, k) for k in STEADY_SETTINGS))
        if same_criterion:
            checkpoint = resume.get("checkpoint")
            if checkpoint:
                n_ok = int(checkpoint["steady_n_ok"])
            else:
                # 再開メタデータ追加前の結果でも、判定用の全履歴があれば連続 OK を復元できる。
                last = start_batch-start_batch % p.steady_window
                for size in range(last, history_start+2*p.steady_window-1, -p.steady_window):
                    h = {k: v[:size] for k, v in hist.items()}
                    q = {k: v[:size] for k, v in charge_monitor_hist.items()}
                    if not (check_steady(h, probes, p, steady_reference)[0] and check_charge_steady(q, p)["ok"]):
                        break
                    n_ok += 1
                    if n_ok >= p.steady_hold:
                        break

    for b in range(start_batch, n_max):
        if stop is not None and stop.is_set():
            stopped = True
            break
        x, y, vx, vy, qm, spc = inject(p, p.n_per_batch, ny, rng, ion_energy, ion_velocity)
        trace_options = {}
        if fields["rf"] is not None:
            rf = {**fields["rf"], "initial_time": np.concatenate([
                RF.launch_times(phase_rng, p.n_per_batch, fields["rf"]["period"]) for _ in range(2)])}
            trace_options["rf"] = rf
        cnt_i, cnt_e, esc, n_lost = particle_tracer(x, y, vx, vy, qm, spc, Ex, Ey, vac, p,
                                          (phi, wface) if p.wall_model == "barrier" else None, **trace_options)

        # 電荷の蓄積 -> Poisson
        rho_prev = rho.copy() if conductor else None
        drho = (cnt_i - cnt_e) * q_macro / dx ** 2
        if cond.any():
            drho[cond] = 0.0                # 導体に当たった分は、下でまとめて半陰的に加える
        rho += drho

        # 表面リーク: 表面セルの電荷を表面に沿って移動させる
        leak_ratio = 0.0
        if leak is not None:
            Qs = rho.flat[leak["flat"]] * dx ** 2           # 表面セルの電荷 [C/m]
            Qs_old = Qs
            # 指定電圧による表面電流も含める (浮遊・接地の既定条件では 0)。
            if np.any(fields["external"]):
                Qs = Qs + p.dt_batch * (leak["G"] @ fields["external"].ravel()[leak["flat"]])
            Qs_new = leak["M"] @ Qs
            rho.flat[leak["flat"]] = Qs_new / dx ** 2
            dQ = Qs_old[leak["region"]].sum() - Qs_new[leak["region"]].sum()  # 下半分から上へ流出した正電荷
            leak_ratio = dQ / p.dt_batch / (E_CHARGE * p.flux * (e1-e0) * dx)
        if driven:
            rho[cond] = 0.0                 # 電位固定の導体に流れた電荷は電源へ逃がす

        if conductor:
            # 導体の電位変化を半陰的に求め、それに対応する導体自身の電荷を 1 セルにまとめて置く
            # (導体の電荷は合計だけが意味を持つ。当たった電荷との差は、電位変化で追い返された/引き込まれた電子の分)
            drift = ((rho - rho_prev) * g_cond).sum() * dx ** 2   # ここまでの電荷変化による導体の電位変化 [V]
            dV = conductor_step(cnt_i[cond].sum(), cnt_e[cond].sum(), q_macro, C_cond,
                                p.electron_temp_eV, drift)
            rho.flat[c_cell] += C_cond * (dV - drift) / dx ** 2

        phi = solve((-rho / EPS0).ravel()).reshape(nx, ny) + fields["external"]
        Ex, Ey = compute_field(phi, vac, p, fields["volts"]["top"])

        # 記録
        hist["t"].append((b + 1) * p.dt_batch)
        for k, (ix, iy) in probes.items():
            hist[k].append(phi[ix, iy])
        norm = p.n_per_batch * frac_open        # 底面幅への一様な直入射に相当する粒子数
        hist["ratio_i_bottom"].append(cnt_i[b0:b1, p.floor_t - 1].sum() / norm)
        hist["ratio_e_bottom"].append(cnt_e[b0:b1, p.floor_t - 1].sum() / norm)
        hist["esc_e"].append(esc[1] / p.n_per_batch)
        hist["lost"].append(n_lost / (2 * p.n_per_batch))
        hist["leak"].append(leak_ratio)
        charges = sample_charge_monitors(rho, q_regions, dx)
        for side in ("left", "right"):
            charge_hist[side].append(charges[side])
        for key, q in charges.items():
            charge_monitor_hist[key].append(q)

        if (b + 1) % 10 == 0 or b == 0:
            log(f"[{b+1:4d}/{n_max}] t={hist['t'][-1]*1e6:7.1f} μs  "
                  f"phi(bottom)={hist['bottom center'][-1]:7.2f} V  "
                  f"phi(side mid)={hist['sidewall middle'][-1]:7.2f} V  "
                  f"phi(right mid)={hist['right sidewall middle'][-1]:7.2f} V  "
                  f"Q壁 L/R={charge_hist['left'][-1]*1e12:+.3f}/{charge_hist['right'][-1]*1e12:+.3f} pC/m  "
                  f"phi(mask top)={hist['mask top'][-1]:7.2f} V  "
                  f"e/i@bottom={np.mean(hist['ratio_e_bottom'][-10:]) / max(np.mean(hist['ratio_i_bottom'][-10:]), 1e-9):.2f}  "
                  f"lost={np.mean(hist['lost'][-10:])*100:.2f}%  "
                  + (f"leak/Iion={np.mean(hist['leak'][-10:]):.3f}  " if leak is not None else "")
                  + f"({time.time()-t0:.0f}s)")

        # 飽和判定 (継続モード)
        W = p.steady_window
        check_info = None
        if p.until_steady and (b + 1) % W == 0 and (b + 1)-history_start >= 2 * W:
            v_ok, name, d, tol = check_steady(hist, probes, p, steady_reference)
            charge = check_charge_steady(charge_monitor_hist, p)
            ok = bool(v_ok and charge["ok"])
            n_ok = min(n_ok+1, p.steady_hold) if ok else 0
            log(f"  [飽和判定] 電位 {'OK' if v_ok else 'NG'}: {name} {d:+.3g} V (許容 {tol:.3g} V); "
                f"電荷 {'OK' if charge['ok'] else 'NG'}: {charge['name']} "
                f"{'継続ドリフト' if charge['kind']=='drift' else '平均差'} {charge['d']*1e12:+.3g} pC/m "
                f"/ {W*p.dt_batch*1e3:g} ms (許容 {charge['tol']*1e12:.3g} pC/m) "
                f"-> {'OK' if ok else 'NG'} ({n_ok}/{p.steady_hold})")
            voltage = dict(ok=bool(v_ok), name=name, d=float(d), tol=float(tol), unit="V", kind="voltage")
            limiting = charge if not charge["ok"] else voltage
            check_info = {**limiting, "ok": ok, "n_ok": n_ok, "voltage": voltage, "charge": charge}
            if n_ok >= p.steady_hold:
                converged_at = b + 1

        if progress is not None:
            shown_phi, _, shown_rhs, shown_volts = field_snapshot(fields, phi, p)
            shown_ex, shown_ey = (compute_field(shown_phi, vac, p, shown_volts["top"])
                                  if fields["rf"] is not None else (Ex.copy(), Ey.copy()))
            progress(dict(batch=b + 1, start_batch=start_batch, n_max=n_max, elapsed=time.time() - t0,
                          steady_samples_remaining=max(0, 2*W-((b+1)-history_start)),
                          phi=shown_phi.copy(), rho=conductor_surface_charge(rho, shown_phi, A, cond, shown_rhs),
                          Ex=shown_ex, Ey=shown_ey,
                          hist={k: np.array(v) for k, v in hist.items()},
                          charge_hist={k: np.array(v) for k, v in charge_hist.items()},
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
    shown_phi, external, shown_rhs, volts = field_snapshot(fields, phi, p)
    shown_ex, shown_ey = (compute_field(shown_phi, vac, p, volts["top"])
                          if fields["rf"] is not None else (Ex, Ey))
    return dict(phi=shown_phi, mean_phi=phi, rho=conductor_surface_charge(rho, shown_phi, A, cond, shown_rhs),
                Ex=shown_ex, Ey=shown_ey,
                rho_raw=rho.copy(), external_phi=external, voltages=volts,
                charge_hist={k: np.array(v) for k, v in charge_hist.items()},
                charge_monitor_hist={k: np.array(v) for k, v in charge_monitor_hist.items()},
                steady_criterion_version=STEADY_CRITERION_VERSION,
                checkpoint=dict(version=CHECKPOINT_VERSION, batches_completed=len(hist["t"]),
                                rng_state=rng.bit_generator.state,
                                phase_rng_state=phase_rng.bit_generator.state if phase_rng is not None else None,
                                steady_n_ok=n_ok),
                start_batch=start_batch,
                params=asdict(p), converged=converged_at is not None,
                backend_used=backend_used,
                solid=solid, mask=mask, i0=i0, i1=i1, hist=hist, probes=probes,
                steady=steady, t_conv=t_conv, stopped=stopped, circuit=circuit)


# ---------------------------------------------------------------- 可視化
def probe_vertical_ions(res, p=None, phase=None, log=print, stop=None):
    """飽和電荷を固定し同じ垂直ビームを比較。circuit_rf では入射位相もそろえる。"""
    if not np.isfinite(res.get("t_conv", np.nan)) or res.get("stopped", False):
        raise ValueError("飽和判定を通過した結果が必要です。飽和まで継続モードで計算してください")
    source = Params(**res["params"])
    p = source if p is None else p
    for key in ("dx", "nx", "trench_w", "trench_offset", "trench_taper_deg", "trench_d", "floor_t", "n_vac",
                "mask_t", "mask_taper_deg", "mask_type", "eps_r", "mask_eps_r", "field_mode", "side_bc",
                "substrate_voltage", "mask_voltage", "top_voltage", "left_voltage", "right_voltage",
                "top_bc", "ion_mass_amu", "wall_model", "bias"):
        if getattr(p, key) != getattr(source, key):
            raise ValueError("診断の形状・境界条件は飽和計算と同じ設定にしてください")
    if (p.probe_ions < 1 or p.probe_trajectories < 1 or p.probe_trajectories > p.probe_ions
            or not np.isfinite(p.probe_energy_eV) or p.probe_energy_eV <= 0):
        raise ValueError("診断の粒子数・エネルギーは正、軌道数は粒子数以下にしてください")
    if phase is not None:
        from dataclasses import replace
        p = replace(p, rf_phase=phase)
    fields = setup_fields(p, res["solid"], res["mask"], res.get("circuit"))
    solid, vac = res["solid"], ~res["solid"]
    particle_tracer, backend_used = make_particle_tracer(p, vac, log)
    ny = solid.shape[1]
    i0, i1 = opening_span(solid, p.floor_t+p.trench_d+p.mask_t-1)
    b0, b1 = opening_span(solid, p.floor_t)
    # 等間隔のビームで角度・位置サンプリングの差を排除する (vx は厳密に 0)。
    x = (i0 + (np.arange(p.probe_ions) + 0.5) * (i1 - i0) / p.probe_ions) * p.dx
    y = np.full(x.size, (ny - 1e-3) * p.dx)
    vx = np.zeros(x.size)
    vy = np.full(x.size, -np.sqrt(2 * p.probe_energy_eV * E_CHARGE / (p.ion_mass_amu * AMU)))
    qm = np.full(x.size, E_CHARGE / (p.ion_mass_amu * AMU))
    spc = np.zeros(x.size, dtype=np.int8)
    charged_phi = fields["solve"]((-res["rho_raw"] / EPS0).ravel()).reshape(solid.shape) + fields["external"]
    wface = np.where(fields["cond"], 1.0, fields["eps"] / (1 + fields["eps"]))
    result = dict(energy_eV=p.probe_energy_eV, n_ions=x.size, phase=p.rf_phase,
                  entrance_i0=i0, entrance_i1=i1, bottom_i0=b0, bottom_i1=b1,
                  source_t_conv=res["t_conv"], source_converged=True,
                  cancelled=False, backend_used=backend_used,
                  rf_cycle=fields["rf"] is not None)
    trace_options = {}
    if fields["rf"] is not None:
        rng = np.random.default_rng(np.random.SeedSequence([p.seed, 0x50524F42]))
        trace_options["rf"] = {**fields["rf"], "initial_time": RF.launch_times(rng, x.size, fields["rf"]["period"])}
        result["rf_period"] = fields["rf"]["period"]
        result["rf_time_steps"] = p.rf_time_steps
        log("RF 全周期の垂直診断: 入射位相を 0–360 deg に分散し、飛行中も回路電場を更新します")
    for label, phi in (("charged", charged_phi), ("uncharged", fields["external"])):
        log(f"垂直イオン診断: {'帯電あり' if label == 'charged' else '帯電なし'}、{x.size} 個、{p.probe_energy_eV:g} eV")
        Ex, Ey = compute_field(phi, vac, p, fields["volts"]["top"])
        record = {}
        ci, _, esc, lost = particle_tracer(x.copy(), y.copy(), vx.copy(), vy.copy(), qm.copy(), spc.copy(),
                               Ex, Ey, vac, p, (phi, wface) if p.wall_model == "barrier" else None,
                               record=record, stop=stop, **trace_options)
        shown_phi, _, _, volts = field_snapshot(fields, phi, p)
        shown_ex, shown_ey = compute_field(shown_phi, vac, p, volts["top"])
        record.update(phi=shown_phi, Ex=shown_ex, Ey=shown_ey, hits=ci, escaped=int(esc[0]), lost=lost,
                      bottom_fraction=float(ci[b0:b1, p.floor_t - 1].sum() / x.size),
                      bottom_flux=ci[b0:b1, p.floor_t - 1] * p.flux * (i1 - i0) / x.size)
        record["angle_deg"] = np.degrees(np.arctan2(record["final_velocity"][:, 0],
                                                   -record["final_velocity"][:, 1]))
        record["impact_energy_eV"] = (p.ion_mass_amu * AMU / (2 * E_CHARGE)
                                      * np.sum(record["final_velocity"] ** 2, axis=1))
        result[label] = record
        if record["cancelled"]:
            result["cancelled"] = True
            return result
        log(f"  底到達 {record['bottom_fraction']:.1%}, 上端流出 {record['escaped']}, 未到達 {lost}")
    return result


def plot_ion_probe(probe, res, p, pyplot=False, compact=False):
    """電場上の軌道、底フラックス、底到達時の偏向角を比較する Figure。"""
    rf_cycle = probe.get("rf_cycle", False)
    if compact:
        if pyplot:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [1.7, 1]})
        else:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(10, 8), constrained_layout=True)
            axs = fig.subplots(2, 2, gridspec_kw={"height_ratios": [1.7, 1]})
        axs = axs.ravel()
    else:
        if rf_cycle:
            if pyplot:
                fig, axs = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
            else:
                from matplotlib.figure import Figure
                fig = Figure(figsize=(16, 10), constrained_layout=True)
                axs = fig.subplots(2, 3)
            axs = axs.ravel()
        else:
            fig, axs = _subplots(4, (19, 7), pyplot)
    solid, mask = res["solid"], res["mask"]
    nx, ny = solid.shape
    dxn = p.dx * 1e9
    xc, yc = (np.arange(nx) + 0.5) * dxn, (np.arange(ny) + 0.5) * dxn
    colors = {"charged": "tab:red", "uncharged": "tab:blue"}
    vmax = max(np.hypot(probe[k]["Ex"], probe[k]["Ey"])[~solid].max() / 1e6
               for k in colors)
    for ax, (label, color) in zip(axs[:2], colors.items()):
        d = probe[label]
        im = ax.imshow(np.ma.masked_where(solid, np.hypot(d["Ex"], d["Ey"]) / 1e6).T,
                       origin="lower", extent=[0, nx * dxn, 0, ny * dxn], cmap="viridis",
                       vmin=0, vmax=max(vmax, 1e-9))
        draw_outline(ax, xc, yc, solid, mask, "w")
        for points in d["paths"]:
            points = points.copy() * 1e9
            jumps = np.abs(np.diff(points[:, 0])) > nx * dxn / 2
            points[1:][jumps] = np.nan  # 周期境界を横断する線を図の中央に引かない
            ax.plot(points[:, 0], points[:, 1], color="orange" if label == "charged" else "cyan", lw=0.8)
        ax.set(xlabel="x [nm]", ylabel="y [nm]", xlim=(0, nx * dxn), ylim=(0, ny * dxn),
               title=f"{label}: bottom {d['bottom_fraction']:.1%}\nescaped {d['escaped']}, unresolved {d['lost']}")
        fig.colorbar(im, ax=ax, shrink=0.6, label="|E| [MV/m]")
        b0, b1 = opening_span(solid, p.floor_t)
        axs[2].step(xc[b0:b1], d["bottom_flux"] / p.flux,
                    where="mid", color=color, label=label)
        bottom = (d["status"] == 1) & (d["hit_cells"][:, 1] == p.floor_t - 1)
        axs[3].scatter(d["initial_x"][bottom] * 1e9, d["angle_deg"][bottom],
                       s=5, color=color, label=label)
        if rf_cycle and not compact:
            _draw_probe_phase(axs[4:], d, bottom, color, label)
    axs[2].set(xlabel="bottom x [nm]", ylabel="bottom flux / incident flux", title="Bottom flux")
    axs[3].set(xlabel="injection x [nm]", ylabel="angle from vertical [deg]", title="Bottom impact angle")
    if rf_cycle and not compact:
        axs[4].set(xlabel="launch RF phase [deg]", ylabel="bottom fraction", xlim=(0, 360),
                   title="Bottom arrival by launch phase")
        axs[5].set(xlabel="launch RF phase [deg]", ylabel="impact energy [eV]", xlim=(0, 360),
                   title="Bottom energy by launch phase")
    for ax in axs[2:]:
        ax.legend(); ax.grid(alpha=0.3)
    phase_label = (f"full RF cycle, field backdrop at {probe['phase']%360:g} deg" if rf_cycle else
                   f"RF phase={probe['phase']%360:g} deg")
    charge_note = "" if probe.get("source_converged", True) else "; source charge not saturated"
    fig.suptitle(f"Vertical ions: {probe['energy_eV']:g} eV, {probe['n_ions']} ions; "
                 f"frozen charge at t={probe['source_t_conv']*1e3:.3g} ms, {phase_label}{charge_note}")
    if not compact and not rf_cycle:
        fig.tight_layout()
    return fig


def _draw_probe_phase(axs, d, bottom, color, label):
    edges = np.linspace(0, 360, 17)
    launched = np.histogram(d["launch_phase"], edges)[0]
    arrived = np.histogram(d["launch_phase"][bottom], edges)[0]
    ratio = np.divide(arrived, launched, out=np.full(16, np.nan), where=launched > 0)
    axs[0].step((edges[:-1] + edges[1:]) / 2, ratio, where="mid", color=color, label=label)
    axs[1].scatter(d["launch_phase"][bottom], d["impact_energy_eV"][bottom], s=5, color=color, label=label)


def plot_rf_probe_phase(probe, p, pyplot=False):
    """GUI の専用 RF 位相タブ用。軌道と分けて到達率・エネルギーを大きく表示する。"""
    if not probe.get("rf_cycle"):
        raise ValueError("RF 全周期の垂直イオン診断が必要です")
    fig, axs = _subplots(2, (11, 5), pyplot)
    for label, color in (("charged", "tab:red"), ("uncharged", "tab:blue")):
        d = probe[label]
        bottom = (d["status"] == 1) & (d["hit_cells"][:, 1] == p.floor_t - 1)
        _draw_probe_phase(axs, d, bottom, color, label)
    axs[0].set(xlabel="launch RF phase [deg]", ylabel="bottom fraction", xlim=(0, 360), ylim=(0, 1.05),
               title="Bottom arrival by launch phase")
    axs[1].set(xlabel="launch RF phase [deg]", ylabel="impact energy [eV]", xlim=(0, 360),
               title="Bottom energy by launch phase")
    for ax in axs:
        ax.legend(); ax.grid(alpha=.3)
    fig.suptitle("Full RF cycle: phase dependence of vertical ions")
    fig.tight_layout()
    return fig


def plot_charge_history(res, pyplot=False):
    fig, axs = _subplots(1, (7, 4.5), pyplot)
    for side, q in res["charge_hist"].items():
        axs.plot(res["hist"]["t"] * 1e3, q * 1e12, label=f"{side} SiO2 wall")
    axs.set(xlabel="time [ms]", ylabel="sidewall charge [pC/m]", title="SiO2 sidewall charge")
    axs.legend(); axs.grid(alpha=0.3)
    fig.tight_layout()
    return fig


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
        label = "no mask"
    else:
        kind = "conductor" if p.mask_type == "conductor" else f"dielectric, eps_r={p.mask_eps_r:g}"
        label = f"mask {p.mask_t * p.dx * 1e9:.0f} nm ({kind})"
    if p.trench_taper_deg or (p.mask_t > 0 and p.mask_taper_deg):
        label += f", taper from vertical: SiO2 {p.trench_taper_deg:g} deg"
        if p.mask_t > 0:
            label += f" / mask {p.mask_taper_deg:g} deg"
    return label


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
    profiles = wall_profiles(solid, p)
    lx, ly = profiles["left"]
    rx, ry = profiles["right"]
    ax = axs[2]
    ax.plot(phi[lx, ly], (ly + 0.5) * dxn, "r-", label="left potential")
    ax.plot(phi[rx, ry], (ry + 0.5) * dxn, "r--", label="right potential")
    ax.set_xlabel("Potential [V]", color="r")
    ax.set_ylabel("y [nm]")
    ax.set_title("Left / right sidewall profiles")
    ax.legend(fontsize=8, loc="lower left")
    ax2 = ax.twiny()
    ax2.plot(rho[lx, ly] * p.dx * 1e3, (ly + 0.5) * dxn, "b--", label="surface charge")
    ax2.plot(rho[rx, ry] * p.dx * 1e3, (ry + 0.5) * dxn, "b:", label="right charge")
    ax2.set_xlabel("Surface charge [mC/m$^2$]", color="b")
    if p.mask_t > 0:
        draw_mask_level(ax, top_ox * dxn)
    ax.set_ylim(extent[2], extent[3])
    ax.grid(alpha=0.3)
    for a in axs[:2]:
        a.set_xlabel("x [nm]"); a.set_ylabel("y [nm]")
    leak_label = f"surface sheet conductance = {p.sigma_s:g} S" if p.sigma_s > 0 else "no surface leakage"
    phase_note = f", full RF cycle (snapshot {p.rf_phase%360:g} deg)" if p.field_mode == "circuit_rf" else ""
    fig.suptitle(f"{mask_label(p)},  {leak_label}{phase_note}")
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
    if p.field_mode == "circuit_rf":
        axs[0].set_title("RF cycle mean potential (charging time)")
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


def plot_circuit_figure(c, pyplot=False):
    """等価回路の波形と IED (1 次元シースならイオンの角度分布も) の図 (Figure) を作る。"""
    n = 3 if "vn" in c else 2
    fig, axs = _subplots(n, (6 * n, 4.2), pyplot)
    PC.plot_circuit(c, *axs)
    fig.tight_layout()
    return fig


def save_results(res, p, out, pyplot=False):
    """場・時間履歴・左右の電荷履歴と、設定を含む再利用可能な NPZ を保存する。
    RF なら回路の図・波形、垂直イオン診断後なら比較図・軌道・到達情報も保存する。"""
    plot_results(res, p, out, pyplot=pyplot)
    files = [f"{out}_fields.png", f"{out}_history.png", f"{out}.npz"]
    c = res.get("circuit")
    rf = {}
    if c is not None:
        plot_circuit_figure(c, pyplot).savefig(f"{out}_rf.png", dpi=130)
        files.append(f"{out}_rf.png")
        rf = {f"rf_{k}": v for k, v in c.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in c.items() if not isinstance(v, np.ndarray) and k != "sheath"}
        sheath = c.get("sheath")
        meta["sheath"] = None if sheath is None else {k: v for k, v in sheath.items() if not isinstance(v, np.ndarray)}
        if sheath is not None:
            rf.update({f"rf_sheath_{k}": v for k, v in sheath.items() if isinstance(v, np.ndarray)})
        rf["circuit_meta_json"] = np.array(json.dumps(meta))
    plot_charge_history(res, pyplot).savefig(f"{out}_charge.png", dpi=130)
    files.append(f"{out}_charge.png")
    diagnostics = {}
    probe = res.get("ion_probe")
    if probe is not None and not probe["cancelled"]:
        plot_ion_probe(probe, res, p, pyplot).savefig(f"{out}_ions.png", dpi=130)
        files.append(f"{out}_ions.png")
        diagnostics["probe_meta_json"] = np.array(json.dumps({k: v for k, v in probe.items()
                                                              if k not in ("charged", "uncharged")}))
        for label in ("charged", "uncharged"):
            d = probe[label]
            diagnostics.update({f"probe_{label}_{k}": v for k, v in d.items() if isinstance(v, np.ndarray)})
            diagnostics[f"probe_{label}_path_lengths"] = np.array([len(v) for v in d["paths"]])
            diagnostics[f"probe_{label}_paths"] = np.concatenate(d["paths"])
            diagnostics[f"probe_{label}_meta_json"] = np.array(json.dumps({k: v for k, v in d.items()
                if not isinstance(v, np.ndarray) and k != "paths"}))
    np.savez(f"{out}.npz", phi=res["phi"], rho=res["rho"], Ex=res["Ex"], Ey=res["Ey"],
             mean_phi=res.get("mean_phi", res["phi"]),
             solid=res["solid"], mask=res["mask"], t_conv=res["t_conv"],
             rho_raw=res["rho_raw"], external_phi=res["external_phi"],
             params_json=np.array(json.dumps(res["params"])), stopped=res["stopped"],
             backend_used=np.array(res.get("backend_used", "cpu")),
             checkpoint_json=np.array(json.dumps(res.get("checkpoint"))),
             convergence_meta_json=np.array(json.dumps(dict(
                 criterion_version=res.get("steady_criterion_version", 0),
                 note=res.get("convergence_note", ""), previous_t_conv=res.get("previous_t_conv")))),
             **{f"qmonitor_{k}": v for k, v in res.get("charge_monitor_hist", {}).items()},
             **{f"charge_{k}": v for k, v in res["charge_hist"].items()},
             **{f"hist_{k}": v for k, v in res["hist"].items()},
             **{f"steady_{k}": v for k, v in res["steady"].items()}, **rf, **diagnostics)
    return files


def load_results(filename):
    """NPZ を pickle なしで読み、蓄積電荷から帯電の再開や飽和後診断を実行できる。"""
    with np.load(filename, allow_pickle=False) as d:
        if "params_json" not in d or "rho_raw" not in d:
            raise ValueError("この結果は旧形式です。設定と内部電荷を含む新形式で再計算・保存してください")
        p = Params(**json.loads(str(d["params_json"])))
        solid, mask, i0, i1 = build_geometry(p)
        res = {k: d[k].copy() for k in ("phi", "rho", "rho_raw", "Ex", "Ey", "external_phi", "solid", "mask")}
        res["mean_phi"] = d["mean_phi"].copy() if "mean_phi" in d else res["phi"].copy()
        if not np.array_equal(res["solid"], solid) or not np.array_equal(res["mask"], mask):
            raise ValueError("保存された形状と設定が一致しません")
        for key in ("phi", "mean_phi", "rho", "rho_raw", "Ex", "Ey", "external_phi"):
            if res[key].shape != solid.shape or not np.isfinite(res[key]).all():
                raise ValueError(f"保存結果の {key} の形状または数値が不正です")
        res.update(i0=i0, i1=i1, params=asdict(p), t_conv=float(d["t_conv"]),
                   stopped=bool(d["stopped"]), hist={k[5:]: d[k].copy() for k in d.files if k.startswith("hist_")},
                   steady={k[7:]: float(d[k]) for k in d.files if k.startswith("steady_")},
                   charge_hist={k[7:]: d[k].copy() for k in d.files if k.startswith("charge_")}, circuit=None)
        res["probes"] = list(res["steady"])
        res["converged"] = np.isfinite(res["t_conv"]) and not res["stopped"]
        res["backend_used"] = str(d["backend_used"]) if "backend_used" in d else "cpu"
        res["charge_monitor_hist"] = {k[9:]: d[k].copy() for k in d.files if k.startswith("qmonitor_")}
        convergence = json.loads(str(d["convergence_meta_json"])) if "convergence_meta_json" in d else {}
        res["steady_criterion_version"] = convergence.get("criterion_version", 0)
        if "checkpoint_json" in d:
            checkpoint = json.loads(str(d["checkpoint_json"]))
            if checkpoint is not None:
                res["checkpoint"] = checkpoint
        if convergence.get("note"):
            res["convergence_note"] = convergence["note"]
        if convergence.get("previous_t_conv") is not None:
            res["previous_t_conv"] = convergence["previous_t_conv"]
        if "circuit_meta_json" in d:
            c = json.loads(str(d["circuit_meta_json"]))
            c.update({k[3:]: d[k].copy() for k in d.files if k.startswith("rf_") and not k.startswith("rf_sheath_")})
            if c.get("sheath") is not None:
                c["sheath"].update({k[10:]: d[k].copy() for k in d.files if k.startswith("rf_sheath_")})
            res["circuit"] = c
        if "probe_meta_json" in d:
            probe = json.loads(str(d["probe_meta_json"]))
            for label in ("charged", "uncharged"):
                prefix = f"probe_{label}_"
                record = json.loads(str(d[prefix + "meta_json"]))
                record.update({k[len(prefix):]: d[k].copy() for k in d.files
                               if k.startswith(prefix) and k != prefix + "meta_json"})
                record["paths"] = list(np.split(record["paths"], np.cumsum(record.pop("path_lengths"))[:-1]))
                probe[label] = record
            res["ion_probe"] = probe
    # 旧判定の「飽和」も、保存された電荷履歴で再検証する。場と履歴は保持する。
    if res["converged"]:
        qhist = res["charge_monitor_hist"] or res["charge_hist"]
        valid = bool(qhist)
        for block in range(p.steady_hold) if valid else ():
            size = len(res["hist"]["t"]) - block*p.steady_window
            part = {k: q[:size] for k, q in qhist.items()}
            if size < 2*p.steady_window or not check_charge_steady(part, p)["ok"]:
                valid = False
                break
        if not valid:
            res["previous_t_conv"] = res["t_conv"]
            res["t_conv"], res["converged"] = np.nan, False
            res["convergence_note"] = "保存結果の電荷は飽和していません。電位・電荷の新しい判定で再計算してください。"
            if "ion_probe" in res:
                res["ion_probe"]["source_converged"] = False
    return res, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = {"mask_type": MASK_TYPES, "wall_model": WALL_MODELS, "bias": PC.BIAS_TYPES, "rf_wave": PC.WAVE_TYPES,
               "ied_model": PC.IED_MODELS, "field_mode": FIELD_MODES, "side_bc": SIDE_BCS,
               "top_bc": ("dirichlet", "neumann"), "backend": BACKENDS}
    for k, v in asdict(Params()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=argparse.SUPPRESS)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=argparse.SUPPRESS, choices=choices.get(k))
    ap.add_argument("--out", help="保存先の接頭辞 (再開時の既定値: 読込元の名前 + _continued)")
    ap.add_argument("--show", action="store_true", help="図をウィンドウにも表示する")
    ap.add_argument("--no_cache", action="store_true", help="等価回路と IED のキャッシュ (.cache/circuit) を使わない")
    ap.add_argument("--vertical_probe", action="store_true", help="飽和後、垂直イオンの軌道と底フラックスを比較する")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--load_result", help="保存 NPZ の飽和結果から診断する (形状・境界は保存設定を使用)")
    source.add_argument("--resume_result", help="保存 NPZ から続きの帯電計算を実行する。--n_batches は追加分、--max_batches は通算上限")
    ap.add_argument("--probe_phase", type=float, help="診断の固定位相 / circuit_rf の背景表示位相 [deg] (保存電荷は固定)")
    args = vars(ap.parse_args())
    out, show, no_cache = args.pop("out"), args.pop("show"), args.pop("no_cache")
    vertical_probe, load_result = args.pop("vertical_probe"), args.pop("load_result")
    resume_result = args.pop("resume_result")
    probe_phase = args.pop("probe_phase")
    p = Params(**args)

    if not show:
        matplotlib.use("Agg")
    if load_result:
        from dataclasses import replace
        res, saved = load_results(load_result)
        p = replace(saved, **{k: v for k, v in args.items()
                              if k in ("backend", "cuda_device", "probe_ions", "probe_energy_eV", "probe_trajectories")})
    else:
        resumed = None
        if resume_result:
            from dataclasses import replace
            resumed, saved = load_results(resume_result)
            if "n_batches" in args and "until_steady" not in args:
                args["until_steady"] = False
            p = replace(saved, **args)
            p = resume_params(resumed, p)
        if vertical_probe:
            p.until_steady = True
        res = run(p, use_cache=not no_cache, resume=resumed)
    if out is None:
        import os
        out = os.path.splitext(resume_result)[0]+"_continued" if resume_result else "trench_charging"
    if vertical_probe:
        if not np.isfinite(res["t_conv"]):
            print("飽和未到達のため垂直イオン診断は実行しません。帯電結果を保存します。")
        else:
            res["ion_probe"] = probe_vertical_ions(res, p, phase=probe_phase)
    files = save_results(res, p, out, pyplot=show)
    print("保存: " + ", ".join(files))
    if show:
        plt.show()


if __name__ == "__main__":
    main()
