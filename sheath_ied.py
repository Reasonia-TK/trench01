#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 次元シースの時間発展からイオンのエネルギー・角度分布を求める (ied_model="sheath")
==================================================================================

等価回路 (plasma_circuit.py) が出したウェハ側シース電圧 V1(t) を境界条件にして、
電極 (ウェハ表面) とプラズマの間の 1 次元・静電のシースを時間発展させる。

* 領域: 電極 y=0 から プラズマ側 y=L まで (L は V1 の最大値での Child 則のシース厚の 1.5 倍 + 5 λ_D。
  実際のシースは Child 則より 1.6〜1.9 λ_D 厚いが、十分に収まる)。
  境界は φ(0) = -V1(t) (電極), φ(L) = 0 (プラズマ)。
* イオン: 粒子 (マクロ粒子)。速度は 3 成分 (y: 電極の法線, x と z: 面内) で追う。y=L から Bohm 速度 u_B
  (法線方向) と、温度 ion_temp の面内の熱速度で、一定のフラックス Γ (Params.flux) で入れる。
  電極に着いたら吸収し、そのときの法線・面内の速度を記録する。
* 電子: ボルツマン分布 n_e = n_s exp(φ/Te) (電位の変化に瞬時に応答)。n_s = Γ / u_B はシース端の密度。
* Poisson 方程式 ε0 φ'' = -e (n_i - n_e(φ)) は電子のせいで非線形なので、毎ステップ Newton 法で解く
  (三重対角)。イオン密度は粒子から線形の重み (CIC) で格子点に集める。
* 時間は leapfrog。電極に着いたイオンの法線方向の運動エネルギーは、最後のステップの中での電位差で
  補正する (M v_n^2 / 2 + e (φ(y_n) - φ(0)), v_n は時刻 n の速度)。
* 最初は領域を密度 n_s・速度 u_B のイオンで満たし、warmup (イオンが領域を何度も入れ替わる時間) だけ
  回してから、RF 周期の整数倍 (min_collect 以上) の間に電極に着いたイオンを集める。
  イオン電流は時間的に一定で注入するので、集めたイオンはそのまま等しい重みのエネルギー・角度分布になる。
* 衝突 (pressure > 0): 背景ガス (イオンと同じ原子, 温度 gas_temp) との弾性散乱と共鳴電荷交換を、
  南部–北谷モデル (K. Nanbu and Y. Kitatani, J. Phys. D: Appl. Phys. 28, 324 (1995)) で扱う。
    - 分極ポテンシャル -a/r^4 (a = α e^2 / (8π ε0), Ar の分極率 α/(4π ε0) = 1.642 Å^3)。
      無次元の衝突パラメーター β = b (E/4a)^(1/4) (E: 相対運動のエネルギー) を 0 < β < β_m から
      面積に比例して選ぶ (β = β_m √U)。全断面積 π b_m^2 が 1/g に比例するので、
      衝突の確率 N σ_T g Δt は速度によらず一定になる (論文 式 (10))。
    - β < 1.001: 捕獲。重心系で等方に散乱する (論文 式 (13))。
      β > 1.001: 偏向角 χ(β) = π - 2√2 β K(1/ε1^4) / ε1, ε1 = [β^2 + (β^4 - 1)^(1/2)]^(1/2) (論文 式 (6)(7))。
      散乱後の速度は論文 式 (2)(3) (cos χ' と |sin χ| を使う)。
    - β < β_ex(E) = max(1, A E^(1/4)) [E: eV] なら確率 1/2 で電荷交換: イオンは中性粒子の散乱後の速度を持つ。
      A = 2.6 eV^-1/4 (Ar+-Ar のドリフト速度が実験と合う値)。電荷交換の断面積は π b_ex^2 / 2 で一定。
    - 打ち切り β_m は最低 3 (χ(3) = -0.42°)。β_m < β_ex だと電荷交換が少なくなるので、
      シース電圧の最大値から見積もった相対エネルギーの上限で β_m > β_ex になるようにする。

パルスでシースが一気に広がる動き (イオンのマトリクスシース) や、イオンが電圧に追従できるかどうか
(周波数とイオンのプラズマ周波数の比)、シースの中での衝突を、粒子の運動として直接扱う。
"""
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ellipk

E_CHARGE = 1.602176634e-19   # [C]   (trench_charging_2d.py と同じ値)
EPS0 = 8.8541878128e-12      # [F/m]
K_B = 1.380649e-23           # [J/K]

# 南部–北谷モデル (Ar+ - Ar)
NK_ALPHA = 1.642e-30         # 分極率 α/(4π ε0) [m^3]
NK_A = 2.6                   # 電荷交換の定数 A [eV^-1/4]
NK_BETA0 = 1.001             # これより小さい β は捕獲 (等方散乱)
NK_BETA_MIN = 3.0            # 打ち切り β_m の最小値
MAX_STORE = 100_000          # 記録するイオンの数の上限 (多ければ間引く)


def _child_thickness(V, el):
    """Child 則のシース厚 [m] (等価回路のシース容量と同じ式)。"""
    V = np.maximum(V, 0.0)
    return (4 / 3) * EPS0 * V / (el["K"] * (V + el["Te"]) ** 0.25)


def nk_chi(beta):
    """分極ポテンシャル -a/r^4 での偏向角 χ(β) [rad] (β > 1。引力なので負)。
    ε0 ε1 = 1 なので楕円積分のモジュラスは ε0/ε1 = 1/ε1^2 (パラメーター m = 1/ε1^4) で、桁落ちしない。"""
    e1 = np.sqrt(beta ** 2 + np.sqrt(beta ** 4 - 1))
    return np.pi - 2 * np.sqrt(2) * beta * ellipk(1 / e1 ** 4) / e1


def nk_beta_m(E_max_eV):
    """相対エネルギーの上限 E_max [eV] で β_ex < β_m になる打ち切り β_m (最低 3, 5% の余裕)。"""
    return max(NK_BETA_MIN, 1.05 * NK_A * max(E_max_eV, 0.0) ** 0.25)


def nk_rate(M, beta_m):
    """衝突の頻度 / 気体の密度 = σ_T g = π β_m^2 (8a/μ)^(1/2) [m^3/s] (速度によらない)。μ = M/2。"""
    a = NK_ALPHA * E_CHARGE ** 2 / (8 * np.pi * EPS0)
    return np.pi * beta_m ** 2 * np.sqrt(8 * a / (M / 2))


def _unit_vectors(rng, n):
    c = 1 - 2 * rng.random(n)
    s = np.sqrt(1 - c ** 2)
    ph = 2 * np.pi * rng.random(n)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=1)


def nk_scatter(v, rng, M, kT, beta_m):
    """南部–北谷モデルで、衝突するイオンの散乱後の速度 (n x 3) を返す。v: 衝突前の速度 (n x 3) [m/s]。
    中性粒子はイオンと同じ質量 M で、温度 kT [J] のマクスウェル分布から選ぶ。
    戻り値: (散乱後の速度, 電荷交換したか (bool), β_ex > β_m だった数)"""
    n = v.shape[0]
    V = rng.normal(0.0, np.sqrt(kT / M), (n, 3))
    g = V - v                                            # 論文の定義 g = V - v
    gm = np.linalg.norm(g, axis=1)
    E_rel = 0.25 * M * gm ** 2 / E_CHARGE                # 相対運動のエネルギー μ g^2 / 2 [eV] (μ = M/2)
    beta = beta_m * np.sqrt(rng.random(n))
    beta_ex = np.maximum(NK_A * E_rel ** 0.25, 1.0)
    cx = (beta < beta_ex) & (rng.random(n) < 0.5)
    out = np.empty_like(v)
    iso = beta < NK_BETA0
    if iso.any():                                        # 捕獲: 重心系で等方 (m = M なので電荷交換は区別しない)
        out[iso] = 0.5 * (v[iso] + V[iso]) - 0.5 * gm[iso, None] * _unit_vectors(rng, int(iso.sum()))
    d = ~iso
    if d.any():
        gd, gmd = g[d], gm[d]
        chi = nk_chi(beta[d])
        cosx, sinx = np.cos(chi), np.abs(np.sin(chi))
        ph = 2 * np.pi * rng.random(int(d.sum()))
        cph, sph = np.cos(ph), np.sin(ph)
        gp = np.maximum(np.sqrt(gd[:, 1] ** 2 + gd[:, 2] ** 2), 1e-300)
        h = np.stack([gp * cph,
                      -(gd[:, 0] * gd[:, 1] * cph + gmd * gd[:, 2] * sph) / gp,
                      -(gd[:, 0] * gd[:, 2] * cph - gmd * gd[:, 1] * sph) / gp], axis=1)
        delta = 0.5 * (gd * (1 - cosx)[:, None] + h * sinx[:, None])   # M/(m+M) = m/(m+M) = 1/2
        out[d] = np.where(cx[d, None], V[d] - delta, v[d] + delta)    # 電荷交換: 中性粒子の散乱後の速度
    return out, cx & ~iso, int((beta_ex > beta_m).sum())


def ion_energies(t, V1, el, flux, ion_temp=0.0, pressure=0.0, gas_temp=300.0, ppc=200, dy_per_debye=0.25,
                 dt_max=0.25e-9, warmup=2e-6, min_collect=1e-6, l_factor=1.5, seed=0, engine="auto", log=None):
    """シース電圧の 1 周期分の波形 V1(t) (t: 等間隔, 周期 = t[1] * len(t)) から、電極に着いたイオンの
    エネルギー [eV] の配列、速度の dict (vn: 法線 (電極向きが正), vx, vz: 面内) [m/s]、計算条件の dict を返す。
    el: plasma_circuit.circuit_elements の dict (Te, m_i, K を使う), flux: イオンフラックス [m^-2 s^-1]
    ion_temp: 注入するイオンの面内の温度 [eV], pressure: 背景ガスの圧力 [Pa] (0 で衝突なし),
    gas_temp: 背景ガスの温度 [K], ppc: プラズマ側の 1 セルあたりのマクロ粒子数,
    dy_per_debye: 格子間隔 / デバイ長, dt_max: 時間刻みの上限 [s] (周期の 1/100 以下にもする),
    warmup: 集める前に回す時間 [s], min_collect: イオンを集める時間の下限 [s] (RF 周期の整数倍に切り上げる),
    l_factor: 領域の長さ L = l_factor x (V1 の最大値での Child 則のシース厚) + 5 λ_D,
    engine: auto (Numba があれば CPU JIT)、numba、numpy (比較用の参照実装)。"""
    if engine not in ("auto", "numba", "numpy"):
        raise ValueError("シースの engine は auto / numba / numpy を指定してください")
    simulate = None
    if engine != "numpy":
        try:
            from sheath_fast import simulate
        except ImportError:
            if engine == "numba":
                raise RuntimeError("Numba を利用できません。uv sync で依存関係を同期してください") from None
    engine_used = "numba" if simulate is not None else "numpy"
    t0_wall = time.time()
    Te, M = el["Te"], el["m_i"]
    T = t[1] * t.size
    tg = np.append(t, T)
    Vg = np.append(V1, V1[0])
    u_B = np.sqrt(E_CHARGE * Te / M)
    n_s = flux / u_B                                     # シース端の密度
    lam_D = np.sqrt(EPS0 * Te / (E_CHARGE * n_s))        # デバイ長
    L = l_factor * _child_thickness(V1.max(), el) + 5 * lam_D
    N = int(np.ceil(L / (dy_per_debye * lam_D)))
    dy = L / N
    y_nodes = np.arange(N + 1) * dy
    dt = min(dt_max, T / 100)
    w = n_s * dy / ppc                                   # マクロ粒子 1 個が表す面密度 [m^-2]
    qm = E_CHARGE / M
    k_e = E_CHARGE / EPS0
    v_th = np.sqrt(E_CHARGE * ion_temp / M)              # 面内の熱速度
    rng = np.random.default_rng(seed)

    # 衝突 (南部–北谷モデル): 相対エネルギーの上限 ~ (最大のイオンエネルギー)/2 で β_m を決める
    kT_gas = K_B * gas_temp
    beta_m = nk_beta_m(0.5 * (V1.max() + Te / 2 + 3 * ion_temp) + 1.0)
    P_col = -np.expm1(-pressure / kT_gas * nk_rate(M, beta_m) * dt) if pressure > 0 else 0.0
    n_col = n_cx = n_over = 0

    # 初期状態: 密度 n_s, 速度 -u_B の一様なイオン
    n0 = int(round(n_s * L / w))
    y = rng.uniform(0, L, n0)
    v = np.full(n0, -u_B)
    vx = rng.normal(0.0, v_th, n0)
    vz = rng.normal(0.0, v_th, n0)

    # Poisson (内部の格子点 1..N-1) の三重対角行列。対角は Newton 反復ごとに更新する
    m = N - 1
    ab = np.zeros((3, m))
    ab[0, 1:] = 1 / dy ** 2
    ab[2, :-1] = 1 / dy ** 2
    phi = np.zeros(N + 1)
    phi[0] = -np.interp(np.mod(-warmup, T), tg, Vg)

    def solve_phi(ni, phi_left):
        phi[0] = phi_left
        p_in = phi[1:-1]
        for _ in range(20):
            ex = np.exp(np.minimum(p_in / Te, 50.0))
            lap = np.empty(m)
            lap[:] = -2 * p_in
            lap[1:] += p_in[:-1]
            lap[:-1] += p_in[1:]
            lap[0] += phi[0]
            lap /= dy ** 2
            F = lap + k_e * (ni[1:-1] - n_s * ex)
            ab[1] = -2 / dy ** 2 - k_e * n_s * ex / Te
            d = solve_banded((1, 1), ab, -F)
            p_in += d
            if np.abs(d).max() < 1e-5:
                break
        else:
            raise RuntimeError("1 次元シースの Poisson 方程式 (Newton 法) が収束しませんでした")
        return phi

    n_collect = max(1, int(np.ceil(min_collect / T)))
    n_steps = int(np.ceil((warmup + n_collect * T) / dt))
    t_start = -warmup
    inj_rate = flux * dt / w                             # 1 ステップに入れるマクロ粒子の数 (平均)
    inj_acc = 0.0
    rec = []
    R = None
    if log is not None:
        log(f"1 次元シース: {'CPU JIT (初回はコンパイル)' if engine_used == 'numba' else 'NumPy'} / "
            f"{N:,} セル / 初期 {n0:,} 粒子 / {n_steps:,} ステップ")
    if simulate is not None:
        R, n_col, n_cx, n_over = simulate(tg, Vg, phi, y, v, vx, vz, rng, Te, M, n_s, dy, dt, L, T,
                                         w, qm, k_e, warmup, n_steps, inj_rate, u_B, v_th, P_col,
                                         kT_gas, beta_m, E_CHARGE, NK_BETA0, NK_A)
    for k in range(n_steps if R is None else 0):
        tk = t_start + k * dt
        # イオン密度 (CIC)
        g = y / dy
        j = np.minimum(g.astype(np.int64), N - 1)
        f = g - j
        ni = (np.bincount(j, 1 - f, N + 1) + np.bincount(j + 1, f, N + 1)) * (w / dy)
        ni[0] *= 2                                       # 境界の格子点は半セル分
        ni[-1] *= 2
        phi = solve_phi(ni, -np.interp(np.mod(tk, T), tg, Vg))
        E_nodes = np.empty(N + 1)
        E_nodes[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dy)
        E_nodes[0] = -(phi[1] - phi[0]) / dy
        E_nodes[-1] = -(phi[-1] - phi[-2]) / dy
        Ep = E_nodes[j] * (1 - f) + E_nodes[j + 1] * f
        # leapfrog
        v_old = v
        v = v + qm * Ep * dt
        if P_col > 0:                                    # 衝突 (速度によらない確率)
            c = np.flatnonzero(rng.random(v.size) < P_col)
            if c.size:
                new, cx, over = nk_scatter(np.stack([vx[c], v[c], vz[c]], axis=1), rng, M, kT_gas, beta_m)
                vx[c], v[c], vz[c] = new[:, 0], new[:, 1], new[:, 2]
                n_over += over
                if tk >= 0:                              # 集める期間の衝突だけ数える (イオン 1 個あたりの回数用)
                    n_col += c.size
                    n_cx += int(cx.sum())
        y_new = y + v * dt
        hit = y_new <= 0
        if hit.any() and tk >= 0:
            vn = 0.5 * (v_old[hit] + v[hit])
            phi_y = phi[j[hit]] * (1 - f[hit]) + phi[j[hit] + 1] * f[hit]
            vn2 = np.maximum(vn ** 2 + 2 * qm * (phi_y - phi[0]), 0.0)   # 電極での法線速度の 2 乗
            rec.append(np.stack([np.sqrt(vn2), vx[hit], vz[hit]], axis=1))
        keep = (~hit) & (y_new < L)
        y, v, vx, vz = y_new[keep], v[keep], vx[keep], vz[keep]
        # プラズマ側からの注入 (時間的に一様)
        inj_acc += inj_rate
        n_new = int(inj_acc)
        inj_acc -= n_new
        if n_new:
            y = np.concatenate([y, L - u_B * dt * rng.uniform(0, 1, n_new)])
            v = np.concatenate([v, np.full(n_new, -u_B)])
            vx = np.concatenate([vx, rng.normal(0.0, v_th, n_new)])
            vz = np.concatenate([vz, rng.normal(0.0, v_th, n_new)])

    if R is None:
        R = np.concatenate(rec) if rec else np.zeros((0, 3))
    n_hit = R.shape[0]
    if n_hit > MAX_STORE:                                # 記録は等しい重みなので、間引いても分布は同じ
        R = R[np.sort(rng.choice(n_hit, MAX_STORE, replace=False))]
    vel = dict(vn=R[:, 0], vx=R[:, 1], vz=R[:, 2])
    E = 0.5 * M * (R ** 2).sum(axis=1) / E_CHARGE
    info = dict(L=L, dy=dy, N=N, dt=dt, steps=n_steps, lam_D=lam_D, n_s=n_s, ions=n_hit, stored=int(E.size),
                periods=n_collect, seconds=time.time() - t0_wall, y=y_nodes, phi=phi.copy(), engine=engine_used,
                pressure=pressure, gas_temp=gas_temp, beta_m=beta_m,
                collisions_per_ion=n_col / max(n_hit, 1) if pressure > 0 else 0.0,
                cx_fraction=n_cx / max(n_col, 1), beta_over=n_over)
    if log is not None:
        log(f"1 次元シース IED 完了: {info['seconds']:.3f} 秒 / 到達 {n_hit:,} イオン")
    return E, vel, info
