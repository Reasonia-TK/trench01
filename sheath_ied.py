#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1 次元シースの時間発展からイオンエネルギー分布 (IED) を求める (ied_model="sheath")
================================================================================

等価回路 (plasma_circuit.py) が出したウェハ側シース電圧 V1(t) を境界条件にして、
電極 (ウェハ表面) とプラズマの間の 1 次元・静電・無衝突のシースを時間発展させる。

* 領域: 電極 y=0 から プラズマ側 y=L まで (L は V1 の最大値での Child 則のシース厚の 1.5 倍 + 5 λ_D。
  実際のシースは Child 則より 1.6〜1.9 λ_D 厚いが、十分に収まる)。
  境界は φ(0) = -V1(t) (電極), φ(L) = 0 (プラズマ)。
* イオン: 粒子 (マクロ粒子)。y=L から Bohm 速度 u_B で、一定のフラックス Γ (Params.flux) で入れる。
  電極に着いたら吸収し、そのときの運動エネルギーを記録する。
* 電子: ボルツマン分布 n_e = n_s exp(φ/Te) (電位の変化に瞬時に応答)。n_s = Γ / u_B はシース端の密度。
* Poisson 方程式 ε0 φ'' = -e (n_i - n_e(φ)) は電子のせいで非線形なので、毎ステップ Newton 法で解く
  (三重対角)。イオン密度は粒子から線形の重み (CIC) で格子点に集める。
* 時間は leapfrog。電極に着いたイオンのエネルギーは、最後のステップの中での電位差で補正する
  (E = M v^2 / 2 + e (φ(y_n) - φ(0)), v は時刻 n の速度)。
* 最初は領域を密度 n_s・速度 u_B のイオンで満たし、warmup (イオンが領域を何度も入れ替わる時間) だけ
  回してから、RF 周期の整数倍 (min_collect 以上) の間に電極に着いたイオンを集める。
  イオン電流は時間的に一定で注入するので、集めたイオンはそのまま等しい重みの IED になる。

パルスでシースが一気に広がる動き (イオンのマトリクスシース) や、イオンが電圧に追従できるかどうか
(周波数とイオンのプラズマ周波数の比) を、粒子の運動として直接扱う。
"""
import time

import numpy as np
from scipy.linalg import solve_banded

E_CHARGE = 1.602176634e-19   # [C]   (trench_charging_2d.py と同じ値)
EPS0 = 8.8541878128e-12      # [F/m]


def _child_thickness(V, el):
    """Child 則のシース厚 [m] (等価回路のシース容量と同じ式)。"""
    V = np.maximum(V, 0.0)
    return (4 / 3) * EPS0 * V / (el["K"] * (V + el["Te"]) ** 0.25)


def ion_energies(t, V1, el, flux, ppc=200, dy_per_debye=0.25, dt_max=0.25e-9, warmup=2e-6,
                 min_collect=1e-6, l_factor=1.5, seed=0):
    """シース電圧の 1 周期分の波形 V1(t) (t: 等間隔, 周期 = t[1] * len(t)) から、電極に着いたイオンの
    エネルギー [eV] の配列と、計算条件の dict を返す。
    el: plasma_circuit.circuit_elements の dict (Te, m_i, K を使う), flux: イオンフラックス [m^-2 s^-1]
    ppc: プラズマ側の 1 セルあたりのマクロ粒子数, dy_per_debye: 格子間隔 / デバイ長,
    dt_max: 時間刻みの上限 [s] (周期の 1/100 以下にもする), warmup: 集める前に回す時間 [s],
    min_collect: イオンを集める時間の下限 [s] (RF 周期の整数倍に切り上げる),
    l_factor: 領域の長さ L = l_factor x (V1 の最大値での Child 則のシース厚) + 5 λ_D"""
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
    rng = np.random.default_rng(seed)

    # 初期状態: 密度 n_s, 速度 -u_B の一様なイオン
    n0 = int(round(n_s * L / w))
    y = rng.uniform(0, L, n0)
    v = np.full(n0, -u_B)

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
    energies = []
    for k in range(n_steps):
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
        y_new = y + v * dt
        hit = y_new <= 0
        if hit.any() and tk >= 0:
            vn = 0.5 * (v_old[hit] + v[hit])
            phi_y = phi[j[hit]] * (1 - f[hit]) + phi[j[hit] + 1] * f[hit]
            energies.append(0.5 * M * vn ** 2 / E_CHARGE + (phi_y - phi[0]))
        keep = (~hit) & (y_new < L)
        y, v = y_new[keep], v[keep]
        # プラズマ側からの注入 (時間的に一様)
        inj_acc += inj_rate
        n_new = int(inj_acc)
        inj_acc -= n_new
        if n_new:
            y = np.concatenate([y, L - u_B * dt * rng.uniform(0, 1, n_new)])
            v = np.concatenate([v, np.full(n_new, -u_B)])

    E = np.concatenate(energies) if energies else np.zeros(0)
    info = dict(L=L, dy=dy, N=N, dt=dt, steps=n_steps, lam_D=lam_D, n_s=n_s, ions=E.size,
                periods=n_collect, seconds=time.time() - t0_wall, y=y_nodes, phi=phi.copy())
    return E, info
