#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF バイアスのプラズマ等価回路 (集中定数モデル)
==============================================

Yu et al., Plasma Sources Sci. Technol. 31, 035012 (2022) の従来型の等価回路 (Fig. 4(b)) を基に、
ブロッキングコンデンサとウェハ上の膜 (SiO2 + マスク) を直列に入れた回路を解く:

    RF 電源 Vrf sin(2πft) ─ C_b ─ ウェハ (Si) ─ C_stack (SiO2 [+ 誘電体マスク]) ─ 表面 (マスク)
        ─ ウェハ側シース ─ プラズマ ─ 壁側シース (面積はウェハの wall_ratio 倍) ─ 接地

* シースは 3 つの要素の並列:
    イオン電流源      I_i = e Γ_i A              (Γ_i = Params.flux, Bohm フラックス)
    電子のダイオード  I_e = e Γ_e A exp(-V/Te)    (Γ_e = n_s v_e / 4, シース端の密度 n_s = Γ_i / u_B)
    非線形の容量      Q = A K (V + Te)^(1/4)      (Child 則のシースが蓄える電荷。K = 2 sqrt(ε0 e Γ_i) (M/2e)^(1/4)。
                                                    + Te はシースが崩壊する付近の正則化)
* 素子はすべて直列なので電流は 1 本。状態変数は直列コンデンサ (C_b, C_stack) の電荷 q と
  ウェハ側シース電圧 V1 で、壁側シース電圧は KVL から V0 = u_s - q / C_s + V1 (C_s: C_b と C_stack の直列)。
  電子電流が電圧の指数関数なので硬い ODE になる。LSODA で RF 周期ごとに積分し、周期定常になるまで回す。
* 周期定常では C_b に流れる電流の平均が 0 (各シースでイオン電流と電子電流の平均が釣り合う) になり、
  C_b に直流電圧 (自己バイアス) が溜まる。
* イオンエネルギー分布 (IED): 時刻 t0 にシースへ入ったイオンが得るエネルギー
      E(t0) = e <V1>(t0) + e Te / 2      (Te / 2: プリシースで得る分)
  <V1> は論文のとおりシース通過時間 τ_i での平均で、箱型の平均の代わりに一次遅れ (時定数 τ_i / 2。
  k 次の高調波に 1 / (1 - i k ω τ_i / 2) を掛ける) で近似する。高い周波数での IED の幅は箱型の平均と同じ
  2 / (ω τ_i) 倍に縮み、箱型にある不自然なゼロ点がない。ied_model="instant" なら瞬時値 (論文 式 (5))。
  τ_i = 3 s sqrt(M / (2 e <V1>)) (s: 平均電圧での Child 則のシース厚)。イオン電流は時間的に一定なので、
  t0 は RF 周期の中で一様に分布する。
"""
import numpy as np
from scipy.integrate import solve_ivp

# 物理定数 (trench_charging_2d.py と同じ値)
E_CHARGE = 1.602176634e-19   # [C]
EPS0 = 8.8541878128e-12      # [F/m]
M_E = 9.1093837015e-31       # [kg]
AMU = 1.66053906660e-27      # [kg]

BIAS_TYPES = ("rf", "dc")             # bias の選択肢 (dc: 全イオンが ion_energy_eV)
IED_MODELS = ("transit", "instant")   # ied_model の選択肢


def circuit_elements(p):
    """回路の素子の値 (dict)。p は trench_charging_2d.Params。"""
    m_i = p.ion_mass_amu * AMU
    Te = p.electron_temp_eV
    A1 = np.pi * (p.wafer_d / 2) ** 2              # ウェハ (電極) の面積 [m^2]
    J_i = E_CHARGE * p.flux                         # イオン電流密度 [A/m^2]
    u_B = np.sqrt(E_CHARGE * Te / m_i)              # Bohm 速度
    J_e = E_CHARGE * 0.25 * (p.flux / u_B) * np.sqrt(8 * E_CHARGE * Te / (np.pi * M_E))  # 電子の熱電流密度
    d_eff = (p.floor_t + p.trench_d) * p.dx / p.eps_r   # 膜の等価な真空の厚さ (マスクの下の SiO2)
    if p.mask_t > 0 and p.mask_type == "dielectric":
        d_eff += p.mask_t * p.dx / p.mask_eps_r
    # 開口 (トレンチ) の部分はウェハとの容量が小さいので、マスクが覆う面積だけ数える
    C_stack = EPS0 * A1 * (1 - p.trench_w / p.nx) / d_eff
    return dict(A1=A1, A0=p.wall_ratio * A1, J_i=J_i, J_e=J_e, Te=Te, m_i=m_i,
                K=2 * np.sqrt(EPS0 * J_i) * (m_i / (2 * E_CHARGE)) ** 0.25,
                C_b=p.c_block, C_stack=C_stack, C_s=1 / (1 / p.c_block + 1 / C_stack))


def _sheath_cap(V, A, el):
    """シースの微分容量 dQ/dV [F] (Child 則, V < 0 では V = 0 の値)。"""
    return A * el["K"] / (4 * (np.maximum(V, 0.0) + el["Te"]) ** 0.75)


def _electron_current(V, A, el):
    """シースを越えて表面に入る電子の電流 [A] (ボルツマン因子, オーバーフロー防止に指数を上限で切る)。"""
    return el["J_e"] * A * np.exp(np.minimum(-V / el["Te"], 60.0))


def _rhs(el, Vrf, w):
    """状態 x = (q, V1) の時間微分を返す関数。
    KVL を時間微分した式から直列電流 i を求める:
        C0 dV0/dt = i - (I_e0 - I_i0),  C1 dV1/dt = (I_e1 - I_i1) - i,  dV0/dt = du_s/dt - i / C_s + dV1/dt"""
    A1, A0, C_s = el["A1"], el["A0"], el["C_s"]
    I_i1, I_i0 = el["J_i"] * A1, el["J_i"] * A0

    def rhs(t, x):
        q, V1 = x
        us, dus = Vrf * np.sin(w * t), Vrf * w * np.cos(w * t)
        V0 = us - q / C_s + V1
        C1, C0 = _sheath_cap(V1, A1, el), _sheath_cap(V0, A0, el)
        d1 = _electron_current(V1, A1, el) - I_i1     # ウェハ側シースの (電子 - イオン) 電流
        d0 = _electron_current(V0, A0, el) - I_i0     # 壁側シース
        i = (C0 * dus + C0 / C1 * d1 - d0) / (1 + C0 / C_s + C0 / C1)
        return [i, (d1 - i) / C1]
    return rhs


def solve_circuit(p, n_phase=1024, max_periods=3000):
    """RF 周期定常まで回路を解き、1 周期分の波形と IED を dict で返す。
    t: 位相 0 からの時刻 [s], us/uw/um/up: RF 電源・ウェハ・表面 (マスク)・プラズマの電位 [V],
    V1: ウェハ側シース電圧 [V], i: 直列電流 [A], E: シースに入った時刻ごとのイオンエネルギー [eV]"""
    if p.ied_model not in IED_MODELS:
        raise ValueError(f"ied_model は {' / '.join(IED_MODELS)} のどちらかにしてください: {p.ied_model!r}")
    el = circuit_elements(p)
    Te = el["Te"]
    w = 2 * np.pi * p.rf_freq
    T = 1.0 / p.rf_freq
    rhs = _rhs(el, p.rf_volt, w)
    atol = [1e-9 * el["C_s"], 1e-7]                 # q [C], V1 [V]

    x = np.array([0.0, Te])
    for k in range(max_periods):
        sol = solve_ivp(rhs, (k * T, (k + 1) * T), x, method="LSODA", rtol=1e-7, atol=atol)
        if not sol.success:
            raise RuntimeError(f"等価回路の積分に失敗しました: {sol.message}")
        dq, dV = np.abs(sol.y[:, -1] - x)
        x = sol.y[:, -1]
        if k >= 3 and dq < 1e-6 * el["C_b"] and dV < 1e-4:   # 1 周期での変化が C_b で 1 μV, V1 で 0.1 mV 未満
            break
    else:
        raise RuntimeError(f"等価回路が {max_periods} 周期で周期定常になりませんでした")

    t = np.arange(n_phase) * T / n_phase
    sol = solve_ivp(rhs, (0.0, T), x, method="LSODA", rtol=1e-9, atol=[a / 100 for a in atol], t_eval=t)
    if not sol.success:
        raise RuntimeError(f"等価回路の積分に失敗しました: {sol.message}")
    q, V1 = sol.y
    us = p.rf_volt * np.sin(w * t)
    uw = us - q / el["C_b"]                         # ウェハ (Si)
    um = uw - q / el["C_stack"]                     # 表面 (マスク)
    up = um + V1                                    # プラズマ (= 壁側シース電圧)
    i = np.array([rhs(tt, xx)[0] for tt, xx in zip(t, sol.y.T)])

    # イオンエネルギー: シース通過時間での平均 (一次遅れ) または瞬時値
    V1m = V1.mean()
    s = (4 / 3) * EPS0 * V1m / (el["K"] * (V1m + Te) ** 0.25)   # Child 則: 表面の電場 4V/(3s) = 電荷 / ε0
    tau_i = 3 * s * np.sqrt(el["m_i"] / (2 * E_CHARGE * V1m))
    if p.ied_model == "transit":
        k_h = np.arange(n_phase // 2 + 1)
        V_eff = np.fft.irfft(np.fft.rfft(V1) / (1 - 1j * k_h * w * tau_i / 2), n_phase)
    else:
        V_eff = V1
    return dict(t=t, us=us, uw=uw, um=um, up=up, V1=V1, i=i, E=V_eff + Te / 2,
                Ie1=_electron_current(V1, el["A1"], el), I_i1=el["J_i"] * el["A1"],
                tau_i=tau_i, s=s, n_periods=k + 1, elements=el)


def ion_energy_sampler(c):
    """RF の位相を一様に選び、IED からイオンエネルギー [eV] を返す関数 sample(rng, n) を作る。"""
    E = c["E"]
    grid = np.arange(E.size + 1)
    Ep = np.append(E, E[0])                         # 周期的に補間する
    return lambda rng, n: np.interp(rng.uniform(0, E.size, n), grid, Ep)


def summary(c):
    """ログ用の説明 (2 行)。"""
    return (f"RF バイアス: 自己バイアス (ウェハの直流電位) {c['uw'].mean():.1f} V, プラズマ電位 {c['up'].mean():.1f} V, "
            f"ウェハ側シース電圧 {c['V1'].min():.0f}〜{c['V1'].max():.0f} V (平均 {c['V1'].mean():.1f} V)\n"
            f"  シース厚 {c['s']*1e6:.0f} μm, イオン通過時間 {c['tau_i']*1e9:.0f} ns, "
            f"IED {c['E'].min():.0f}〜{c['E'].max():.0f} eV (平均 {c['E'].mean():.1f} eV), "
            f"周期定常まで {c['n_periods']} 周期")


def plot_circuit(c, ax_wave, ax_ied):
    """周期定常の波形 (ax_wave) と IED (ax_ied) を描く。"""
    t = c["t"] * 1e9
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ax_wave.plot(t, c["us"], color=colors[0], lw=1.5, label="RF source")
    ax_wave.plot(t, c["uw"], color=colors[1], lw=1.5, label="wafer = mask surface")
    ax_wave.plot(t, c["up"], color=colors[2], lw=1.5, label="plasma")
    ax_wave.plot(t, c["V1"], color=colors[3], lw=2, label="wafer sheath voltage")
    ax_wave.axhline(0, color="0.6", lw=0.6)
    ax_wave.set_xlabel("time in one RF period [ns]")
    ax_wave.set_ylabel("potential [V]")
    ax_wave.set_title("Equivalent circuit (periodic steady state)", fontsize=10)
    ax_wave.legend(fontsize=7, loc="lower left")
    ax_wave.grid(alpha=0.3)

    E = c["E"]
    bins = np.linspace(E.min() - 1, E.max() + 1, 60)
    ax_ied.hist(E, bins=bins, density=True, color=colors[0], alpha=0.85)
    ax_ied.axvline(E.mean(), color="0.3", ls="--", lw=1)
    ax_ied.text(E.mean(), ax_ied.get_ylim()[1] * 0.95, f" mean {E.mean():.0f} eV", fontsize=8, va="top")
    ax_ied.set_xlabel("ion energy [eV]")
    ax_ied.set_ylabel("probability density [1/eV]")
    ax_ied.set_title(f"IED (ion transit {c['tau_i']*1e9:.0f} ns)", fontsize=10)
    ax_ied.grid(alpha=0.3)
