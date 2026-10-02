#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トレンチ表面帯電シミュレーター GUI (Tkinter + matplotlib)
=========================================================

trench_charging_2d.py と同じフォルダに置いて実行する:

    uv run trench_charging_gui.py

* 左の「形状 / 電場 / プラズマ / 計算」で条件を設定し、[計算を実行] を押す。
* 上部の指標と、電位・電場 / 左右側壁 / 時間履歴のタブをリアルタイム更新する。
  [停止] で中断できる(それまでの結果は残る)。[図を拡大] で選択中の図を大きく確認できる。
* 実行モードは「固定長」と「飽和まで継続」から選べる。
* [結果を保存] で場・履歴・左右電荷の図と再利用可能な .npz を保存できる (RF なら _rf.png も)。
* [垂直イオンを追跡] で帯電あり/なしの軌道・底フラックス・偏向角を比較し、_ions.png に保存。
* [回路・IED を確認] で、入力条件の RF 回路波形とイオンエネルギー分布 (IED) を専用タブに表示する。
* [設定を保存/読込] でパラメータを JSON で保存・復元できる。

依存パッケージ (numpy, scipy, matplotlib) は uv で管理している (pyproject.toml / uv.lock)。
Tkinter は Python に同梱 (Linux のシステム Python では `sudo apt install python3-tk` が必要な場合がある)
"""
import json
import gc
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict, replace

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import trench_charging_2d as T
import plasma_circuit as PC


# ---------------------------------------------------------------- 入力欄の定義
# (キー, ラベル, 単位, 表示スケール(内部値 = 表示値 x scale), 型 / 選択肢)
GEOMETRY = [
    ("nx", "横セル数 (周期境界)", "セル", 1, int),
    ("dx", "セルサイズ", "nm", 1e-9, float),
    ("trench_w", "トレンチ幅 (SiO2 上面)", "セル", 1, int),
    ("trench_offset", "トレンチ中心の移動 (+右 / -左)", "セル", 1, int),
    ("trench_taper_deg", "テーパー角 (鉛直から / +底を狭く)", "deg", 1, float),
    ("trench_d", "トレンチ深さ", "セル", 1, int),
    ("floor_t", "底の誘電体の厚さ", "セル", 1, int),
    ("n_vac", "上部の真空領域", "セル", 1, int),
    ("eps_r", "比誘電率 (SiO2 = 3.9)", "", 1, float),
    ("top_bc", "上端の境界条件", "", 1, ("dirichlet", "neumann")),
]
MASK = [
    ("mask_t", "マスク厚 (0 = なし)", "セル", 1, int),
    ("mask_taper_deg", "テーパー角 (鉛直から / +上を広く)", "deg", 1, float),
    ("mask_type", "マスクの種類", "", 1, T.MASK_TYPES),
    ("mask_eps_r", "マスクの比誘電率 (dielectric)", "", 1, float),
]
FIELDS = [
    ("field_mode", "電場モード (RF 全周期: circuit_rf)", "", 1, T.FIELD_MODES),
    ("rf_phase", "RF 位相 (circuit:固定 / circuit_rf:表示)", "deg", 1, float),
    ("rf_time_steps", "RF 1周期の時間分割数 (粒子 dt 上限)", "分割", 1, int),
    ("substrate_voltage", "基板電位 (fixed / floating)", "V", 1, float),
    ("mask_voltage", "マスク電位 (fixed)", "V", 1, float),
    ("top_voltage", "上端電位 (fixed / floating)", "V", 1, float),
    ("side_bc", "SiO2 両端 (pillar: 線形補間)", "", 1, T.SIDE_BCS),
    ("left_voltage", "SiO2 左端電位 (fixed)", "V", 1, float),
    ("right_voltage", "SiO2 右端電位 (fixed)", "V", 1, float),
]
PROBE = [
    ("probe_ions", "垂直イオン数 (開口一様)", "個", 1, int),
    ("probe_energy_eV", "垂直イオン入射エネルギー", "eV", 1, float),
    ("probe_trajectories", "表示・保存する軌道数", "本", 1, int),
]
PLASMA = [
    ("flux", "粒子フラックス", "m⁻²s⁻¹", 1, float),
    ("ion_mass_amu", "イオン質量 (Ar = 40)", "amu", 1, float),
    ("ion_energy_eV", "イオンエネルギー (dc)", "eV", 1, float),
    ("ion_temp_eV", "イオン横方向温度", "eV", 1, float),
    ("electron_temp_eV", "電子温度", "eV", 1, float),
]
RF = [
    ("bias", "バイアス (rf: 等価回路)", "", 1, PC.BIAS_TYPES),
    ("rf_wave", "波形 (pulse: 負のパルス)", "", 1, PC.WAVE_TYPES),
    ("rf_freq", "RF 周波数", "MHz", 1e6, float),
    ("rf_volt", "振幅 / パルスの高さ", "V", 1, float),
    ("rf_duty", "パルス幅 (周期に対する割合)", "%", 0.01, float),
    ("rf_rise", "パルスの立ち上がり時間", "ns", 1e-9, float),
    ("c_block", "ブロッキングコンデンサ", "pF", 1e-12, float),
    ("wafer_d", "ウェハ直径", "mm", 1e-3, float),
    ("wall_ratio", "壁 / ウェハの面積比", "", 1, float),
    ("ied_model", "IED のモデル", "", 1, PC.IED_MODELS),
    ("gas_pressure", "ガス圧力 (シースの衝突, 0=なし)", "Pa", 1, float),
    ("gas_temp", "ガス温度", "K", 1, float),
]
RF_KEYS = {f[0] for f in RF} - {"bias"}
PULSE_KEYS = {"rf_duty", "rf_rise"}
SHEATH_KEYS = {"gas_pressure", "gas_temp"}      # 1 次元シース (ied_model=sheath) のときだけ使う
LEAK = [
    ("sigma_s", "表面シート伝導度 (0=なし)", "S", 1, float),
]
NUMERICS = [
    ("backend", "粒子追跡 (cpu / cuda / auto)", "", 1, T.BACKENDS),
    ("cuda_device", "CUDA デバイス番号", "", 1, int),
    ("n_per_batch", "1バッチの粒子数 (各種)", "個", 1, int),
    ("dt_batch", "1バッチの物理時間", "μs", 1e-6, float),
    ("cfl", "CFL (1ステップの移動量)", "セル", 1, float),
    ("max_steps", "粒子の最大ステップ数", "", 1, int),
    ("wall_model", "壁際の扱い (barrier: 電位障壁で反射)", "", 1, T.WALL_MODELS),
    ("seed", "乱数シード", "", 1, int),
]
FIXED_FIELDS = [
    ("n_batches", "バッチ数 / 再開時の追加数", "", 1, int),
]
STEADY_FIELDS = [
    ("max_batches", "上限バッチ数（通算）", "", 1, int),
    ("steady_window", "判定ブロック長", "バッチ", 1, int),
    ("steady_rtol", "電位の許容変化率", "", 1, float),
    ("steady_charge_rtol", "電荷平均の許容変化率", "%", 0.01, float),
    ("steady_charge_atol", "電荷の絶対許容値", "pC/m", 1e-12, float),
    ("steady_hold", "連続 OK 回数", "回", 1, int),
]
FIXED_KEYS = {f[0] for f in FIXED_FIELDS}
STEADY_KEYS = {f[0] for f in STEADY_FIELDS}


def fmt(v):
    return str(v) if isinstance(v, int) else f"{v:.10g}"


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 60}分{sec % 60:02d}秒" if sec >= 60 else f"{sec}秒"


def validate(p):
    """入力値のチェック。問題があればメッセージ(str)、なければ None を返す。"""
    if p.nx < 6:
        return "横セル数は 6 以上にしてください。"
    if not (1 <= p.trench_w <= p.nx - 2):
        return "トレンチ幅は 1 以上、(横セル数 - 2) 以下にしてください。"
    if any(not np.isfinite(v) for v in asdict(p).values() if isinstance(v, (int, float))):
        return "数値には有限の値を指定してください。"
    i0 = (p.nx - p.trench_w) // 2 + p.trench_offset
    if i0 < 1 or i0 + p.trench_w > p.nx - 1:
        return "トレンチ移動後も左右に 1 セル以上の固体を残してください。"
    if p.trench_d < 1 or p.floor_t < 1:
        return "トレンチ深さと底の厚さは 1 セル以上にしてください。"
    if p.n_vac < 5:
        return "上部の真空領域は 5 セル以上にしてください。"
    if p.dx <= 0 or p.eps_r < 1:
        return "セルサイズは正、比誘電率は 1 以上にしてください。"
    if p.mask_t < 0:
        return "マスク厚は 0 以上にしてください。"
    try:
        T.build_geometry(p)
    except ValueError as e:
        return str(e)
    if p.mask_type == "dielectric" and p.mask_eps_r < 1:
        return "マスクの比誘電率は 1 以上にしてください。"
    if p.field_mode != "floating" and p.mask_t > 0 and p.mask_type != "conductor":
        return "外部電位を与えるマスクは conductor にしてください。"
    if p.field_mode in T.CIRCUIT_FIELDS and (p.bias != "rf" or p.top_bc != "dirichlet"):
        return "回路電位モードには bias=rf、上端 dirichlet が必要です。"
    if p.side_bc == "pillar" and p.field_mode == "floating":
        return "ピラー境界には電場モード fixed / circuit / circuit_rf が必要です。"
    if p.field_mode == "circuit_rf":
        if p.rf_time_steps < 16:
            return "RF 1周期の時間分割数は 16 以上にしてください。"
        if p.rf_freq > 0 and p.dt_batch * p.rf_freq < 1 - 1e-12:
            return "RF 全周期モードでは 1バッチの時間を RF 1周期以上にしてください。"
    if (p.probe_ions < 1 or p.probe_energy_eV <= 0
            or not 1 <= p.probe_trajectories <= p.probe_ions):
        return "診断のイオン数とエネルギーは正、軌道数はイオン数以下にしてください。"
    if p.flux <= 0 or p.ion_mass_amu <= 0 or p.ion_energy_eV <= 0 or p.electron_temp_eV <= 0:
        return "フラックス・イオン質量・イオンエネルギー・電子温度は正の値にしてください。"
    if p.ion_temp_eV < 0 or p.sigma_s < 0:
        return "イオン温度と表面シート伝導度は 0 以上にしてください。"
    if p.backend not in T.BACKENDS or p.cuda_device < 0:
        return "計算方式を cpu / cuda / auto から選び、CUDA デバイス番号を 0 以上にしてください。"
    if p.bias == "rf" and min(p.rf_freq, p.c_block, p.wafer_d, p.wall_ratio) <= 0:
        return "RF 周波数・ブロッキングコンデンサ・ウェハ直径・面積比は正の値にしてください。"
    if p.bias == "rf" and p.rf_volt < 0:
        return "RF 振幅 (パルスの高さ) は 0 以上にしてください。"
    if p.bias == "rf" and p.ied_model == "sheath" and (p.gas_pressure < 0 or p.gas_temp <= 0):
        return "ガス圧力は 0 以上、ガス温度は正の値にしてください。"
    if p.bias == "rf" and p.rf_wave == "pulse":
        if not 0 < p.rf_duty < 1:
            return "パルス幅は 0% より大きく 100% より小さくしてください。"
        if not 0 < p.rf_rise <= min(p.rf_duty, 1 - p.rf_duty) / p.rf_freq:
            return (f"パルスの立ち上がり時間は 0 より大きく、パルス幅とパルスの間隔 "
                    f"({min(p.rf_duty, 1 - p.rf_duty) / p.rf_freq * 1e9:.3g} ns) 以下にしてください。")
    if p.n_per_batch < 100 or p.dt_batch <= 0:
        return "1バッチの粒子数は 100 以上、物理時間は正の値にしてください。"
    if not (0 < p.cfl <= 1) or p.max_steps < 100:
        return "CFL は 0 より大きく 1 以下、最大ステップ数は 100 以上にしてください。"
    if p.until_steady:
        if p.steady_window < 5 or p.steady_hold < 1 or p.steady_rtol <= 0:
            return "判定ブロック長は 5 以上、連続 OK 回数は 1 以上、許容変化率は正の値にしてください。"
        if p.max_batches < 2 * p.steady_window:
            return "上限バッチ数は 判定ブロック長 の 2 倍以上にしてください。"
        if p.steady_charge_rtol <= 0 or p.steady_charge_atol < 0:
            return "電荷の許容変化率は正、絶対許容値は0以上にしてください。"
    elif p.n_batches < 1:
        return "バッチ数は 1 以上にしてください。"
    return None


# ---------------------------------------------------------------- リアルタイム描画
COLORS = dict(bg="#f2f5fa", card="#ffffff", text="#172b4d", muted="#64748b",
              border="#dce4ef", blue="#2563eb", amber="#d97706", green="#15803d")


def style_figure(fig):
    """Tk のダッシュボードと同じ配色。計算・保存用の図の設定は変更しない。"""
    fig.set_facecolor(COLORS["card"])
    for ax in fig.axes:
        ax.set_facecolor(COLORS["card"])
        ax.tick_params(colors=COLORS["muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(COLORS["text"])


class LivePlot:
    """図の artist を最初に作っておき、データだけ差し替えて高速に更新する。"""

    def __init__(self, fig, wall_fig, history_fig):
        self.fig, self.wall_fig, self.history_fig = fig, wall_fig, history_fig
        self.ready = False
        self.vlines = []
        self.lines = {}

    def setup(self, p, solid, mask, i0):
        fig = self.fig
        for f in (fig, self.wall_fig, self.history_fig):
            f.clear()
        self.p, self.solid, self.i0 = p, solid, i0
        self.vlines, self.lines = [], {}
        self.compact_history = self.compact_walls = False
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("compressed")
        nx, ny = solid.shape
        dxn = p.dx * 1e9
        self.dxn = dxn
        xc = (np.arange(nx) + 0.5) * dxn
        yc = (np.arange(ny) + 0.5) * dxn
        extent = [0, nx * dxn, 0, ny * dxn]
        gs = fig.add_gridspec(1, 2)

        # (a) 電位
        ax = fig.add_subplot(gs[0, 0])
        self.im_phi = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=extent,
                                cmap="RdBu_r", vmin=-1, vmax=1)
        T.draw_outline(ax, xc, yc, solid, mask, "k")
        fig.colorbar(self.im_phi, ax=ax, shrink=0.75)
        ax.set_title("Potential [V]", fontsize=12, pad=12)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

        # (b) 電場
        ax = fig.add_subplot(gs[0, 1])
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="0.85")
        self.im_E = ax.imshow(np.ma.masked_array(np.zeros((ny, nx)), mask=True), origin="lower", extent=extent,
                              cmap=cmap, vmin=0, vmax=1)
        T.draw_outline(ax, xc, yc, solid, mask, "k")
        s = 3
        X, Y = np.meshgrid(xc[::s], yc[::s], indexing="ij")
        self.qmask = (~solid)[::s, ::s]
        self.qs = s
        nq = int(self.qmask.sum())
        self.quiv = ax.quiver(X[self.qmask], Y[self.qmask], np.zeros(nq), np.zeros(nq),
                              color="w", scale=45, width=0.003)
        fig.colorbar(self.im_E, ax=ax, shrink=0.75)
        ax.set_title("Electric field [MV/m]", fontsize=12, pad=12)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

        # 左右の側壁を同じ色で全ビューに表示する。
        wf = self.wall_fig
        wg = wf.add_gridspec(2, 2)
        ax = wf.add_subplot(wg[0, 0])
        top_ox = p.floor_t + p.trench_d
        self.iy = np.arange(p.floor_t, top_ox + p.mask_t)
        self.yw = (self.iy + 0.5) * dxn
        self.wall_cells = T.wall_profiles(solid, p)
        (self.l_phi,) = ax.plot([], [], color=COLORS["blue"], label="left wall", lw=2)
        (self.r_phi,) = ax.plot([], [], "--", color=COLORS["amber"], label="right wall", lw=2)
        ax.legend(fontsize=9)
        ax.set_xlabel("Potential [V]")
        ax.set_ylabel("y [nm]")
        ax.set_ylim(0, ny * dxn)
        ax.grid(alpha=0.3)
        if p.mask_t > 0:
            T.draw_mask_level(ax, top_ox * dxn)
        ax.set_title("Sidewall potential", fontsize=12, pad=12)
        ax2 = wf.add_subplot(wg[0, 1], sharey=ax)
        (self.l_sig,) = ax2.plot([], [], color=COLORS["blue"], label="left wall", lw=2)
        (self.r_sig,) = ax2.plot([], [], "--", color=COLORS["amber"], label="right wall", lw=2)
        ax2.legend(fontsize=9)
        ax2.set_xlabel("Surface charge [mC/m$^2$]")
        ax2.set_ylabel("y [nm]")
        ax2.set_title("Sidewall surface charge", fontsize=12, pad=12)
        ax2.grid(alpha=0.2)
        if p.mask_t > 0:
            T.draw_mask_level(ax2, top_ox * dxn)
        self.ax_wall, self.ax_wall2 = ax, ax2

        self.ax_charge = wf.add_subplot(wg[1, :])
        (self.l_charge,) = self.ax_charge.plot([], [], color=COLORS["blue"], label="left SiO2", lw=2)
        (self.r_charge,) = self.ax_charge.plot([], [], "--", color=COLORS["amber"], label="right SiO2", lw=2)
        self.ax_charge.set(xlabel="time [ms]", ylabel="wall charge [pC/m]",
                           title="Accumulated SiO2 wall charge")
        self.ax_charge.legend(fontsize=9); self.ax_charge.grid(alpha=0.2)

        # (d) 電位の時間変化
        hf = self.history_fig
        hg = hf.add_gridspec(2, 2)
        self.ax_h = hf.add_subplot(hg[0, :])
        self.ax_h.set_title("RF cycle mean surface potential" if p.field_mode == "circuit_rf" else
                            "Surface potential history", fontsize=12, pad=12)
        self.ax_h.set_xlabel("time [ms]"); self.ax_h.set_ylabel("potential [V]")
        self.ax_h.grid(alpha=0.3)

        # (e) 底でのフラックス比
        self.ax_f = hf.add_subplot(hg[1, 0])
        (self.l_fi,) = self.ax_f.plot([], [], label="ion (bottom)")
        (self.l_fe,) = self.ax_f.plot([], [], label="electron (bottom)")
        self.ax_f.set_xlabel("time [ms]"); self.ax_f.set_ylabel("relative flux")
        self.ax_f.set_title("Bottom flux", fontsize=12, pad=12)
        self.ax_f.legend(fontsize=8); self.ax_f.grid(alpha=0.3)

        # (f) 表面リーク電流
        self.ax_l = hf.add_subplot(hg[1, 1])
        self.ax_l.set_xlabel("time [ms]")
        self.ax_l.grid(alpha=0.3)
        if p.sigma_s > 0:
            (self.l_leak,) = self.ax_l.plot([], [])
            self.ax_l.set_ylabel("relative current")
        else:
            self.l_leak = None
            self.ax_l.text(.5, .5, "Surface leakage disabled\n(sigma_s = 0)", ha="center", va="center",
                           transform=self.ax_l.transAxes, color=COLORS["muted"])
        self.ax_l.set_title("Leakage", fontsize=12, pad=12)
        for f in (fig, wf, hf):
            style_figure(f)
        self.ready = True

    def fit_layout(self, key, height, width=None):
        """低い画面では横並びにし、データを保持したままラベルの重なりを防ぐ。"""
        if key == "fields" and not self.ready and hasattr(self, "preview_title"):
            compact = height < 400
            narrow = (width if width is not None else self.fig.get_size_inches()[0]*self.fig.dpi) < 900
            self.preview_title.set_fontsize(13 if compact else 18)
            self.preview_title.set_position((.05, .98 if compact else .90))
            self.preview_details.set_fontsize(9 if compact or narrow else 10)
            self.preview_details.set_linespacing(1.25 if compact else 1.8)
            self.preview_details.set_position((.05, .74 if compact else .76))
            self.preview_details.set_text(self.preview_detail_text.replace(" / Mask ", "\nMask taper from vertical: ")
                                          if compact or narrow else self.preview_detail_text)
            self.preview_footer.set_visible(not (compact or narrow))
            return
        if not self.ready or key not in ("walls", "history"):
            return
        compact = height < 360
        attr = "compact_" + key
        if getattr(self, attr) == compact:
            return
        setattr(self, attr, compact)
        if key == "history":
            g = self.history_fig.add_gridspec(1, 3, width_ratios=[2, 1, 1]) if compact else self.history_fig.add_gridspec(2, 2)
            positions = (g[0, 0], g[0, 1], g[0, 2]) if compact else (g[0, :], g[1, 0], g[1, 1])
            for ax, pos in zip((self.ax_h, self.ax_f, self.ax_l), positions):
                ax.set_subplotspec(pos)
            if self.lines:
                self.ax_h.legend(fontsize=6 if compact else 8, loc="upper left", ncol=2 if compact else 3)
        else:
            g = self.wall_fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5]) if compact else self.wall_fig.add_gridspec(2, 2)
            positions = (g[0, 0], g[0, 1], g[0, 2]) if compact else (g[0, 0], g[0, 1], g[1, :])
            for ax, pos in zip((self.ax_wall, self.ax_wall2, self.ax_charge), positions):
                ax.set_subplotspec(pos)
                ax.title.set_fontsize(9 if compact else 12)

    def preview(self, p):
        """実行前にはゼロ電場を結果として描かず、入力形状を表示する。"""
        solid, mask, i0, _ = T.build_geometry(p)
        self.ready = False
        self.fig.clear()
        if hasattr(self.fig, "set_layout_engine"):
            self.fig.set_layout_engine("constrained")
        ax, note = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
        from matplotlib.colors import ListedColormap
        materials = solid.astype(int) + mask.astype(int)
        dxn = p.dx * 1e9
        nx, ny = solid.shape
        ax.imshow(materials.T, origin="lower", extent=[0, nx*dxn, 0, ny*dxn],
                  cmap=ListedColormap(["#eaf2ff", "#cbd5e1", "#334155"]), vmin=0, vmax=2)
        ax.axvline((i0+p.trench_w/2)*dxn, color=COLORS["blue"], ls="--", lw=1)
        ax.set(xlabel="x [nm]", ylabel="y [nm]", title="Geometry preview")
        note.axis("off")
        widths = T.opening_widths(p)
        mask_angle = f"{p.mask_taper_deg:g}°" if p.mask_t > 0 else "none"
        b0, b1 = T.opening_span(solid, p.floor_t)
        e0, e1 = T.opening_span(solid, p.floor_t+p.trench_d+p.mask_t-1)
        self.preview_title = note.text(.05, .90, "Ready to simulate", fontsize=18, color=COLORS["text"], va="top")
        self.preview_details = note.text(.05, .76, f"SiO2 top opening  {p.trench_w*dxn:g} nm\n"
                  f"Bottom  {widths['bottom']*dxn:.1f} nm  (grid {(b1-b0)*dxn:g})\n"
                  f"Entrance  {widths['top']*dxn:.1f} nm  (grid {(e1-e0)*dxn:g})\n"
                  f"Depth  {p.trench_d*dxn:g} nm / AR  {p.trench_d/p.trench_w:.2f}\n"
                  f"Taper from vertical: SiO2 {p.trench_taper_deg:g}° / Mask {mask_angle}\n"
                  f"Center offset  {p.trench_offset:+d} cells",
                  fontsize=10, linespacing=1.8, color=COLORS["muted"], va="top")
        self.preview_detail_text = self.preview_details.get_text()
        self.preview_footer = note.text(.05, .12, "Vacuum / plasma     SiO2     Mask\n"
                  "Positive taper: narrower toward the bottom.\n"
                  "Geometry uses the simulation cell grid.", fontsize=9,
                  linespacing=1.6, color=COLORS["muted"], va="bottom")
        style_figure(self.fig)
        self.fit_layout("fields", self.fig.get_size_inches()[1]*self.fig.dpi)

    def update(self, d):
        """d: phi, rho, Ex, Ey, hist, probes, t_conv を持つ dict (途中経過・最終結果どちらも可)"""
        if not self.ready:
            return
        p, solid, dxn = self.p, self.solid, self.dxn
        phi, rho, Ex, Ey, hist = d["phi"], d["rho"], d["Ex"], d["Ey"], d["hist"]
        vac = ~solid

        vmax = max(np.abs(phi).max(), 1e-9)
        self.im_phi.set_data(phi.T)
        self.im_phi.set_clim(-vmax, vmax)

        Emag = np.hypot(Ex, Ey) / 1e6
        self.im_E.set_data(np.ma.masked_where(solid, Emag).T)
        self.im_E.set_clim(0, max(Emag[vac].max(), 1e-9))
        s = self.qs
        ex, ey = Ex[::s, ::s][self.qmask], Ey[::s, ::s][self.qmask]
        mag = np.hypot(ex, ey) + 1e-30
        self.quiv.set_UVC(ex / mag, ey / mag)

        lx, ly = self.wall_cells["left"]
        rx, ry = self.wall_cells["right"]
        self.l_phi.set_data(phi[lx, ly], (ly+0.5)*dxn)
        self.l_sig.set_data(rho[lx, ly] * p.dx * 1e3, (ly+0.5)*dxn)
        self.r_phi.set_data(phi[rx, ry], (ry+0.5)*dxn)
        self.r_sig.set_data(rho[rx, ry] * p.dx * 1e3, (ry+0.5)*dxn)
        for a in (self.ax_wall, self.ax_wall2):
            a.relim()
            a.autoscale_view(scaley=False)

        t = hist["t"] * 1e3
        if not self.lines:
            for k in d["probes"]:
                (self.lines[k],) = self.ax_h.plot([], [], label=k)
            self.ax_h.legend(fontsize=8, loc="upper left", ncol=3)
        for k, line in self.lines.items():
            line.set_data(t, hist[k])
        n = len(t)
        if n == 0:
            return
        k = min(10, max(1, n // 5))
        ma = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
        self.l_fi.set_data(t[k - 1:], ma(hist["ratio_i_bottom"]))
        self.l_fe.set_data(t[k - 1:], ma(hist["ratio_e_bottom"]))
        if self.l_leak is not None:
            self.l_leak.set_data(t[k - 1:], ma(hist["leak"]))
        self.l_charge.set_data(t, d["charge_hist"]["left"] * 1e12)
        self.r_charge.set_data(t, d["charge_hist"]["right"] * 1e12)
        for a in (self.ax_h, self.ax_f, self.ax_l, self.ax_charge):
            a.relim()
            a.autoscale_view()

        for v in self.vlines:
            v.remove()
        self.vlines = []
        tc = d.get("t_conv", np.nan)
        if tc is not None and np.isfinite(tc):
            for a in (self.ax_h, self.ax_f, self.ax_l, self.ax_charge):
                self.vlines.append(a.axvline(tc * 1e3, color="k", ls=":", lw=1))


# ---------------------------------------------------------------- アプリ本体
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("トレンチ表面帯電シミュレーター (2D)")
        self.geometry("1500x920")
        self.minsize(1150, 720)

        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.result = None
        self.p_run = None
        self.running = False
        self.pending = None
        self.last_draw = 0.0
        self.last_check = None
        self.preview_after = None
        self.circuit_busy = False
        self.views = {}

        self.mode = tk.StringVar(value="fixed")
        self.info = tk.StringVar()
        self.backend_label = tk.StringVar(value="粒子追跡  CPU")
        self.state_label = tk.StringVar(value="待機中")
        self.result_label = tk.StringVar(value="入力形状のプレビュー · 計算結果はまだありません")
        self.metrics = {key: tk.StringVar(value="—") for key in ("time", "bottom", "left", "right", "flux")}
        self.log_visible = tk.BooleanVar(value=False)
        self.vars, self.specs, self.entries, self.radios = {}, {}, {}, []

        self._build_style()
        self._build_shell()
        self._build_left()
        self._build_right()
        self.set_params(T.Params())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll)

    def _build_style(self):
        self.configure(background=COLORS["bg"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Yu Gothic UI", 10), foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("TLabel", background=COLORS["bg"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"])
        style.configure("Title.TLabel", font=("Yu Gothic UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Yu Gothic UI", 11, "bold"))
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("Card.TLabel", background=COLORS["card"])
        style.configure("Metric.TLabel", background=COLORS["card"], font=("Segoe UI", 21, "bold"))
        style.configure("MetricName.TLabel", background=COLORS["card"], foreground=COLORS["muted"])
        style.configure("TLabelframe", background=COLORS["card"], bordercolor=COLORS["border"], padding=8)
        style.configure("TLabelframe.Label", background=COLORS["card"], font=("Yu Gothic UI", 10, "bold"))
        style.configure("TEntry", padding=5, fieldbackground=COLORS["card"], bordercolor=COLORS["border"])
        style.configure("TCombobox", padding=5, fieldbackground=COLORS["card"], bordercolor=COLORS["border"])
        style.map("TEntry", fieldbackground=[("disabled", "#f1f5f9")], foreground=[("disabled", "#94a3b8")])
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["card"]), ("disabled", "#f1f5f9")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("TButton", background=COLORS["card"], bordercolor=COLORS["border"], padding=(10, 7))
        style.map("TButton", background=[("active", "#e8eef8")])
        style.configure("Accent.TButton", background=COLORS["blue"], foreground="white", borderwidth=0,
                        font=("Yu Gothic UI", 10, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#b7c8e6"), ("active", "#1d4ed8")],
                  foreground=[("disabled", "#f1f5f9")])
        style.configure("Stop.TButton", foreground="#b91c1c")
        style.configure("TRadiobutton", background=COLORS["card"])
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 8), background="#e9eef6")
        style.map("TNotebook.Tab", background=[("selected", COLORS["card"])],
                  foreground=[("selected", COLORS["blue"])])
        style.configure("Horizontal.TProgressbar", troughcolor="#e1e8f2", background=COLORS["blue"],
                        borderwidth=0, thickness=5)
        style.configure("Badge.TLabel", background="#dbeafe", foreground="#1d4ed8", padding=(10, 4))

    def _build_shell(self):
        header = ttk.Frame(self, padding=(20, 12, 20, 10))
        header.pack(fill="x")
        ttk.Label(header, text="トレンチ表面帯電", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="2D プラズマ・粒子軌道シミュレーター", style="Muted.TLabel").pack(side="left", padx=18)
        ttk.Label(header, textvariable=self.backend_label, style="Badge.TLabel").pack(side="right")
        self.body = ttk.Panedwindow(self, orient="horizontal")
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 12))

    # ---------------- 左: パラメータ入力
    def _build_left(self):
        left = ttk.Frame(self.body, width=382, padding=(0, 0, 10, 0))
        self.body.add(left, weight=0)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text="計算条件", style="Section.TLabel").pack(side="left")
        self.btn_cfg_save = ttk.Button(row, text="設定保存", command=self.on_cfg_save)
        self.btn_cfg_load = ttk.Button(row, text="読込", command=self.on_cfg_load)
        self.btn_cfg_load.pack(side="right")
        self.btn_cfg_save.pack(side="right", padx=4)
        self.config_tabs = ttk.Notebook(left)
        self.config_tabs.pack(fill="both", expand=True)
        pages = {name: self._scroll_page(self.config_tabs, name) for name in ("形状", "電場", "プラズマ", "計算")}

        self._group(pages["形状"], "トレンチ形状", GEOMETRY)
        self._group(pages["形状"], "マスク", MASK)
        self._group(pages["電場"], "境界電位・位相固定 / RF 全周期", FIELDS)
        self._group(pages["電場"], "表面リーク", LEAK)
        ttk.Label(pages["電場"], text="circuit は指定位相の電位を固定します。\ncircuit_rf は全周期を使い、飛行中も電位を更新します。位相指定は場の表示にだけ使います。\npillar は基板・マスク間の端面電位を線形補間します。",
                  wraplength=335, style="Muted.TLabel").pack(anchor="w", padx=10, pady=10)
        self._group(pages["プラズマ"], "入射粒子", PLASMA)
        self._group(pages["プラズマ"], "RF バイアス・等価回路", RF)

        g = ttk.LabelFrame(pages["計算"], text="実行モード")
        g.pack(fill="x", padx=6, pady=6)
        r = ttk.Radiobutton(g, text="固定長（バッチ数を指定）", variable=self.mode, value="fixed",
                            command=self.apply_states)
        r.grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        self.radios.append(r)
        row = 1
        for spec in FIXED_FIELDS:
            self._field(g, row, spec); row += 1
        r = ttk.Radiobutton(g, text="飽和まで継続", variable=self.mode, value="steady", command=self.apply_states)
        r.grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(12, 4))
        self.radios.append(r)
        row += 1
        for spec in STEADY_FIELDS:
            self._field(g, row, spec); row += 1
        ttk.Label(g, text="電位と電荷の両方が安定したら停止します。\n電荷が増え続けている間は計算を継続します。",
                  wraplength=330, style="Muted.TLabel").grid(row=row, column=0, columnspan=3,
                                                           sticky="w", padx=6, pady=(6, 8))
        ttk.Label(pages["計算"], text="再開は「結果を読込」→「続きから計算」。\n固定長は指定バッチ数を追加します。飽和まで継続する場合は通算の上限を増やしてください。形状・物理条件は保存結果を引き継ぎます。",
                  wraplength=335, style="Muted.TLabel").pack(anchor="w", padx=10, pady=8)
        self._group(pages["計算"], "粒子追跡・数値設定", NUMERICS)
        self._group(pages["計算"], "飽和後の垂直イオン診断", PROBE)

        g = ttk.LabelFrame(left, text="入力条件の確認")
        g.pack(fill="x", pady=(10, 6))
        ttk.Label(g, textvariable=self.info, justify="left", wraplength=335,
                  style="Card.TLabel").pack(anchor="w", padx=2, pady=2)
        self.btn_reset = ttk.Button(left, text="設定を既定値に戻す", command=lambda: self.set_params(T.Params()))
        self.btn_reset.pack(fill="x")

    def _scroll_page(self, notebook, title):
        page = ttk.Frame(notebook)
        notebook.add(page, text=title)
        cv = tk.Canvas(page, width=360, background=COLORS["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(page, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        window = cv.create_window((0, 0), window=inner, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(window, width=e.width))
        cv.configure(yscrollcommand=vsb.set)
        cv.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        # Entry 上でもホイールを受け取る。アプリのウィジェット階層に限定する。
        def wheel(e):
            child = e.widget
            while child is not None:
                if child == page:
                    cv.yview_scroll(-1 if (getattr(e, "num", 0) == 4 or e.delta > 0) else 1, "units")
                    return "break"
                child = getattr(child, "master", None)
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind(ev, wheel, add="+")
        return inner

    def _group(self, parent, title, specs):
        g = ttk.LabelFrame(parent, text=title)
        g.pack(fill="x", padx=6, pady=6)
        for i, spec in enumerate(specs):
            self._field(g, i, spec)

    def _field(self, parent, row, spec):
        key, label, unit, scale, typ = spec
        self.specs[key] = (label, unit, scale, typ)
        var = tk.StringVar()
        self.vars[key] = var
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text=label, wraplength=172, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(2, 6), pady=5)
        if isinstance(typ, tuple):
            w = ttk.Combobox(parent, textvariable=var, values=typ, width=10, state="readonly")
            w.bind("<<ComboboxSelected>>", lambda e: self.apply_states())
        else:
            w = ttk.Entry(parent, textvariable=var, width=11, justify="right")
            w.bind("<KeyRelease>", self.update_info)
        w.grid(row=row, column=1, padx=2, pady=1)
        ttk.Label(parent, text=unit, width=6, style="Card.TLabel").grid(row=row, column=2, sticky="w", padx=(4, 0))
        self.entries[key] = w

    # ---------------- 右: ボタン・グラフ・ログ
    def _build_right(self):
        right = ttk.Frame(self.body, padding=(10, 0, 0, 0))
        self.body.add(right, weight=1)

        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(0, 10))
        self.btn_run = ttk.Button(bar, text="▶ 計算を実行", command=self.on_run, style="Accent.TButton")
        self.btn_resume = ttk.Button(bar, text="続きから計算", command=self.on_resume, state="disabled")
        self.btn_stop = ttk.Button(bar, text="■ 停止", command=self.on_stop, style="Stop.TButton", state="disabled")
        self.btn_save = ttk.Button(bar, text="結果を保存", command=self.on_save, state="disabled")
        self.btn_load_result = ttk.Button(bar, text="結果を読込", command=self.on_load_result)
        for b in (self.btn_run, self.btn_resume, self.btn_stop, self.btn_save, self.btn_load_result):
            b.pack(side="left", padx=(0, 6))
        ttk.Label(bar, textvariable=self.state_label, style="Badge.TLabel").pack(side="right")

        cards = ttk.Frame(right)
        cards.pack(fill="x", pady=(0, 12))
        specs = [("time", "帯電時間", "ms", COLORS["text"]),
                 ("bottom", "底面中心電位", "V", COLORS["text"]),
                 ("left", "左側壁の電荷", "pC/m", COLORS["blue"]),
                 ("right", "右側壁の電荷", "pC/m", COLORS["amber"]),
                 ("flux", "底面イオン流束", "入射流束比 · 直近10バッチ", COLORS["green"])]
        for i, (key, label, unit, color) in enumerate(specs):
            cards.columnconfigure(i, weight=1, uniform="metrics")
            card = ttk.Frame(cards, style="Card.TFrame", padding=(12, 10))
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 6 if i < 4 else 0))
            ttk.Label(card, text=label, style="MetricName.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.metrics[key], style="Metric.TLabel", foreground=color).pack(anchor="w", pady=3)
            unit_label = ttk.Label(card, text=unit, style="MetricName.TLabel", font=("Yu Gothic UI", 8), wraplength=170)
            unit_label.pack(anchor="w")
            card.bind("<Configure>", lambda e, label=unit_label: label.configure(wraplength=max(75, e.width-24)))

        self.status1 = tk.StringVar(value="条件を設定して「計算を実行」を押してください。")
        self.status2 = tk.StringVar(value="飽和に達した結果から、垂直イオンの軌道と底面フラックスを比較できます。")
        self.pbar = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.pbar.configure(style="Horizontal.TProgressbar")
        self.pbar.pack(fill="x", pady=(0, 6))
        ttk.Label(right, textvariable=self.status1, font=("Yu Gothic UI", 10, "bold")).pack(anchor="w")
        self.detail_label = ttk.Label(right, textvariable=self.status2, style="Muted.TLabel", wraplength=950)
        self.detail_label.pack(anchor="w", pady=(2, 10))
        right.bind("<Configure>", self._fit_labels)

        tools = ttk.Frame(right)
        tools.pack(fill="x", pady=(0, 8))
        self.btn_probe = ttk.Button(tools, text="垂直イオンを追跡", command=self.on_probe, state="disabled")
        self.btn_circuit = ttk.Button(tools, text="回路・IED を確認", command=self.on_circuit)
        self.btn_probe.pack(side="left", padx=(0, 6))
        self.btn_circuit.pack(side="left")
        self.btn_log = ttk.Button(tools, text="計算ログ ▸", command=self.toggle_log)
        self.btn_log.pack(side="right")
        self.btn_expand = ttk.Button(tools, text="図を拡大", command=self.expand_plot)
        self.btn_expand.pack(side="right", padx=6)

        self.log_frame = ttk.Frame(right)
        self.log_text = tk.Text(self.log_frame, height=5, state="disabled", wrap="none", font=("Consolas", 9),
                                background="#152238", foreground="#dbe6f8", relief="flat", padx=10, pady=8)
        lsb = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=lsb.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        lsb.pack(side="right", fill="y")

        self.context_label = ttk.Label(right, textvariable=self.result_label, style="Muted.TLabel",
                                       font=("Yu Gothic UI", 8), wraplength=1000)
        self.context_label.pack(side="bottom", fill="x", pady=(6, 0))
        self.result_tabs = ttk.Notebook(right)
        self.result_tabs.pack(fill="both", expand=True)
        for key, title in (("fields", "電位・電場"), ("walls", "左右側壁"), ("history", "時間履歴"),
                           ("ions", "イオン軌道"), ("phase", "RF位相"), ("circuit", "回路・IED")):
            page = ttk.Frame(self.result_tabs, style="Card.TFrame")
            self.result_tabs.add(page, text=title)
            caption = tk.StringVar()
            caption_label = ttk.Label(page, textvariable=caption, style="Card.TLabel", wraplength=950)
            if key in ("ions", "phase", "circuit"):
                caption_label.pack(fill="x", padx=12, pady=(8, 0))
                page.bind("<Configure>", lambda e, label=caption_label: label.configure(wraplength=max(350, e.width-24)))
            self.views[key] = dict(page=page, fig=None, canvas=None, toolbar=None, caption=caption)
            self._install_figure(key, Figure(figsize=(8, 5), dpi=100, constrained_layout=True))
        self.fig = self.views["fields"]["fig"]
        self.canvas = self.views["fields"]["canvas"]
        self.plot = LivePlot(self.fig, self.views["walls"]["fig"], self.views["history"]["fig"])
        self._empty_view("walls", "Sidewall profiles and accumulated charge",
                         "Run or load a charging result to compare the left and right walls.")
        self._empty_view("history", "Charging history", "Surface potential, particle flux and leakage appear here.")
        self._empty_view("ions", "Vertical ion trajectories",
                         "Calculate to saturation, then use the vertical ion tracing button.\n"
                         "Charged and uncharged trajectories are compared using the same beam.")
        self._empty_view("circuit", "RF circuit and ion distributions",
                         "Choose RF bias in the plasma settings, then calculate the circuit / IED.")
        self._empty_view("phase", "RF phase dependence",
                         "Use circuit_rf and trace vertical ions after saturation.\n"
                         "Bottom arrival fraction and impact energy are resolved by launch phase.")
        self.result_tabs.bind("<<NotebookTabChanged>>", self._draw_active)

    def _fit_labels(self, event):
        width = max(350, event.width-20)
        self.detail_label.configure(wraplength=width)
        self.context_label.configure(wraplength=width)

    def _install_figure(self, key, fig):
        view = self.views[key]
        if view["canvas"] is not None:
            pending = getattr(view["canvas"], "_idle_draw_id", None)
            if pending is not None:
                view["canvas"].get_tk_widget().after_cancel(pending)
            view["toolbar"].destroy()
            view["canvas"].get_tk_widget().destroy()
        style_figure(fig)
        canvas = FigureCanvasTkAgg(fig, master=view["page"])
        toolbar = NavigationToolbar2Tk(canvas, view["page"], pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.get_tk_widget().bind("<Configure>", self._fit_figure, add="+")
        view.update(fig=fig, canvas=canvas, toolbar=toolbar)

    def _empty_view(self, key, title, hint):
        fig = self.views[key]["fig"]
        fig.clear()
        self.views[key]["caption"].set("")
        ax = fig.add_subplot()
        ax.axis("off")
        ax.text(.5, .58, title, ha="center", fontsize=17, color=COLORS["text"], transform=ax.transAxes)
        ax.text(.5, .43, hint, ha="center", fontsize=10, linespacing=1.8,
                color=COLORS["muted"], transform=ax.transAxes)

    def _draw_active(self, *_):
        selected = self.result_tabs.select()
        for view in self.views.values():
            if str(view["page"]) == selected:
                view["canvas"].draw_idle()

    def toggle_log(self):
        visible = not self.log_visible.get()
        self.log_visible.set(visible)
        if visible:
            self.log_frame.pack(side="bottom", fill="x", pady=(8, 0))
        else:
            self.log_frame.pack_forget()
        self.btn_log.configure(text="計算ログ ▾" if visible else "計算ログ ▸")

    def expand_plot(self):
        """選択中の図を独立した大きいウィンドウで確認する。元の図とは共有しない。"""
        import pickle
        selected = self.result_tabs.select()
        view = next(v for v in self.views.values() if str(v["page"]) == selected)
        fig = pickle.loads(pickle.dumps(view["fig"]))
        win = tk.Toplevel(self)
        win.title("グラフの拡大表示（開いた時点の図）")
        win.geometry("1200x850")
        canvas = FigureCanvasTkAgg(fig, master=win)
        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

    def _fit_figure(self, event=None):
        """図のピクセルサイズをウィジェットに合わせる (ずれている時だけ)。"""
        for key, view in self.views.items():
            fig, canvas = view["fig"], view["canvas"]
            w = canvas.get_tk_widget()
            pw, ph = w.winfo_width(), w.winfo_height()
            if not w.winfo_ismapped() or pw < 100 or ph < 100:
                continue
            if hasattr(self, "plot"):
                self.plot.fit_layout(key, ph, pw)
            dpi = fig.dpi
            cur = fig.get_size_inches() * dpi
            if abs(cur[0] - pw) > 2 or abs(cur[1] - ph) > 2:
                fig.set_size_inches(pw / dpi, ph / dpi, forward=False)
                canvas.draw_idle()

    # ---------------- パラメータの入出力
    def set_params(self, p):
        for key, (label, unit, scale, typ) in self.specs.items():
            v = getattr(p, key)
            self.vars[key].set(v if isinstance(typ, tuple) else fmt(v / scale if scale != 1 else v))
        self.mode.set("steady" if p.until_steady else "fixed")
        self.apply_states()

    def get_params(self):
        kw = {}
        for key, (label, unit, scale, typ) in self.specs.items():
            txt = str(self.vars[key].get()).strip()
            try:
                if isinstance(typ, tuple):
                    if txt not in typ:
                        raise ValueError
                    kw[key] = txt
                elif typ is int:
                    f = float(txt)
                    if f != int(f):
                        raise ValueError
                    kw[key] = int(f)
                else:
                    kw[key] = float(txt) * scale
            except (ValueError, OverflowError):
                raise ValueError(f"「{label}」の値が正しくありません: '{txt}'")
        kw["until_steady"] = self.mode.get() == "steady"
        return T.Params(**kw)

    def apply_states(self):
        steady = self.mode.get() == "steady"
        for key, w in self.entries.items():
            if self.running:
                st = "disabled"
            elif key in FIXED_KEYS:
                st = "disabled" if steady else "normal"
            elif key in STEADY_KEYS:
                st = "normal" if steady else "disabled"
            elif key == "mask_eps_r":
                st = "normal" if self.vars["mask_type"].get() == "dielectric" else "disabled"
            elif key == "cuda_device":
                st = "disabled" if self.vars["backend"].get() == "cpu" else "normal"
            elif key == "rf_phase":
                st = "normal" if self.vars["field_mode"].get() in T.CIRCUIT_FIELDS else "disabled"
            elif key == "rf_time_steps":
                st = "normal" if self.vars["field_mode"].get() == "circuit_rf" else "disabled"
            elif key in ("substrate_voltage", "top_voltage"):
                st = "disabled" if self.vars["field_mode"].get() in T.CIRCUIT_FIELDS else "normal"
            elif key == "mask_voltage":
                st = "normal" if self.vars["field_mode"].get() == "fixed" else "disabled"
            elif key in ("left_voltage", "right_voltage"):
                st = "normal" if self.vars["side_bc"].get() == "fixed" else "disabled"
            elif key in SHEATH_KEYS:
                st = "normal" if (self.vars["bias"].get() == "rf"
                                  and self.vars["ied_model"].get() == "sheath") else "disabled"
            elif key in PULSE_KEYS:
                st = "normal" if (self.vars["bias"].get() == "rf"
                                  and self.vars["rf_wave"].get() == "pulse") else "disabled"
            elif key in RF_KEYS:
                st = "normal" if self.vars["bias"].get() == "rf" else "disabled"
            elif key == "ion_energy_eV":
                st = "normal" if self.vars["bias"].get() == "dc" else "disabled"
            else:
                st = "normal"
            if isinstance(w, ttk.Combobox) and st == "normal":
                st = "readonly"
            w.configure(state=st)
        for r in self.radios:
            r.configure(state="disabled" if self.running else "normal")
        self.update_info()

    def update_info(self, *_):
        try:
            p = self.get_params()
        except (ValueError, OverflowError) as e:
            self.info.set(str(e))
            return
        err = validate(p)
        if err:
            self.info.set("入力を確認: " + err)
            return
        ny = p.floor_t + p.trench_d + p.mask_t + p.n_vac
        lines = [f"格子 {p.nx} × {ny} セル (領域幅 {p.nx * p.dx * 1e9:.0f} nm)",
                 f"トレンチ 幅 {p.trench_w * p.dx * 1e9:.0f} nm × 深さ {p.trench_d * p.dx * 1e9:.0f} nm "
                 f"(AR = {p.trench_d / p.trench_w:.1f})"]
        center = ((p.nx-p.trench_w)//2 + p.trench_offset + p.trench_w/2) * p.dx * 1e9
        lines.append(f"中心 {center:g} nm / 電場 {p.field_mode} / 両端 {p.side_bc}")
        widths = T.opening_widths(p)
        lines.append(f"SiO2 テーパー {p.trench_taper_deg:g}° / 底幅 {widths['bottom']*p.dx*1e9:.1f} nm")
        if p.mask_t > 0:
            kind = "導体" if p.mask_type == "conductor" else f"誘電体 εr={p.mask_eps_r:g}"
            lines.append(f"マスク {p.mask_t * p.dx * 1e9:.0f} nm ({kind}) / テーパー {p.mask_taper_deg:g}°")
            lines.append(f"入口幅 {widths['top']*p.dx*1e9:.1f} nm (設計値)")
        else:
            lines.append("マスクなし")
        if p.until_steady:
            lines.append(f"飽和まで / 上限 {p.max_batches * p.dt_batch * 1e3:g} ms")
        else:
            lines.append(f"固定長 {p.n_batches * p.dt_batch * 1e3:g} ms / {p.n_batches} バッチ")
        self.info.set("\n".join(lines))
        if self.result is None and not self.running:
            self.backend_label.set("粒子追跡  " + {"cpu": "CPU", "cuda": "CUDA を使用", "auto": "CPU / CUDA 自動選択"}[p.backend])
            if self.preview_after is not None:
                self.after_cancel(self.preview_after)
            self.preview_after = self.after(250, self._preview_geometry)

    def _preview_geometry(self):
        self.preview_after = None
        if self.result is not None or self.running:
            return
        try:
            p = self.get_params()
            if validate(p):
                return
            self.plot.preview(p)
            self._draw_active()
        except (ValueError, OverflowError):
            pass

    def update_metrics(self, data):
        h = data["hist"]
        if not len(h["t"]):
            return
        self.metrics["time"].set(f"{h['t'][-1]*1e3:.3g}")
        self.metrics["bottom"].set(f"{h['bottom center'][-1]:+.2f}")
        for side in ("left", "right"):
            value = f"{data['charge_hist'][side][-1]*1e12:+.3g}"
            self.metrics[side].set(value.replace("e+0", "e").replace("e-0", "e-"))
        self.metrics["flux"].set(f"{np.mean(h['ratio_i_bottom'][-10:]):.3f} ×")

    def _result_context(self):
        p = self.p_run
        rf_note = f"場は {p.rf_phase%360:g}° / 履歴・飽和は周期平均 / " if p.field_mode == "circuit_rf" else ""
        self.result_label.set(f"表示結果: {p.nx} × {p.floor_t+p.trench_d+p.mask_t+p.n_vac} セル / "
                              f"中心移動 {p.trench_offset:+d} / {p.field_mode} / "
                              f"{rf_note}{p.bias.upper()}（入力変更後もこの条件の結果を表示）")

    # ---------------- 実行・停止
    def on_run(self):
        if self.running or self.circuit_busy:
            return
        try:
            p = self.get_params()
            err = validate(p)
            if err:
                raise ValueError(err)
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            return
        self._start_calculation(p)

    def on_resume(self):
        if self.running or self.circuit_busy or self.result is None:
            return
        try:
            p = T.resume_params(self.result, self.get_params())
            err = validate(p)
            if err:
                raise ValueError(err)
            T.validate_resume(self.result, p)
        except ValueError as e:
            messagebox.showerror("再開の入力エラー", str(e))
            return
        self.set_params(p)
        self._start_calculation(p, resume=self.result)

    def _start_calculation(self, p, resume=None):
        self.p_run, self.result = p, resume
        self.stop_event = threading.Event()
        self.pending, self.last_check = None, None
        solid, mask, i0, _ = T.build_geometry(p)
        self.plot.setup(p, solid, mask, i0)
        for key in ("fields", "walls", "history"):
            self.views[key]["toolbar"].update()
        for key in ("ions", "phase", "circuit"):
            self._empty_view(key, "Waiting for this run", "Diagnostic results appear after calculation.")
        for metric in self.metrics.values():
            metric.set("—")
        if resume is not None:
            self.plot.update(resume)
            self.update_metrics(resume)
        self.result_tabs.select(self.views["fields"]["page"])
        self._draw_active()
        self._result_context()
        self.backend_label.set("粒子追跡  " + {"cpu": "CPU", "cuda": "CUDA を使用", "auto": "CPU / CUDA 自動選択"}[p.backend])

        self._log("---- 続きから計算 ----" if resume is not None else "---- 実行開始 ----")
        start = len(resume["hist"]["t"]) if resume is not None else 0
        target = p.max_batches if p.until_steady else start+p.n_batches
        self.pbar.configure(value=100*start/target)
        self.status1.set(f"t={start*p.dt_batch*1e3:g} ms、{start} バッチの続きから計算しています…"
                         if resume is not None else "計算を開始しています…")
        self.status2.set("")
        self.state_label.set("再開中" if resume is not None else "計算中")
        self.set_running(True)
        # Tk の画像・変数の循環参照は、JIT の初回ロードが GC を起動する前に
        # メインスレッドで回収する。バックグラウンドでの Tk 呼び出しを防ぐ。
        gc.collect()
        threading.Thread(target=self._work, args=(p, self.stop_event, resume), daemon=True).start()

    def _work(self, p, stop, resume=None):
        """ワーカースレッド。Tk には触らず、queue 経由でメインスレッドに渡す。"""
        try:
            res = T.run(p, log=lambda s: self.q.put(("log", s)),
                        progress=lambda info: self.q.put(("progress", info)), stop=stop, resume=resume)
            self.q.put(("done", res))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def on_stop(self):
        self.stop_event.set()
        self.btn_stop.configure(state="disabled")
        self.status1.set("停止しています (現在のバッチが終わるまで少しお待ちください)…")
        self.state_label.set("停止処理中")

    def set_running(self, running):
        self.running = running
        self.btn_run.configure(state="disabled" if running or self.circuit_busy else "normal")
        self.btn_resume.configure(state="normal" if self.result is not None and not running and not self.circuit_busy else "disabled")
        self.btn_stop.configure(state="normal" if running else "disabled")
        for b in (self.btn_save, self.btn_cfg_save, self.btn_cfg_load, self.btn_reset, self.btn_load_result):
            b.configure(state="disabled" if running else "normal")
        if not running and self.result is None:
            self.btn_save.configure(state="disabled")
        ready = self.result is not None and np.isfinite(self.result["t_conv"]) and not self.result["stopped"]
        self.btn_probe.configure(state="normal" if ready and not running else "disabled")
        self.btn_circuit.configure(state="disabled" if running or self.circuit_busy else "normal")
        self.apply_states()

    # ---------------- メインスレッド側の受信ループ
    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    self.pending = payload
                elif kind == "done":
                    self.pending = None
                    self.on_done(payload)
                elif kind == "error":
                    self.pending = None
                    self.set_running(False)
                    self.status1.set("エラーが発生しました")
                    self.state_label.set("エラー")
                    if not self.log_visible.get():
                        self.toggle_log()
                    self._log(payload)
                    messagebox.showerror("計算エラー", payload.splitlines()[-1])
                elif kind == "circuit":
                    self.show_circuit(payload)
                elif kind == "circuit_error":
                    self.circuit_busy = False
                    self.btn_run.configure(state="disabled" if self.running else "normal")
                    self.btn_resume.configure(state="normal" if self.result is not None and not self.running else "disabled")
                    self.btn_circuit.configure(state="disabled" if self.running else "normal")
                    self._log(payload)
                    messagebox.showerror("等価回路のエラー", payload.splitlines()[-1])
                elif kind == "probe_done":
                    probe, params = payload
                    self.set_running(False)
                    if probe["cancelled"]:
                        self.status1.set("垂直イオン診断を中断しました")
                        self.state_label.set("診断中断")
                    else:
                        self.result["ion_probe"] = probe
                        self.show_probe(probe, params)
                        self.status1.set("垂直イオン診断が完了しました (結果を保存で軌道も保存)")
                        self.status2.set("イオン軌道タブに結果を表示しました。入射エネルギー・粒子数を変えて再評価できます。")
                        self.state_label.set("診断完了")
        except queue.Empty:
            pass
        self._fit_figure()
        if self.pending is not None and time.time() - self.last_draw > 0.3:
            self.on_progress(self.pending)
            self.pending = None
        self.after(100, self.poll)

    def on_progress(self, info):
        b, n_max, el = info["batch"], info["n_max"], info["elapsed"]
        start = info.get("start_batch", 0)
        p = self.p_run
        self.pbar.configure(value=100.0 * b / n_max)
        t_ms = b * p.dt_batch * 1e3
        s1 = f"バッチ {b}/{n_max}  (t = {t_ms:.1f} ms)   経過 {fmt_time(el)}"
        if start:
            s1 = f"再開後 +{b-start} / 通算バッチ {b}/{n_max}  (t = {t_ms:.1f} ms)   経過 {fmt_time(el)}"
        if not p.until_steady and b < n_max:
            s1 += f"   残り約 {fmt_time(el / max(1, b-start) * (n_max - b))}"
        self.status1.set(s1)

        s2 = "左右の側壁電荷は「左右側壁」、各観測点の電位・粒子流束は「時間履歴」で確認できます。"
        if info["check"] is not None:
            self.last_check = info["check"]
        if p.until_steady and self.last_check is not None:
            c = self.last_check
            scale, unit = (1e12, "pC/m") if c.get("unit") == "C/m" else (1, "V")
            kind = "正味の充電" if c.get("kind") == "drift" else "変化"
            states = (f"電位 {'OK' if c['voltage']['ok'] else 'NG'} / 電荷 {'OK' if c['charge']['ok'] else 'NG'} · "
                      if "voltage" in c and "charge" in c else "")
            s2 = (f"飽和判定: {states}{c['name']} の{kind} {c['d']*scale:+.3g} {unit}/ブロック "
                  f"(許容 {c['tol']*scale:.3g} {unit}) · "
                  f"{'安定' if c['ok'] else '計算継続'} ({c['n_ok']}/{p.steady_hold})")
        elif p.until_steady and info.get("steady_samples_remaining", 0):
            s2 = f"飽和判定用の電位・電荷履歴を蓄積しています (あと {info['steady_samples_remaining']} バッチ)。"
        self.status2.set(s2)
        self.update_metrics(info)
        self.plot.update(info)
        self._draw_active()
        self.last_draw = time.time()

    def on_done(self, res):
        self.result = res
        self.set_running(False)
        if res is None:
            self.status1.set("中断しました (結果なし)")
            self.state_label.set("中断")
            self.result_label.set("結果なし · 入力形状のプレビュー")
            self.btn_save.configure(state="disabled")
            return
        self.plot.update(res)
        self.update_metrics(res)
        self._draw_active()
        self._result_context()
        self.backend_label.set("粒子追跡  " + ("CUDA" if res.get("backend_used") == "cuda" else "CPU"))
        self.last_draw = time.time()
        n = len(res["hist"]["t"])
        t_ms = res["hist"]["t"][-1] * 1e3
        if res["stopped"]:
            self.state_label.set("中断")
            msg = f"中断しました (t = {t_ms:.1f} ms, {n} バッチ)"
        elif np.isfinite(res["t_conv"]):
            self.state_label.set("飽和到達")
            msg = f"飽和に到達しました (t = {res['t_conv'] * 1e3:.1f} ms, {n} バッチ)"
        elif res.get("convergence_note"):
            self.state_label.set("未飽和")
            msg = res["convergence_note"]
        elif self.p_run.until_steady:
            self.state_label.set("上限到達")
            msg = f"上限 ({n} バッチ, {t_ms:.0f} ms) までに飽和しませんでした — 上限バッチ数を増やしてください"
        else:
            self.state_label.set("完了")
            msg = f"完了しました (t = {t_ms:.1f} ms, {n} バッチ)"
        self.status1.set(msg)
        self.pbar.configure(value=100 if not res["stopped"] else self.pbar["value"])
        self.status2.set("垂直イオン診断を実行できます。帯電あり / なしで軌道・底面流束・入射角を比較します。"
                         if np.isfinite(res["t_conv"]) and not res["stopped"] else
                         "結果は図と NPZ で保存できます。垂直イオン診断には「飽和まで継続」の結果が必要です。")
        self.btn_save.configure(state="normal")
        if res.get("circuit") is not None:
            self.show_circuit(res["circuit"], select=False)

    # ---------------- 飽和電荷を使う垂直ビーム診断
    def on_probe(self):
        if self.running or self.result is None:
            return
        try:
            entered = self.get_params()
            params = replace(self.p_run, probe_ions=entered.probe_ions,
                             backend=entered.backend, cuda_device=entered.cuda_device,
                             probe_energy_eV=entered.probe_energy_eV,
                             probe_trajectories=entered.probe_trajectories)
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            return
        self.stop_event = threading.Event()
        self.set_running(True)
        self.status1.set("飽和電荷を固定して垂直イオンを追跡しています…")
        self.state_label.set("イオン追跡中")
        gc.collect()
        threading.Thread(target=self._probe_work, args=(params, entered.rf_phase, self.stop_event), daemon=True).start()

    def _probe_work(self, params, phase, stop):
        try:
            probe = T.probe_vertical_ions(self.result, params, phase=phase,
                                         log=lambda s: self.q.put(("log", s)), stop=stop)
            self.q.put(("probe_done", (probe, params)))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def show_probe(self, probe, params, select=True):
        fig = T.plot_ion_probe(probe, self.result, params, compact=True)
        fig.suptitle("Full RF cycle: vertical ion trajectories and bottom impact" if probe.get("rf_cycle") else
                     "Vertical ion trajectories and bottom impact", fontsize=12, color=COLORS["text"])
        self._install_figure("ions", fig)
        self.views["ions"]["caption"].set(
            ("未飽和の保存電荷 · " if not probe.get("source_converged", True) else "") +
            (f"RF 全周期（背景電場は {probe['phase']%360:g}°） · " if probe.get("rf_cycle") else "") +
            f"入射 {probe['energy_eV']:g} eV / {probe['n_ions']:,} イオン · "
            f"保存時刻 {probe['source_t_conv']*1e3:.3g} ms の電荷を固定 · 底面到達率 "
            f"帯電あり {probe['charged']['bottom_fraction']:.1%} / なし {probe['uncharged']['bottom_fraction']:.1%}")
        if probe.get("rf_cycle"):
            self._install_figure("phase", T.plot_rf_probe_phase(probe, params))
            self.views["phase"]["caption"].set(
                "同じ入射位相・位置・速度で帯電あり / なしを比較。横軸は入射位相、飛行中にも電場は変化します。")
        else:
            self._empty_view("phase", "RF phase dependence", "This diagnostic uses a fixed field.")
        if select:
            self.result_tabs.select(self.views["ions"]["page"])
        self._draw_active()

    def on_load_result(self):
        path = filedialog.askopenfilename(title="帯電結果を読込", filetypes=[("NumPy result", "*.npz")])
        if not path:
            return
        try:
            res, params = T.load_results(path)
        except Exception as e:
            messagebox.showerror("読込エラー", str(e))
            return
        self.result, self.p_run = res, params
        self.set_params(params)
        self.plot.setup(params, res["solid"], res["mask"], res["i0"])
        for key in ("fields", "walls", "history"):
            self.views[key]["toolbar"].update()
        self._empty_view("ions", "Vertical ion trajectories", "No vertical ion diagnostic in this result.")
        self._empty_view("phase", "RF phase dependence", "No RF cycle diagnostic in this result.")
        self._empty_view("circuit", "RF circuit and ion distributions", "No RF circuit in this result.")
        self.on_done(res)
        if "ion_probe" in res:
            self.show_probe(res["ion_probe"], params, select=False)
        self.result_tabs.select(self.views["fields"]["page"])
        self._log(f"帯電結果を読込: {path}")

    # ---------------- 等価回路の確認
    def on_circuit(self):
        """今の入力値で RF バイアスの等価回路を解き、専用タブに表示する。
        1 次元シースの計算は数〜20 秒かかるので別スレッドで行い、終わったら poll() から show_circuit を呼ぶ。"""
        if self.running or self.circuit_busy:
            return
        try:
            p = self.get_params()
            err = validate(p)
            if err:
                raise ValueError(err)
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            return
        if p.bias != "rf":
            messagebox.showinfo("回路の波形", "バイアスが dc のときは等価回路を使いません (全イオンが同じエネルギー)。")
            return
        self.btn_circuit.configure(state="disabled")
        self.circuit_busy = True
        self.btn_run.configure(state="disabled")
        self.btn_resume.configure(state="disabled")
        self.status1.set("RF 回路・1 次元シースの入射分布を計算しています…")
        self.state_label.set("回路計算中")
        self.result_tabs.select(self.views["circuit"]["page"])
        self._empty_view("circuit", "Calculating RF circuit / IED", "The interface remains available while calculating.")
        self._draw_active()
        self._log(f"回路の波形: {PC.source_label(p)} の等価回路と IED ({p.ied_model}) を計算しています…")
        gc.collect()
        threading.Thread(target=self._circuit_work, args=(p,), daemon=True).start()

    def _circuit_work(self, p):
        """ワーカースレッド。Tk には触らず、queue 経由で結果を渡す。"""
        try:
            self.q.put(("circuit", PC.solve_circuit_cached(p, log=lambda s: self.q.put(("log", s)))))
        except Exception:
            self.q.put(("circuit_error", traceback.format_exc()))

    def show_circuit(self, c, select=True):
        self.circuit_busy = False
        self.btn_run.configure(state="disabled" if self.running else "normal")
        self.btn_resume.configure(state="normal" if self.result is not None and not self.running else "disabled")
        self.btn_circuit.configure(state="disabled" if self.running else "normal")
        if select and not self.running:
            self.status1.set("RF 回路・IED の計算が完了しました")
            self.state_label.set("回路完了")
        n = 3 if "vn" in c else 2
        fig = Figure(figsize=(10, 5), dpi=100, constrained_layout=True)
        PC.plot_circuit(c, *fig.subplots(1, n))
        self._install_figure("circuit", fig)
        self.views["circuit"]["caption"].set("RF 等価回路の結果（回路確認時の入力条件）\n" + PC.summary(c))
        if select:
            self.result_tabs.select(self.views["circuit"]["page"])
        self._draw_active()
        self._log(PC.summary(c))

    # ---------------- 保存・設定
    def on_save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(title="保存先 (ファイル名のみ。_fields.png などが付きます)",
                                            initialfile="trench_charging")
        if not path:
            return
        base, ext = os.path.splitext(path)
        if ext.lower() in (".png", ".npz"):
            path = base
        try:
            files = T.save_results(self.result, self.p_run, path, pyplot=False)
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))
            return
        self._log("保存: " + ", ".join(files))
        messagebox.showinfo("保存しました", "\n".join(files))

    def on_cfg_save(self):
        try:
            p = self.get_params()
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            return
        path = filedialog.asksaveasfilename(title="設定を保存", defaultextension=".json",
                                            filetypes=[("JSON", "*.json")], initialfile="trench_settings.json")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(p), f, indent=2, ensure_ascii=False)
            self._log(f"設定を保存: {path}")

    def on_cfg_load(self):
        path = filedialog.askopenfilename(title="設定を読込", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            known = asdict(T.Params())
            p = T.Params(**{k: type(known[k])(v) for k, v in d.items() if k in known})
        except Exception as e:
            messagebox.showerror("読込エラー", f"設定を読み込めませんでした: {e}")
            return
        self.set_params(p)
        self._log(f"設定を読込: {path}")

    # ---------------- その他
    def _log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def on_close(self):
        self.stop_event.set()
        self.destroy()


def main():
    try:  # Windows の高 DPI 対応 (文字のぼやけ防止)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()


if __name__ == "__main__":
    main()
