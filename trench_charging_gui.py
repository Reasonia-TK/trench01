#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
トレンチ表面帯電シミュレーター GUI (Tkinter + matplotlib)
=========================================================

trench_charging_2d.py と同じフォルダに置いて実行する:

    python trench_charging_gui.py

* 左のパネルでパラメータを入力し、[実行] を押すと計算が始まる。
* 計算は別スレッドで走り、電位・電場・側壁プロファイル・電位の時間変化などが
  バッチごとにリアルタイム更新される。[停止] でいつでも中断できる(それまでの結果は残る)。
* 実行モードは「固定長」と「飽和まで継続」から選べる。
* [結果を保存] で図 2 枚 (_fields.png, _history.png) と .npz を保存できる。
* [設定を保存/読込] でパラメータを JSON で保存・復元できる。

必要なもの: Python 3.9+, numpy, scipy, matplotlib (Tkinter は Python に同梱。
Linux では `sudo apt install python3-tk` が必要な場合がある)
"""
import json
import os
import queue
import threading
import time
import traceback
from dataclasses import asdict

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import trench_charging_2d as T


# ---------------------------------------------------------------- 入力欄の定義
# (キー, ラベル, 単位, 表示スケール(内部値 = 表示値 x scale), 型 / 選択肢)
GEOMETRY = [
    ("nx", "横セル数 (周期境界)", "セル", 1, int),
    ("dx", "セルサイズ", "nm", 1e-9, float),
    ("trench_w", "トレンチ幅", "セル", 1, int),
    ("trench_d", "トレンチ深さ", "セル", 1, int),
    ("floor_t", "底の誘電体の厚さ", "セル", 1, int),
    ("n_vac", "上部の真空領域", "セル", 1, int),
    ("eps_r", "比誘電率 (SiO2 = 3.9)", "", 1, float),
    ("top_bc", "上端の境界条件", "", 1, ("dirichlet", "neumann")),
]
PLASMA = [
    ("flux", "粒子フラックス", "m⁻²s⁻¹", 1, float),
    ("ion_mass_amu", "イオン質量 (Ar = 40)", "amu", 1, float),
    ("ion_energy_eV", "イオンエネルギー", "eV", 1, float),
    ("ion_temp_eV", "イオン横方向温度", "eV", 1, float),
    ("electron_temp_eV", "電子温度", "eV", 1, float),
]
LEAK = [
    ("sigma_s", "表面シート伝導度 (0=なし)", "S", 1, float),
]
NUMERICS = [
    ("n_per_batch", "1バッチの粒子数 (各種)", "個", 1, int),
    ("dt_batch", "1バッチの物理時間", "μs", 1e-6, float),
    ("cfl", "CFL (1ステップの移動量)", "セル", 1, float),
    ("max_steps", "粒子の最大ステップ数", "", 1, int),
    ("seed", "乱数シード", "", 1, int),
]
FIXED_FIELDS = [
    ("n_batches", "バッチ数", "", 1, int),
]
STEADY_FIELDS = [
    ("max_batches", "上限バッチ数", "", 1, int),
    ("steady_window", "判定ブロック長", "バッチ", 1, int),
    ("steady_rtol", "許容変化率", "", 1, float),
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
    if (p.nx - p.trench_w) % 2:
        return "「横セル数 - トレンチ幅」は偶数にしてください (トレンチを中央に置くため)。"
    if p.trench_d < 1 or p.floor_t < 1:
        return "トレンチ深さと底の厚さは 1 セル以上にしてください。"
    if p.n_vac < 5:
        return "上部の真空領域は 5 セル以上にしてください。"
    if p.dx <= 0 or p.eps_r < 1:
        return "セルサイズは正、比誘電率は 1 以上にしてください。"
    if p.flux <= 0 or p.ion_mass_amu <= 0 or p.ion_energy_eV <= 0 or p.electron_temp_eV <= 0:
        return "フラックス・イオン質量・イオンエネルギー・電子温度は正の値にしてください。"
    if p.ion_temp_eV < 0 or p.sigma_s < 0:
        return "イオン温度と表面シート伝導度は 0 以上にしてください。"
    if p.n_per_batch < 100 or p.dt_batch <= 0:
        return "1バッチの粒子数は 100 以上、物理時間は正の値にしてください。"
    if not (0 < p.cfl <= 1) or p.max_steps < 100:
        return "CFL は 0 より大きく 1 以下、最大ステップ数は 100 以上にしてください。"
    if p.until_steady:
        if p.steady_window < 5 or p.steady_hold < 1 or p.steady_rtol <= 0:
            return "判定ブロック長は 5 以上、連続 OK 回数は 1 以上、許容変化率は正の値にしてください。"
        if p.max_batches < 2 * p.steady_window:
            return "上限バッチ数は 判定ブロック長 の 2 倍以上にしてください。"
    elif p.n_batches < 1:
        return "バッチ数は 1 以上にしてください。"
    return None


# ---------------------------------------------------------------- リアルタイム描画
class LivePlot:
    """図の artist を最初に作っておき、データだけ差し替えて高速に更新する。"""

    def __init__(self, fig):
        self.fig = fig
        self.ready = False
        self.vlines = []
        self.lines = {}

    def setup(self, p, solid, i0):
        fig = self.fig
        fig.clear()
        self.p, self.solid, self.i0 = p, solid, i0
        self.vlines, self.lines = [], {}
        nx, ny = solid.shape
        dxn = p.dx * 1e9
        self.dxn = dxn
        xc = (np.arange(nx) + 0.5) * dxn
        yc = (np.arange(ny) + 0.5) * dxn
        extent = [0, nx * dxn, 0, ny * dxn]
        gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1])
        outline = solid.T.astype(float)

        # (a) 電位
        ax = fig.add_subplot(gs[0, 0])
        self.im_phi = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=extent,
                                cmap="RdBu_r", vmin=-1, vmax=1)
        ax.contour(xc, yc, outline, levels=[0.5], colors="k", linewidths=0.8)
        fig.colorbar(self.im_phi, ax=ax, shrink=0.75)
        ax.set_title("Potential [V]", fontsize=10)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

        # (b) 電場
        ax = fig.add_subplot(gs[0, 1])
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("0.85")
        self.im_E = ax.imshow(np.ma.masked_array(np.zeros((ny, nx)), mask=True), origin="lower", extent=extent,
                              cmap=cmap, vmin=0, vmax=1)
        ax.contour(xc, yc, outline, levels=[0.5], colors="k", linewidths=0.8)
        s = 3
        X, Y = np.meshgrid(xc[::s], yc[::s], indexing="ij")
        self.qmask = (~solid)[::s, ::s]
        self.qs = s
        nq = int(self.qmask.sum())
        self.quiv = ax.quiver(X[self.qmask], Y[self.qmask], np.zeros(nq), np.zeros(nq),
                              color="w", scale=45, width=0.003)
        fig.colorbar(self.im_E, ax=ax, shrink=0.75)
        ax.set_title("|E| [MV/m] and direction", fontsize=10)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

        # (c) 左側壁プロファイル
        ax = fig.add_subplot(gs[0, 2])
        self.iy = np.arange(p.floor_t, p.floor_t + p.trench_d)
        self.yw = (self.iy + 0.5) * dxn
        (self.l_phi,) = ax.plot([], [], "r-")
        ax.set_xlabel("Potential [V]", color="r", fontsize=9)
        ax.set_ylabel("y [nm]")
        ax.set_ylim(0, ny * dxn)
        ax.grid(alpha=0.3)
        ax2 = ax.twiny()
        (self.l_sig,) = ax2.plot([], [], "b--")
        ax2.set_xlabel("Surface charge [mC/m$^2$]", color="b", fontsize=9)
        ax.set_title("Left sidewall profile", fontsize=10, pad=28)
        self.ax_wall, self.ax_wall2 = ax, ax2

        # (d) 電位の時間変化
        self.ax_h = fig.add_subplot(gs[1, 0])
        self.ax_h.set_xlabel("time [ms]"); self.ax_h.set_ylabel("surface potential [V]")
        self.ax_h.grid(alpha=0.3)

        # (e) 底でのフラックス比
        self.ax_f = fig.add_subplot(gs[1, 1])
        (self.l_fi,) = self.ax_f.plot([], [], label="ion (bottom)")
        (self.l_fe,) = self.ax_f.plot([], [], label="electron (bottom)")
        self.ax_f.set_xlabel("time [ms]"); self.ax_f.set_ylabel("flux / incident flux")
        self.ax_f.legend(fontsize=8); self.ax_f.grid(alpha=0.3)

        # (f) 表面リーク電流
        self.ax_l = fig.add_subplot(gs[1, 2])
        self.ax_l.set_xlabel("time [ms]")
        self.ax_l.grid(alpha=0.3)
        if p.sigma_s > 0:
            (self.l_leak,) = self.ax_l.plot([], [])
            self.ax_l.set_ylabel("leak current / ion current")
        else:
            self.l_leak = None
            self.ax_l.text(0.5, 0.5, "surface leakage off", transform=self.ax_l.transAxes,
                           ha="center", va="center", color="0.5")
        self.ready = True

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

        iw = self.i0 - 1
        self.l_phi.set_data(phi[iw, self.iy], self.yw)
        self.l_sig.set_data(rho[iw, self.iy] * p.dx * 1e3, self.yw)
        for a in (self.ax_wall, self.ax_wall2):
            a.relim()
            a.autoscale_view(scaley=False)

        t = hist["t"] * 1e3
        if not self.lines:
            for k in d["probes"]:
                (self.lines[k],) = self.ax_h.plot([], [], label=k)
            self.ax_h.legend(fontsize=7, loc="upper left")
        for k, line in self.lines.items():
            line.set_data(t, hist[k])
        n = len(t)
        k = min(10, max(1, n // 5))
        ma = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
        self.l_fi.set_data(t[k - 1:], ma(hist["ratio_i_bottom"]))
        self.l_fe.set_data(t[k - 1:], ma(hist["ratio_e_bottom"]))
        if self.l_leak is not None:
            self.l_leak.set_data(t[k - 1:], ma(hist["leak"]))
        for a in (self.ax_h, self.ax_f, self.ax_l):
            a.relim()
            a.autoscale_view()

        for v in self.vlines:
            v.remove()
        self.vlines = []
        tc = d.get("t_conv", np.nan)
        if tc is not None and np.isfinite(tc):
            for a in (self.ax_h, self.ax_f, self.ax_l):
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

        self.mode = tk.StringVar(value="fixed")
        self.info = tk.StringVar()
        self.vars, self.specs, self.entries, self.radios = {}, {}, {}, []

        self._build_left()
        self._build_right()
        self.set_params(T.Params())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll)

    # ---------------- 左: パラメータ入力
    def _build_left(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y")
        cv = tk.Canvas(left, width=350, highlightthickness=0)
        vsb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=vsb.set)
        cv.pack(side="left", fill="y")
        vsb.pack(side="right", fill="y")

        def wheel(e):
            cv.yview_scroll(-1 if (e.num == 4 or e.delta > 0) else 1, "units")
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            cv.bind(ev, wheel)
            inner.bind(ev, wheel)

        self._group(inner, "形状", GEOMETRY)
        self._group(inner, "プラズマ", PLASMA)
        self._group(inner, "表面リーク", LEAK)

        g = ttk.LabelFrame(inner, text="実行モード")
        g.pack(fill="x", padx=6, pady=4)
        r = ttk.Radiobutton(g, text="固定長 (バッチ数を指定)", variable=self.mode, value="fixed",
                            command=self.apply_states)
        r.grid(row=0, column=0, columnspan=3, sticky="w", padx=4)
        self.radios.append(r)
        row = 1
        for spec in FIXED_FIELDS:
            self._field(g, row, spec); row += 1
        r = ttk.Radiobutton(g, text="飽和まで継続", variable=self.mode, value="steady",
                            command=self.apply_states)
        r.grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 0))
        self.radios.append(r)
        row += 1
        for spec in STEADY_FIELDS:
            self._field(g, row, spec); row += 1

        self._group(inner, "数値設定", NUMERICS)

        g = ttk.LabelFrame(inner, text="構成の確認")
        g.pack(fill="x", padx=6, pady=4)
        ttk.Label(g, textvariable=self.info, justify="left", wraplength=320).pack(anchor="w", padx=6, pady=4)

    def _group(self, parent, title, specs):
        g = ttk.LabelFrame(parent, text=title)
        g.pack(fill="x", padx=6, pady=4)
        for i, spec in enumerate(specs):
            self._field(g, i, spec)

    def _field(self, parent, row, spec):
        key, label, unit, scale, typ = spec
        self.specs[key] = (label, unit, scale, typ)
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(14, 4), pady=1)
        if isinstance(typ, tuple):
            w = ttk.Combobox(parent, textvariable=var, values=typ, width=11, state="readonly")
            w.bind("<<ComboboxSelected>>", self.update_info)
        else:
            w = ttk.Entry(parent, textvariable=var, width=12, justify="right")
            w.bind("<KeyRelease>", self.update_info)
        w.grid(row=row, column=1, padx=2, pady=1)
        ttk.Label(parent, text=unit, width=7).grid(row=row, column=2, sticky="w")
        self.entries[key] = w

    # ---------------- 右: ボタン・グラフ・ログ
    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        bar = ttk.Frame(right)
        bar.pack(fill="x", padx=6, pady=(6, 2))
        self.btn_run = ttk.Button(bar, text="▶ 実行", command=self.on_run, width=10)
        self.btn_stop = ttk.Button(bar, text="■ 停止", command=self.on_stop, width=10, state="disabled")
        self.btn_save = ttk.Button(bar, text="結果を保存…", command=self.on_save, state="disabled")
        self.btn_cfg_save = ttk.Button(bar, text="設定を保存…", command=self.on_cfg_save)
        self.btn_cfg_load = ttk.Button(bar, text="設定を読込…", command=self.on_cfg_load)
        self.btn_reset = ttk.Button(bar, text="既定値に戻す", command=lambda: self.set_params(T.Params()))
        for b in (self.btn_run, self.btn_stop, self.btn_save, self.btn_cfg_save, self.btn_cfg_load, self.btn_reset):
            b.pack(side="left", padx=2)

        self.pbar = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.pbar.pack(fill="x", padx=8, pady=(4, 2))
        self.status1 = tk.StringVar(value="待機中 — パラメータを設定して [実行] を押してください。")
        self.status2 = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status1, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=8)
        ttk.Label(right, textvariable=self.status2).pack(anchor="w", padx=8)

        self.fig = Figure(figsize=(8, 5.5), dpi=100, constrained_layout=True)
        self.plot = LivePlot(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tb = ttk.Frame(right)
        tb.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(self.canvas, tb)

        logf = ttk.Frame(right)
        logf.pack(side="bottom", fill="x", padx=6, pady=(0, 4))
        self.log_text = tk.Text(logf, height=7, state="disabled", wrap="none", font=("TkFixedFont", 9))
        lsb = ttk.Scrollbar(logf, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=lsb.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        lsb.pack(side="right", fill="y")

        widget = self.canvas.get_tk_widget()
        widget.pack(side="top", fill="both", expand=True)
        # 画面の拡大率(DPI)の違いで図がウィジェットよりはみ出すことがあるため、
        # matplotlib 標準のリサイズ処理の後で、図の大きさをウィジェットに合わせ直す
        widget.bind("<Configure>", self._fit_figure, add="+")

    def _fit_figure(self, event=None):
        """図のピクセルサイズをウィジェットに合わせる (ずれている時だけ)。"""
        w = self.canvas.get_tk_widget()
        pw, ph = max(w.winfo_width(), 50), max(w.winfo_height(), 50)
        dpi = self.fig.dpi
        cur = self.fig.get_size_inches() * dpi
        if abs(cur[0] - pw) > 2 or abs(cur[1] - ph) > 2:
            self.fig.set_size_inches(pw / dpi, ph / dpi, forward=False)
            self.canvas.draw_idle()

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
            except ValueError:
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
        except ValueError:
            self.info.set("(入力値を確認してください)")
            return
        ny = p.floor_t + p.trench_d + p.n_vac
        lines = [f"格子 {p.nx} × {ny} セル (領域幅 {p.nx * p.dx * 1e9:.0f} nm)",
                 f"トレンチ 幅 {p.trench_w * p.dx * 1e9:.0f} nm × 深さ {p.trench_d * p.dx * 1e9:.0f} nm "
                 f"(AR = {p.trench_d / p.trench_w:.1f})"]
        if p.ion_energy_eV > 0 and p.ion_temp_eV >= 0:
            lines.append(f"イオン角度広がり ≈ {np.degrees(np.sqrt(p.ion_temp_eV / (2 * p.ion_energy_eV))):.1f}°")
        if p.until_steady:
            lines.append(f"飽和判定 {p.steady_window * p.dt_batch * 1e3:.1f} ms ごと / 上限 "
                         f"{p.max_batches * p.dt_batch * 1e3:.0f} ms")
        else:
            lines.append(f"総帯電時間 {p.n_batches * p.dt_batch * 1e3:.1f} ms")
        self.info.set("\n".join(lines))

    # ---------------- 実行・停止
    def on_run(self):
        if self.running:
            return
        try:
            p = self.get_params()
            err = validate(p)
            if err:
                raise ValueError(err)
        except ValueError as e:
            messagebox.showerror("入力エラー", str(e))
            return

        self.p_run, self.result = p, None
        self.stop_event = threading.Event()
        self.pending, self.last_check = None, None
        solid, i0, _ = T.build_geometry(p)
        self.plot.setup(p, solid, i0)
        self.canvas.draw_idle()

        self._log("---- 実行開始 ----")
        self.pbar.configure(value=0)
        self.status1.set("計算を開始しています…")
        self.status2.set("")
        self.set_running(True)
        threading.Thread(target=self._work, args=(p, self.stop_event), daemon=True).start()

    def _work(self, p, stop):
        """ワーカースレッド。Tk には触らず、queue 経由でメインスレッドに渡す。"""
        try:
            res = T.run(p, log=lambda s: self.q.put(("log", s)),
                        progress=lambda info: self.q.put(("progress", info)), stop=stop)
            self.q.put(("done", res))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def on_stop(self):
        self.stop_event.set()
        self.btn_stop.configure(state="disabled")
        self.status1.set("停止しています (現在のバッチが終わるまで少しお待ちください)…")

    def set_running(self, running):
        self.running = running
        self.btn_run.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")
        for b in (self.btn_save, self.btn_cfg_save, self.btn_cfg_load, self.btn_reset):
            b.configure(state="disabled" if running else "normal")
        if not running and self.result is None:
            self.btn_save.configure(state="disabled")
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
                    self._log(payload)
                    messagebox.showerror("計算エラー", payload.splitlines()[-1])
        except queue.Empty:
            pass
        self._fit_figure()
        if self.pending is not None and time.time() - self.last_draw > 0.3:
            self.on_progress(self.pending)
            self.pending = None
        self.after(100, self.poll)

    def on_progress(self, info):
        b, n_max, el = info["batch"], info["n_max"], info["elapsed"]
        p = self.p_run
        self.pbar.configure(value=100.0 * b / n_max)
        t_ms = b * p.dt_batch * 1e3
        s1 = f"バッチ {b}/{n_max}  (t = {t_ms:.1f} ms)   経過 {fmt_time(el)}"
        if not p.until_steady and b < n_max:
            s1 += f"   残り約 {fmt_time(el / b * (n_max - b))}"
        self.status1.set(s1)

        h = info["hist"]
        k = min(10, b)
        r = np.mean(h["ratio_e_bottom"][-k:]) / max(np.mean(h["ratio_i_bottom"][-k:]), 1e-9)
        s2 = (f"電位: 底 {h['bottom center'][-1]:.1f} V | 側壁 上 {h['sidewall upper'][-1]:.1f} / "
              f"中 {h['sidewall middle'][-1]:.1f} / 下 {h['sidewall lower'][-1]:.1f} V | "
              f"マスク上面 {h['mask top'][-1]:.1f} V | 底の電子/イオン比 {r:.2f}")
        if info["check"] is not None:
            self.last_check = info["check"]
        if p.until_steady and self.last_check is not None:
            c = self.last_check
            s2 += (f"\n飽和判定: {c['name']} の変化 {c['d']:+.2f} V/ブロック (許容 {c['tol']:.2f} V) → "
                   f"{'OK' if c['ok'] else 'NG'} ({c['n_ok']}/{p.steady_hold})")
        self.status2.set(s2)

        self.plot.update(info)
        self.canvas.draw_idle()
        self.last_draw = time.time()

    def on_done(self, res):
        self.result = res
        self.set_running(False)
        if res is None:
            self.status1.set("中断しました (結果なし)")
            self.btn_save.configure(state="disabled")
            return
        self.plot.update(res)
        self.canvas.draw_idle()
        self.last_draw = time.time()
        n = len(res["hist"]["t"])
        t_ms = res["hist"]["t"][-1] * 1e3
        if res["stopped"]:
            msg = f"中断しました (t = {t_ms:.1f} ms, {n} バッチ)"
        elif np.isfinite(res["t_conv"]):
            msg = f"飽和に到達しました (t = {res['t_conv'] * 1e3:.1f} ms, {n} バッチ)"
        elif self.p_run.until_steady:
            msg = f"上限 ({n} バッチ, {t_ms:.0f} ms) までに飽和しませんでした — 上限バッチ数を増やしてください"
        else:
            msg = f"完了しました (t = {t_ms:.1f} ms, {n} バッチ)"
        self.status1.set(msg)
        self.pbar.configure(value=100 if not res["stopped"] else self.pbar["value"])
        self.status2.set("飽和値: " + " | ".join(f"{k} {v:.1f} V" for k, v in res["steady"].items()))
        self.btn_save.configure(state="normal")

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
