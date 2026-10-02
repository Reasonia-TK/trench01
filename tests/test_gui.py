"""ダッシュボードの表示データ、非同期操作、結果の再利用を実際の Tk で確認する。"""
import importlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def app_window():
    tk = pytest.importorskip("tkinter")
    try:
        G = importlib.import_module("trench_charging_gui")
        a = G.App()
    except (tk.TclError, ImportError) as e:
        if any(reason in str(e).lower() for reason in ("no display name", "couldn't connect to display", "headless")):
            pytest.skip(f"Tk の画面を利用できません: {e}")
        raise
    a.withdraw()
    errors = []
    a.report_callback_exception = lambda *args: errors.append(args)
    yield a
    a.on_close()
    assert not errors, errors


@pytest.fixture
def app(app_window):
    # Tk はユーザーの起動と同じく1プロセス1インタープリターにする。
    # Windows Tcl の再初期化を各テストで繰り返さず、状態だけを独立させる。
    import queue
    import threading
    G = sys.modules["trench_charging_gui"]
    a = app_window
    a.result = a.p_run = a.pending = a.last_check = None
    a.circuit_busy = False
    a.stop_event = threading.Event()
    a.q = queue.Queue()
    a.set_params(G.T.Params())
    a.set_running(False)
    a.result_tabs.select(a.views["fields"]["page"])
    yield a


def small_params(app, **changes):
    G = sys.modules["trench_charging_gui"]
    return replace(G.T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                   mask_t=4, n_vac=8, bias="dc", n_batches=3, n_per_batch=200,
                   max_steps=800, probe_ions=120, probe_trajectories=8, **changes)


def wait_for_worker(app, timeout=20):
    deadline = time.monotonic() + timeout
    while app.running and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert not app.running, "ワーカーが終了しませんでした"
    app.update()


def calculate(app, **changes):
    app.set_params(small_params(app, **changes))
    app.on_run()
    wait_for_worker(app)
    assert app.result is not None
    return app.result


def test_live_views_and_metrics_use_the_calculation_data(app):
    res = calculate(app)
    p = app.p_run
    np.testing.assert_array_equal(app.plot.im_phi.get_array(), res["phi"].T)
    np.testing.assert_array_equal(app.plot.l_charge.get_ydata(), res["charge_hist"]["left"]*1e12)
    np.testing.assert_array_equal(app.plot.r_charge.get_ydata(), res["charge_hist"]["right"]*1e12)
    np.testing.assert_array_equal(app.plot.r_phi.get_xdata(), res["phi"][res["i1"], app.plot.iy])
    assert float(app.metrics["bottom"].get()) == pytest.approx(res["hist"]["bottom center"][-1], abs=.005)
    assert float(app.metrics["time"].get()) == pytest.approx(p.n_batches*p.dt_batch*1e3)
    assert "完了" == app.state_label.get()
    assert str(app.btn_save.cget("state")) == "normal"
    assert str(app.btn_probe.cget("state")) == "disabled"


def test_taper_inputs_preview_profiles_and_resume_use_the_actual_geometry(app, tmp_path, monkeypatch):
    G = sys.modules["trench_charging_gui"]
    p = small_params(app, trench_taper_deg=8, mask_taper_deg=20, sigma_s=1e-14)
    app.set_params(p)
    assert app.get_params().trench_taper_deg == 8
    assert app.get_params().mask_taper_deg == 20
    app._preview_geometry()
    assert "テーパー 8°" in app.info.get() and "入口幅" in app.info.get()
    materials = G.T.build_geometry(p)[0].astype(int)+G.T.build_geometry(p)[1].astype(int)
    np.testing.assert_array_equal(app.plot.fig.axes[0].images[0].get_array(), materials.T)
    app.on_run()
    wait_for_worker(app)
    res = app.result
    assert res is not None
    for side in ("left", "right"):
        x, y = G.T.wall_profiles(res["solid"], app.p_run)[side]
        line = app.plot.l_phi if side == "left" else app.plot.r_phi
        np.testing.assert_array_equal(line.get_xdata(), res["phi"][x, y])
        np.testing.assert_allclose(line.get_ydata(), (y+.5)*p.dx*1e9, atol=1e-12, rtol=0)
    G.T.save_results(res, app.p_run, str(tmp_path/"taper"))
    monkeypatch.setattr(G.filedialog, "askopenfilename", lambda **_: str(tmp_path/"taper.npz"))
    app.on_load_result()
    assert app.get_params().mask_taper_deg == 20
    app.vars["n_batches"].set("1")
    app.on_resume()
    wait_for_worker(app)
    assert len(app.result["hist"]["t"]) == 4
    assert app.p_run.trench_taper_deg == 8


@pytest.mark.parametrize("key,value", [("trench_taper_deg", "45"), ("mask_taper_deg", "89"),
                                       ("mask_taper_deg", "nan")])
def test_invalid_taper_does_not_replace_preview_or_result(app, key, value):
    res = calculate(app)
    image = app.plot.im_phi.get_array().copy()
    app.vars[key].set(value)
    app.update_info()
    app._preview_geometry()
    assert "入力を確認" in app.info.get()
    assert app.result is res
    np.testing.assert_array_equal(app.plot.im_phi.get_array(), image)


def test_taper_preview_text_fits_minimum_height_and_expands_without_overlap(app):
    p = small_params(app, trench_taper_deg=8.123456, mask_taper_deg=20.123456)
    for width, height in ((720, 225), (720, 440), (1100, 380), (1100, 440)):
        app.plot.fig.set_size_inches(width/100, height/100)
        app.plot.preview(p)
        app.plot.fit_layout("fields", height)
        app.plot.fig.canvas.draw()
        renderer = app.plot.fig.canvas.get_renderer()
        title = app.plot.preview_title.get_window_extent(renderer)
        detail = app.plot.preview_details.get_window_extent(renderer)
        axes = app.plot.preview_details.axes.get_window_extent(renderer)
        assert title.y0 > detail.y1
        assert detail.y0 >= axes.y0 and detail.x1 < app.plot.fig.bbox.x1
        assert detail.x0 > app.plot.fig.axes[0].get_window_extent(renderer).x1
        if app.plot.preview_footer.get_visible():
            footer = app.plot.preview_footer.get_window_extent(renderer)
            assert footer.y1 < detail.y0


def test_resume_button_appends_batches_and_matches_continuous_calculation(app):
    G = sys.modules["trench_charging_gui"]
    part = calculate(app)
    old_charge = part["rho_raw"].copy()
    assert str(app.btn_resume.cget("state")) == "normal"
    app.on_resume()
    assert app.running and app.state_label.get() == "再開中"
    assert str(app.btn_resume.cget("state")) == "disabled"
    wait_for_worker(app)
    assert len(app.result["hist"]["t"]) == 6 and app.result["start_batch"] == 3
    np.testing.assert_array_equal(part["rho_raw"], old_charge)
    np.testing.assert_array_equal(app.result["hist"]["t"][:3], part["hist"]["t"])
    full = G.T.run(replace(app.p_run, n_batches=6), log=lambda _: None)
    np.testing.assert_array_equal(app.result["rho_raw"], full["rho_raw"])
    np.testing.assert_array_equal(app.result["hist"]["bottom center"], full["hist"]["bottom center"])
    assert str(app.btn_resume.cget("state")) == "normal"


def test_loaded_result_resumes_to_total_cap_in_steady_mode(app, monkeypatch, tmp_path):
    G = sys.modules["trench_charging_gui"]
    part = calculate(app)
    G.T.save_results(part, app.p_run, str(tmp_path / "partial"))
    monkeypatch.setattr(G.filedialog, "askopenfilename", lambda **_: str(tmp_path / "partial.npz"))
    app.on_load_result()
    app.mode.set("steady")
    app.vars["steady_window"].set("5")
    app.vars["max_batches"].set("12")
    app.apply_states()
    app.on_resume()
    wait_for_worker(app)
    assert len(app.result["hist"]["t"]) == 12
    assert app.result["start_batch"] == 3 and app.p_run.until_steady
    np.testing.assert_array_equal(app.result["charge_hist"]["right"][:3], part["charge_hist"]["right"])


def test_invalid_resume_keeps_result_and_reports_changed_physics(app, monkeypatch):
    G = sys.modules["trench_charging_gui"]
    part = calculate(app)
    errors = []
    monkeypatch.setattr(G.messagebox, "showerror", lambda *args: errors.append(args))
    app.vars["trench_offset"].set("1")
    app.on_resume()
    assert not app.running and app.result is part
    assert errors and "trench_offset" in errors[-1][1]
    assert str(app.btn_resume.cget("state")) == "normal"


def test_stopped_resume_keeps_charge_and_can_continue_again(app):
    G = sys.modules["trench_charging_gui"]
    part = calculate(app)
    app.vars["n_batches"].set("1000")
    app.on_resume()
    app.on_stop()
    wait_for_worker(app)
    interrupted = app.result
    assert interrupted["stopped"] and len(interrupted["hist"]["t"]) >= 3
    np.testing.assert_array_equal(interrupted["charge_hist"]["left"][:3], part["charge_hist"]["left"])
    app.vars["n_batches"].set("2")
    app.on_resume()
    wait_for_worker(app)
    n = len(interrupted["hist"]["t"])+2
    assert len(app.result["hist"]["t"]) == n and not app.result["stopped"]
    full = G.T.run(replace(app.p_run, n_batches=n), log=lambda _: None)
    np.testing.assert_array_equal(app.result["rho_raw"], full["rho_raw"])


def test_charge_criterion_controls_and_progress_use_charge_units(app):
    G = sys.modules["trench_charging_gui"]
    p = replace(small_params(app), until_steady=True, max_batches=100,
                steady_charge_rtol=0.007, steady_charge_atol=2e-15)
    app.set_params(p)
    assert float(app.vars["steady_charge_rtol"].get()) == pytest.approx(0.7)
    assert float(app.vars["steady_charge_atol"].get()) == pytest.approx(0.002)
    assert app.get_params().steady_charge_rtol == pytest.approx(p.steady_charge_rtol)
    assert app.get_params().steady_charge_atol == pytest.approx(p.steady_charge_atol)
    assert G.validate(replace(p, steady_charge_rtol=0))
    assert G.validate(replace(p, steady_charge_atol=-1))
    assert str(app.entries["steady_charge_atol"].cget("state")) == "normal"
    app.p_run = p
    res = G.T.run(replace(p, until_steady=False, n_batches=3), log=lambda _: None)
    charge = dict(name="right", kind="drift", d=537e-12, tol=50e-12,
                  unit="C/m", ok=False)
    check = {**charge, "n_ok": 0, "charge": charge, "voltage": {"ok": True}}
    app.on_progress({**res, "batch": 3, "n_max": p.max_batches, "elapsed": .1, "check": check})
    status = app.status2.get()
    assert "電位 OK / 電荷 NG" in status and "正味の充電" in status
    assert "+537 pC/m" in status and "50 pC/m" in status and "計算継続" in status


def test_rf_cycle_controls_and_history_show_their_distinct_meaning(app):
    p = replace(small_params(app), bias="rf", field_mode="circuit_rf", side_bc="pillar", rf_phase=90)
    app.set_params(p)
    app.apply_states()
    assert str(app.entries["rf_phase"].cget("state")) == "normal"
    assert str(app.entries["rf_time_steps"].cget("state")) == "normal"
    assert str(app.entries["substrate_voltage"].cget("state")) == "disabled"
    app.on_run()
    wait_for_worker(app)
    assert app.result is not None and app.p_run.field_mode == "circuit_rf"
    assert "周期平均" in app.result_label.get() and "90°" in app.result_label.get()
    assert "cycle mean" in app.plot.ax_h.get_title()
    np.testing.assert_array_equal(app.plot.im_phi.get_array(), app.result["phi"].T)
    np.testing.assert_array_equal(app.plot.lines["bottom center"].get_ydata(), app.result["hist"]["bottom center"])


def test_rf_cycle_gui_probe_uses_phase_tab_and_frozen_saturation_charge(app):
    p = replace(small_params(app), bias="rf", field_mode="circuit_rf", side_bc="pillar",
                sigma_s=1e-14, until_steady=True, max_batches=300, steady_window=15,
                steady_hold=2, n_per_batch=128, max_steps=1000, probe_energy_eV=1.5)
    app.set_params(p)
    app.on_run()
    wait_for_worker(app)
    assert app.result["converged"]
    before = app.result["rho_raw"].copy()
    app.on_probe()
    wait_for_worker(app)
    assert app.result["ion_probe"]["rf_cycle"]
    assert len(app.views["phase"]["fig"].axes) == 2
    assert "入射位相" in app.views["phase"]["caption"].get()
    assert len(app.views["ions"]["fig"].axes) == 6
    np.testing.assert_array_equal(before, app.result["rho_raw"])


@pytest.mark.parametrize("value", ["0", "", "nan", "inf"])
def test_partial_or_invalid_input_does_not_crash_or_replace_results(app, value):
    res = calculate(app)
    image = app.plot.im_phi.get_array().copy()
    app.vars["trench_w"].set(value)
    app.update_info()
    app._preview_geometry()
    assert app.info.get()
    assert app.result is res
    np.testing.assert_array_equal(app.plot.im_phi.get_array(), image)


def test_stop_restores_controls(app):
    app.set_params(replace(small_params(app), n_batches=1000))
    app.on_run()
    assert app.running
    assert str(app.entries["trench_offset"].cget("state")) == "disabled"
    app.on_stop()
    wait_for_worker(app)
    assert app.state_label.get() == "中断"
    assert str(app.btn_run.cget("state")) == "normal"
    assert str(app.entries["trench_offset"].cget("state")) == "normal"
    if app.result is not None:
        assert app.result["stopped"]
        assert len(app.result["hist"]["t"]) < 1000


def test_compact_layout_preserves_all_series_and_can_be_restored(app):
    res = calculate(app)
    app.plot.fit_layout("history", 280)
    app.plot.fit_layout("walls", 280)
    assert app.plot.ax_h.get_subplotspec().get_gridspec().nrows == 1
    np.testing.assert_array_equal(app.plot.lines["bottom center"].get_ydata(), res["hist"]["bottom center"])
    app.plot.fit_layout("history", 500)
    app.plot.fit_layout("walls", 500)
    assert app.plot.ax_h.get_subplotspec().get_gridspec().nrows == 2
    np.testing.assert_array_equal(app.plot.r_charge.get_ydata(), res["charge_hist"]["right"]*1e12)


def test_loaded_saturation_result_restores_diagnostic_and_keeps_its_geometry(app, monkeypatch, tmp_path):
    G = sys.modules["trench_charging_gui"]
    p = replace(small_params(app), until_steady=True, max_batches=300, steady_window=15, steady_hold=2,
                n_per_batch=500, trench_offset=3, field_mode="fixed", substrate_voltage=-20,
                mask_voltage=-10, side_bc="pillar", sigma_s=1e-14, probe_energy_eV=30)
    res = G.T.run(p, log=lambda _: None)
    assert np.isfinite(res["t_conv"])
    res["ion_probe"] = G.T.probe_vertical_ions(res, p, log=lambda _: None)
    G.T.save_results(res, p, str(tmp_path / "charged"))
    monkeypatch.setattr(G.filedialog, "askopenfilename", lambda **_: str(tmp_path / "charged.npz"))
    app.on_load_result()
    assert app.p_run.nx == p.nx
    assert app.views["ions"]["fig"].axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().nrows == 2
    assert str(app.btn_probe.cget("state")) == "normal"
    before = app.result["rho_raw"].copy()
    app.vars["nx"].set("30")
    app.on_probe()
    wait_for_worker(app)
    assert app.p_run.nx == p.nx
    np.testing.assert_array_equal(app.result["rho_raw"], before)
    assert app.result["ion_probe"]["charged"]["Ex"].shape == before.shape
    assert app.result_tabs.select() == str(app.views["ions"]["page"])


def test_legacy_charge_growth_is_shown_as_unsaturated_and_disables_new_probe(app, monkeypatch, tmp_path):
    G = sys.modules["trench_charging_gui"]
    p = replace(small_params(app), until_steady=True, max_batches=300, steady_window=15, steady_hold=2,
                n_per_batch=500, field_mode="fixed", substrate_voltage=-20,
                mask_voltage=-10, side_bc="pillar", sigma_s=1e-14, probe_energy_eV=30)
    res = G.T.run(p, log=lambda _: None)
    assert res["converged"]
    res["ion_probe"] = G.T.probe_vertical_ions(res, p, log=lambda _: None)
    # 旧電位判定だけを通過し、電荷ログには増加が続いていた保存結果を再現する。
    res["charge_hist"] = {side: np.arange(len(res["hist"]["t"]))*1e-12 for side in ("left", "right")}
    res.pop("charge_monitor_hist")
    res["steady_criterion_version"] = 0
    G.T.save_results(res, p, str(tmp_path / "legacy"))
    monkeypatch.setattr(G.filedialog, "askopenfilename", lambda **_: str(tmp_path / "legacy.npz"))
    app.on_load_result()
    assert app.state_label.get() == "未飽和" and not app.result["converged"]
    assert "電荷は飽和していません" in app.status1.get()
    assert str(app.btn_probe.cget("state")) == "disabled"
    assert "未飽和の保存電荷" in app.views["ions"]["caption"].get()
    np.testing.assert_array_equal(app.result["rho_raw"], res["rho_raw"])
    assert str(app.btn_save.cget("state")) == "normal"


def test_expanded_figure_is_an_independent_snapshot(app):
    calculate(app)
    app.expand_plot()
    windows = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
    assert len(windows) == 1
    assert app.views["fields"]["fig"].canvas is app.canvas
    app.update()
    windows[0].destroy()


def test_rf_circuit_worker_displays_in_its_tab_and_restores_run_button(app):
    app.set_params(replace(small_params(app), bias="rf", ied_model="instant"))
    app.on_circuit()
    assert app.circuit_busy
    assert str(app.btn_run.cget("state")) == "disabled"
    deadline = time.monotonic()+20
    while app.circuit_busy and time.monotonic() < deadline:
        app.update()
        time.sleep(.01)
    assert not app.circuit_busy
    assert str(app.btn_run.cget("state")) == "normal"
    assert len(app.views["circuit"]["fig"].axes) == 2
    assert "RF" in app.views["circuit"]["caption"].get()
    assert app.result_tabs.select() == str(app.views["circuit"]["page"])
