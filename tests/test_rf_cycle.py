"""RF 時間依存場の解析解、周期平均、CUDA、保存・診断の検証。"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T
import trench_rf as RF


def params(**changes):
    base = replace(T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                   mask_t=4, n_vac=8, field_mode="circuit_rf", side_bc="pillar",
                   rf_freq=2.5e9, sigma_s=0, n_batches=3, n_per_batch=128,
                   max_steps=1000, probe_ions=128, probe_trajectories=8)
    return replace(base, **changes)


def waveform(period=4e-10, amplitude=35):
    t = np.arange(256) * period / 256
    up = 10 + 2*np.cos(2*np.pi*t/period)
    uw = -80 - amplitude*np.sin(2*np.pi*t/period) + up
    um = -40 - amplitude/2*np.sin(2*np.pi*t/period) + up
    return dict(t=t, uw=uw, um=um, up=up, V1=up-um, us=uw.copy(), i=t*0,
                E=np.zeros(0), cached=False, ied_computed=False, sheath=None,
                ied_model="sheath", n_periods=1, elements={})


@pytest.mark.parametrize("side_bc", ["periodic", "pillar", "fixed"])
def test_linear_basis_matches_independent_poisson_at_every_phase(side_bc):
    p = params(side_bc=side_bc, left_voltage=-20, right_voltage=5)
    c = waveform()
    solid, mask, _, _ = T.build_geometry(p)
    f = T.setup_fields(p, solid, mask, c)
    for phase in (0, 45, 90, 180, 270, 359.9, 405):
        snapshot, _, rhs, volts = T.field_snapshot(f, f["external"], replace(p, rf_phase=phase))
        direct = T.setup_fields(replace(p, field_mode="circuit", rf_phase=phase), solid, mask, c)
        np.testing.assert_allclose(snapshot, direct["external"], atol=2e-12, rtol=2e-13)
        np.testing.assert_allclose(rhs, direct["boundary_rhs"], atol=1e3, rtol=2e-13)
        assert volts["substrate"] == pytest.approx(direct["volts"]["substrate"])
        assert volts["mask"] == pytest.approx(direct["volts"]["mask"])


def test_stratified_launch_covers_full_cycle_without_position_order():
    times = RF.launch_times(np.random.default_rng(123), 128, 1e-8)
    np.testing.assert_array_equal(np.sort((times/1e-8*128).astype(int)), np.arange(128))
    assert abs(np.corrcoef(times, np.arange(128))[0, 1]) < .2
    p = params()
    solid, mask, _, _ = T.build_geometry(p)
    rf = T.setup_fields(p, solid, mask, waveform())["rf"]
    np.testing.assert_allclose(RF.coefficients(rf, times), RF.coefficients(rf, times + 4e-10), atol=2e-12)


def analytic_trajectory(steps, tracer=T.trace_cpu):
    # 空間一様の E_y=A sin(omega t)。壁に達しない粒子を 2/3 周期追跡。
    p = replace(T.Params(), dx=.1, max_steps=steps*2//3, probe_trajectories=4)
    shape = (3, 100)
    vac, z = np.ones(shape, bool), np.zeros(shape)
    n = 12
    initial = np.linspace(0, .9, n)
    wave = np.column_stack((.05*np.sin(np.arange(4096)*2*np.pi/4096), np.zeros(4096)))
    rf = dict(period=1., dt_max=1/steps, initial_time=initial, wave=wave,
              Ex=np.zeros((2, *shape)), Ey=np.array([np.ones(shape), z]), phi=np.zeros((2, *shape)))
    record = {}
    tracer(np.full(n, .15), np.full(n, 8.), np.zeros(n), np.full(n, -.1),
           np.ones(n), np.zeros(n, np.int8), z, z, vac, p, record=record, rf=rf)
    t, omega = record["flight_time"], 2*np.pi
    expected_v = -.1 + .05/omega*(np.cos(omega*initial)-np.cos(omega*(initial+t)))
    expected_y = 8 - .1*t + .05/omega*np.cos(omega*initial)*t - .05/omega**2*(
        np.sin(omega*(initial+t))-np.sin(omega*initial))
    assert np.all(record["status"] == 3)
    np.testing.assert_allclose(t, 2/3, atol=1e-13)
    error = max(np.max(np.abs(record["final_velocity"][:, 1]-expected_v)),
                np.max(np.abs(record["endpoints"][:, 1]-expected_y)))
    return error, record


def test_flight_clock_force_matches_sinusoidal_analytic_solution_and_converges():
    coarse, _ = analytic_trajectory(96)
    fine, d = analytic_trajectory(192)
    assert fine < 2e-6
    assert coarse > fine*3.8
    np.testing.assert_allclose(d["final_phase"], (d["launch_phase"]+240)%360, atol=1e-10)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_time_dependent_wall_barrier_uses_instantaneous_voltage(backend):
    p = replace(T.Params(), dx=.001, max_steps=2, probe_trajectories=1)
    vac = np.ones((3, 4), bool)
    vac[:, :2] = False
    z = np.zeros(vac.shape)
    basis = np.zeros((2, *vac.shape)); basis[0, :, :2] = 1
    rf = dict(period=1., dt_max=1., initial_time=np.zeros(1), wave=np.array([[1., 0.], [1., 0.]]),
              Ex=np.zeros_like(basis), Ey=np.zeros_like(basis), phi=basis)
    args = (np.array([.0015]), np.array([.00201]), np.zeros(1), np.array([-.1]),
            np.ones(1), np.zeros(1, np.int8), z, z, vac, p, (z, np.ones_like(z)))
    tracer = T.trace_cpu
    if backend == "cuda":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA GPU がありません")
        from trench_cuda import CudaTracer
        tracer = CudaTracer(vac)
    d = {}
    ci, _, _, lost = tracer(*args, record=d, rf=rf)
    assert ci.sum() == 0 and lost == 1 and d["final_velocity"][0, 1] > 0
    ci, _, _, lost = tracer(*args)
    assert ci.sum() == 1 and lost == 0


def test_constant_wave_reduces_to_frozen_field():
    p = params()
    c = waveform(amplitude=0)
    solid, mask, _, _ = T.build_geometry(p)
    f = T.setup_fields(p, solid, mask, c)
    ex, ey = T.compute_field(f["external"], ~solid, p, 0)
    args = T.inject(p, p.n_per_batch, solid.shape[1], np.random.default_rng(1))
    frozen, dynamic = {}, {}
    rf = {**f["rf"], "initial_time": np.zeros(2*p.n_per_batch)}
    T.trace_cpu(*args, ex, ey, ~solid, p, record=frozen)
    T.trace_cpu(*args, ex, ey, ~solid, p, record=dynamic, rf=rf)
    np.testing.assert_array_equal(frozen["status"], dynamic["status"])
    np.testing.assert_allclose(frozen["final_velocity"], dynamic["final_velocity"], atol=1e-7, rtol=1e-12)


def test_cycle_charging_uses_full_wave_and_display_phase_does_not_change_charge(monkeypatch):
    c = waveform()
    calls = []
    def circuit(p, *args, **kw):
        calls.append(kw["include_ied"])
        return c
    monkeypatch.setattr(T.PC, "solve_circuit_cached", circuit)
    p = params()
    a = T.run(p, log=lambda _: None)
    b = T.run(replace(p, rf_phase=90), log=lambda _: None)
    assert calls == [False, False]
    np.testing.assert_array_equal(a["rho_raw"], b["rho_raw"])
    for key in a["hist"]:
        np.testing.assert_array_equal(a["hist"][key], b["hist"][key])
    assert np.max(np.abs(a["phi"]-b["phi"])) > 10
    np.testing.assert_allclose(a["mean_phi"][a["mask"]], -40, atol=1e-11)


def test_saturation_scale_is_charging_potential_not_large_rf_bias():
    p = params(steady_window=30)
    charging = np.arange(60)*.1
    hist = {"wall": charging-1000}
    assert T.check_steady(hist, ["wall"], p)[0]  # 大きな外部電圧を含めると早すぎる判定
    correct = T.check_steady(hist, ["wall"], p, dict(wall=-1000))
    no_bias = T.check_steady({"wall": charging}, ["wall"], p, dict(wall=0))
    assert not correct[0]
    assert correct[2:] == pytest.approx(no_bias[2:])


@pytest.mark.parametrize("change", [dict(bias="dc"), dict(top_bc="neumann"),
                                  dict(rf_time_steps=0), dict(dt_batch=1e-12)])
def test_invalid_cycle_settings_are_rejected(change):
    p = params(**change)
    solid, mask, _, _ = T.build_geometry(p)
    with pytest.raises(ValueError):
        T.setup_fields(p, solid, mask, waveform())


def steady_result(p):
    solid, mask, i0, i1 = T.build_geometry(p)
    z = np.zeros(solid.shape)
    return dict(params=asdict(p), solid=solid, mask=mask, i0=i0, i1=i1, rho_raw=z.copy(),
                t_conv=.001, stopped=False, circuit=waveform())


def test_full_cycle_probe_uses_identical_phases_and_preserves_charge():
    p = params(probe_energy_eV=1.5)
    res = steady_result(p)
    res["rho_raw"][res["i0"]-1, p.floor_t:p.floor_t+p.trench_d] = 3e6
    before = res["rho_raw"].copy()
    d = T.probe_vertical_ions(res, log=lambda _: None)
    assert d["rf_cycle"]
    np.testing.assert_array_equal(d["charged"]["launch_phase"], d["uncharged"]["launch_phase"])
    assert np.max(np.abs(d["charged"]["final_phase"]-d["charged"]["launch_phase"])) > 1
    np.testing.assert_array_equal(res["rho_raw"], before)
    for label in ("charged", "uncharged"):
        assert d[label]["hits"].sum()+d[label]["escaped"]+d[label]["lost"] == p.probe_ions
    fig = T.plot_ion_probe(d, res, p, compact=True)
    assert len(fig.axes) == 6  # GUI: 4 パネル + 電場のカラーバー
    assert len(T.plot_rf_probe_phase(d, p).axes) == 2
    assert len(T.plot_ion_probe(d, res, p).axes) == 8  # 保存: 6 パネル + カラーバー


def test_cuda_rf_matches_analytic_solution_and_cpu_paths():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA GPU がありません")
    from trench_cuda import CudaTracer
    gpu = CudaTracer(np.ones((3, 100), bool))
    cpu_error, cpu = analytic_trajectory(192)
    gpu_error, d = analytic_trajectory(192, gpu)
    assert gpu_error == pytest.approx(cpu_error, abs=1e-12)
    for key in ("endpoints", "final_velocity", "flight_time", "final_phase"):
        np.testing.assert_allclose(d[key], cpu[key], atol=1e-12, rtol=1e-12)
    for a, b in zip(d["paths"], cpu["paths"]):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_real_circuit_cycle_cuda_charging_and_saved_probe(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("CUDA GPU がありません")
    p = params(rf_freq=13.56e6, sigma_s=1e-14, backend="cuda", until_steady=True,
               steady_window=15, steady_hold=2, max_batches=300, probe_energy_eV=1.5)
    res = T.run(p, log=lambda _: None)
    assert np.isfinite(res["t_conv"]) and res["backend_used"] == "cuda"
    cpu = T.run(replace(p, backend="cpu", n_batches=3, until_steady=False), log=lambda _: None)
    gpu = T.run(replace(p, n_batches=3, until_steady=False), log=lambda _: None)
    np.testing.assert_array_equal(cpu["rho_raw"], gpu["rho_raw"])
    np.testing.assert_allclose(cpu["phi"], gpu["phi"], atol=1e-10)
    res["ion_probe"] = T.probe_vertical_ions(res, log=lambda _: None)
    out = str(tmp_path/"rf_cycle")
    T.save_results(res, p, out)
    restored, q = T.load_results(out+".npz")
    assert q.field_mode == "circuit_rf" and restored["ion_probe"]["rf_cycle"]
    np.testing.assert_array_equal(restored["mean_phi"], res["mean_phi"])
    np.testing.assert_array_equal(restored["ion_probe"]["charged"]["launch_phase"],
                                  res["ion_probe"]["charged"]["launch_phase"])
    again = T.probe_vertical_ions(restored, q, log=lambda _: None)
    np.testing.assert_array_equal(again["charged"]["hits"], res["ion_probe"]["charged"]["hits"])
