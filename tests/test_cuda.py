"""CPU/CUDA の同じ入射条件による照合。CUDA がない環境では GPU テストだけスキップする。"""
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T
import trench_cuda as C


def params(**changes):
    return replace(T.Params(), nx=30, trench_w=10, trench_d=20, floor_t=5,
                   mask_t=6, n_vac=12, bias="dc", sigma_s=0,
                   n_batches=8, n_per_batch=400, max_steps=1000,
                   probe_ions=300, probe_trajectories=12, **changes)


@pytest.fixture(scope="module")
def cuda_ready():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA GPU がありません")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA ドライバを利用できません: {exc}")
    # GPU がある場合のコンパイル失敗はスキップしない。
    solid, _, _, _ = T.build_geometry(params())
    return C.CudaTracer(~solid)


def test_auto_falls_back_but_explicit_cuda_reports_error(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("test CUDA unavailable")
    monkeypatch.setattr(C, "CudaTracer", unavailable)
    solid, _, _, _ = T.build_geometry(params())
    logs = []
    tracer, used = T.make_particle_tracer(params(backend="auto"), ~solid, logs.append)
    assert tracer is T.trace_cpu and used == "cpu"
    assert any("CPU" in s and "unavailable" in s for s in logs)
    with pytest.raises(RuntimeError, match="unavailable"):
        T.make_particle_tracer(params(backend="cuda"), ~solid)
    assert T.make_particle_tracer(params(), ~solid) == (T.trace_cpu, "cpu")


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("wall_model", ["absorb", "barrier"])
def test_identical_particles_have_same_hits_and_paths(cuda_ready, wall_model, recording):
    p = params(wall_model=wall_model)
    solid, mask, i0, i1 = T.build_geometry(p)
    fields = T.setup_fields(p, solid, mask)
    rho = np.zeros(solid.shape)
    rho[i0-1, p.floor_t:p.floor_t+p.trench_d] = 2e7
    rho[i1, p.floor_t:p.floor_t+p.trench_d] = -1e7
    phi = fields["solve"]((-rho/T.EPS0).ravel()).reshape(solid.shape)
    ex, ey = T.compute_field(phi, ~solid, p)
    barrier = (phi, np.where(mask, 1.0, fields["eps"]/(1+fields["eps"]))) if wall_model == "barrier" else None
    particles = T.inject(p, p.n_per_batch, solid.shape[1], np.random.default_rng(20))
    records = []
    results = []
    for tracer in (T.trace_cpu, cuda_ready):
        record = {} if recording else None
        # Fortran 配列も CUDA 側で連続配列に変換し、誤ったアドレスを読まない。
        results.append(tracer(*(v.copy() for v in particles), np.asfortranarray(ex), np.asfortranarray(ey),
                              ~solid, p, barrier, record))
        records.append(record)
    for cpu, gpu in zip(*results):
        np.testing.assert_array_equal(cpu, gpu)
    if recording:
        cpu, gpu = records
        for k in ("selected", "status", "hit_cells"):
            np.testing.assert_array_equal(cpu[k], gpu[k])
        for k in ("endpoints", "final_velocity"):
            np.testing.assert_allclose(cpu[k], gpu[k], rtol=1e-10, atol=1e-13)
        for cpu_path, gpu_path in zip(cpu["paths"], gpu["paths"]):
            np.testing.assert_allclose(cpu_path, gpu_path, rtol=1e-10, atol=1e-15)


def test_cuda_cancellation_keeps_initial_paths(cuda_ready):
    p = params()
    solid, _, _, _ = T.build_geometry(p)
    particles = T.inject(p, 20, solid.shape[1], np.random.default_rng(1))
    stop = threading.Event(); stop.set()
    record = {}
    z = np.zeros(solid.shape)
    ci, ce, esc, lost = cuda_ready(*particles, z, z, ~solid, p, record=record, stop=stop)
    assert ci.sum() == ce.sum() == esc.sum() == 0 and lost == 40
    assert record["cancelled"] and all(v.shape == (1, 2) for v in record["paths"])
    np.testing.assert_array_equal(record["endpoints"], np.column_stack(particles[:2]))


def test_empty_particles_do_not_launch_invalid_cuda_grid(cuda_ready):
    p = params()
    solid, _, _, _ = T.build_geometry(p)
    z = np.zeros(solid.shape)
    particles = T.inject(p, 0, solid.shape[1], np.random.default_rng(1))
    record = {}
    ci, ce, esc, lost = cuda_ready(*particles, z, z, ~solid, p, record=record)
    assert ci.sum() == ce.sum() == esc.sum() == lost == 0
    assert record["paths"] == []


@pytest.mark.parametrize("field_mode", ["floating", "fixed"])
def test_cpu_cuda_charging_matches(cuda_ready, field_mode):
    p = params(field_mode=field_mode, substrate_voltage=-20 if field_mode == "fixed" else 0,
               mask_voltage=-10 if field_mode == "fixed" else 0)
    p.sigma_s = 1e-14
    cpu = T.run(p, log=lambda _: None)
    gpu = T.run(replace(p, backend="cuda"), log=lambda _: None)
    assert gpu["backend_used"] == "cuda"
    for key in ("rho_raw", "rho", "phi", "Ex", "Ey"):
        np.testing.assert_allclose(cpu[key], gpu[key], rtol=1e-10, atol=1e-8)
    for k in cpu["hist"]:
        np.testing.assert_allclose(cpu["hist"][k], gpu["hist"][k], rtol=1e-10, atol=1e-10)
    for k in cpu["charge_hist"]:
        np.testing.assert_allclose(cpu["charge_hist"][k], gpu["charge_hist"][k], rtol=1e-10, atol=1e-18)


def test_cuda_saturated_vertical_probe_roundtrip(cuda_ready, tmp_path):
    p = params(backend="cuda", until_steady=True, flux=1e12, max_batches=20,
               steady_window=5, steady_hold=1)
    res = T.run(p, log=lambda _: None)
    assert np.isfinite(res["t_conv"])
    before = res["rho_raw"].copy()
    gpu = T.probe_vertical_ions(res, log=lambda _: None)
    cpu = T.probe_vertical_ions(res, replace(p, backend="cpu"), log=lambda _: None)
    assert gpu["backend_used"] == "cuda" and cpu["backend_used"] == "cpu"
    for label in ("charged", "uncharged"):
        np.testing.assert_array_equal(cpu[label]["hits"], gpu[label]["hits"])
        np.testing.assert_allclose(cpu[label]["angle_deg"], gpu[label]["angle_deg"], atol=1e-9)
        for path in gpu["uncharged"]["paths"]:
            np.testing.assert_array_equal(path[:, 0], np.full(len(path), path[0, 0]))
    np.testing.assert_array_equal(res["rho_raw"], before)
    res["ion_probe"] = gpu
    T.save_results(res, p, str(tmp_path / "gpu"))
    loaded, _ = T.load_results(tmp_path / "gpu.npz")
    assert loaded["backend_used"] == loaded["ion_probe"]["backend_used"] == "cuda"


def test_invalid_cuda_device_is_reported(cuda_ready):
    solid, _, _, _ = T.build_geometry(params())
    with pytest.raises(RuntimeError, match="デバイス"):
        T.make_particle_tracer(params(backend="cuda", cuda_device=999), ~solid)
