"""増加中の電荷による誤収束と、平衡電荷の停止・保存互換性を検証する。"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T


def params(**changes):
    return replace(T.Params(), steady_window=30, **changes)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("age", [10_000, 1_000_000])
def test_constant_charging_is_rejected_even_when_relative_change_is_small(sign, age):
    p = params()
    q = sign * (age + np.arange(2*p.steady_window)) * 1e-12
    # 平均電荷の相対変化だけなら、計算時間の経過で許容内に入ってしまう。
    cur, prev = q[-p.steady_window:].mean(), q[:p.steady_window].mean()
    assert abs(cur-prev) < p.steady_charge_rtol*max(abs(cur), abs(prev))
    check = T.check_charge_steady({"right": q}, p)
    assert not check["ok"] and check["kind"] == "drift"
    assert check["d"] == pytest.approx(sign*p.steady_window*1e-12, rel=1e-8)
    assert check["rate"] == pytest.approx(sign*1e-12/p.dt_batch, rel=1e-8)


@pytest.mark.parametrize("seed", [3, 17])
def test_stationary_charge_with_sampling_noise_is_accepted(seed):
    p = params()
    noise = np.random.default_rng(seed).normal(0, 0.01e-12, 2*p.steady_window)
    check = T.check_charge_steady({"left": 20e-12+noise, "right": -50e-12-noise}, p)
    assert check["ok"]


def test_recent_step_in_charge_fails_the_block_mean_check():
    p = params()
    q = np.r_[np.full(p.steady_window, 20e-12), np.full(p.steady_window, 21e-12)]
    check = T.check_charge_steady({"bottom": q}, p)
    assert not check["ok"] and check["kind"] == "mean"
    assert check["d"] == pytest.approx(1e-12)


def test_opposite_wall_charging_cannot_cancel_in_the_total():
    p = params()
    q = (100+np.arange(2*p.steady_window)*0.05)*1e-12
    check = T.check_charge_steady({"left": q, "right": -q, "surface": q-q}, p)
    assert not check["ok"] and check["name"] in ("left", "right")


def test_surface_charge_magnitude_detects_growth_with_zero_signed_total():
    p = params()
    q = np.arange(2*p.steady_window)*1e-12
    check = T.check_charge_steady({"surface": np.zeros_like(q), "surface absolute": q}, p)
    assert not check["ok"] and check["name"] == "surface absolute"


def test_negligible_drift_uses_the_absolute_charge_tolerance():
    p = params()
    q = np.arange(2*p.steady_window)*(p.steady_charge_atol/(2*p.steady_window))
    assert T.check_charge_steady({"left": q}, p)["ok"]
    assert T.check_charge_steady({"left": np.zeros_like(q)}, replace(p, steady_charge_atol=0))["ok"]


@pytest.mark.parametrize("changes", [dict(steady_charge_rtol=0), dict(steady_charge_rtol=np.nan),
                                     dict(steady_charge_atol=-1), dict(steady_charge_atol=np.inf),
                                     dict(steady_window=4), dict(dt_batch=0)])
def test_invalid_charge_criterion_is_rejected(changes):
    p = replace(params(), **changes)
    with pytest.raises(ValueError):
        T.check_charge_steady({"left": np.zeros(60)}, p)


@pytest.mark.parametrize("mask_type", ["conductor", "dielectric"])
def test_charge_monitors_cover_both_walls_height_bands_floor_and_mask(mask_type):
    p = replace(T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                mask_t=4, n_vac=8, mask_type=mask_type)
    solid, mask, i0, i1 = T.build_geometry(p)
    regions = T.charge_monitors(solid, mask, i0, i1, p)
    rho = np.zeros_like(solid, dtype=float)
    # 左右の総電荷は相殺するが、高さ別の電荷と絶対値は変化する。
    rho[i0-1, p.floor_t+1] = 3/p.dx**2
    rho[i0-1, p.floor_t+p.trench_d-1] = -3/p.dx**2
    rho[i1, p.floor_t+1] = 2/p.dx**2
    rho[i0+1, p.floor_t-1] = 5/p.dx**2
    mask_cell = np.flatnonzero(mask & T.surface_cells(solid))[0]
    rho.flat[mask_cell] = 7/p.dx**2
    q = T.sample_charge_monitors(rho, regions, p.dx)
    assert q["left"] == pytest.approx(0)
    assert q["left band 1"] == pytest.approx(3)
    assert q["left band 3"] == pytest.approx(-3)
    assert q["right"] == pytest.approx(2)
    assert q["bottom"] == pytest.approx(5)
    assert q["mask"] == pytest.approx(7)
    assert q["surface"] == pytest.approx(7 if mask_type == "conductor" else 14)
    assert q["surface absolute"] == pytest.approx(13 if mask_type == "conductor" else 20)


def controlled_run(monkeypatch, charge_batches=None):
    p = replace(T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                mask_t=4, n_vac=8, bias="dc", field_mode="fixed", backend="cpu",
                sigma_s=0, flux=1e18, n_per_batch=128, until_steady=True,
                steady_window=5, steady_hold=2, max_batches=40)
    batches, progress, logs = [0], [], []

    def make_tracer(_p, vac, log):
        right = T.build_geometry(_p)[3]

        def trace(*args, **kwargs):
            batches[0] += 1
            ions = np.zeros_like(vac, dtype=float)
            ions[right, _p.floor_t+1] = 1
            electrons = np.zeros_like(ions)
            if charge_batches is not None and batches[0] > charge_batches:
                electrons[right, _p.floor_t+1] = 1
            return ions, electrons, np.zeros(2), 0

        return trace, "cpu"

    monkeypatch.setattr(T, "make_particle_tracer", make_tracer)
    # 電位が先に許容内に入った条件を、粒子の電荷収支と独立に再現する。
    monkeypatch.setattr(T, "check_steady", lambda *args: (True, "bottom center", 0.0, 1.0))
    res = T.run(p, progress=progress.append, log=logs.append)
    return res, p, progress, logs


def test_run_continues_when_voltage_is_stable_but_charge_keeps_growing(monkeypatch):
    res, p, progress, logs = controlled_run(monkeypatch)
    assert len(res["hist"]["t"]) == p.max_batches
    assert not res["converged"] and np.isnan(res["t_conv"])
    check = progress[-1]["check"]
    assert check["voltage"]["ok"] and not check["charge"]["ok"]
    assert check["unit"] == "C/m" and check["n_ok"] == 0
    assert any("電位 OK" in line and "電荷 NG" in line and "pC/m" in line for line in logs)


def test_run_stops_only_after_charge_plateau_passes_the_required_blocks(monkeypatch):
    res, p, progress, _ = controlled_run(monkeypatch, charge_batches=4)
    assert res["converged"] and len(res["hist"]["t"]) == 20
    assert res["t_conv"] == pytest.approx(20*p.dt_batch)
    checks = [r["check"] for r in progress if r["check"] is not None]
    assert [c["n_ok"] for c in checks] == [0, 1, 2]
    assert all(c["voltage"]["ok"] for c in checks)


def test_charge_monitors_and_valid_saturation_round_trip(monkeypatch, tmp_path):
    res, p, _, _ = controlled_run(monkeypatch, charge_batches=4)
    T.save_results(res, p, str(tmp_path / "plateau"))
    loaded, saved = T.load_results(tmp_path / "plateau.npz")
    assert loaded["converged"] and loaded["t_conv"] == res["t_conv"]
    assert saved.steady_charge_atol == p.steady_charge_atol
    assert loaded["steady_criterion_version"] == T.STEADY_CRITERION_VERSION
    for key, q in res["charge_monitor_hist"].items():
        np.testing.assert_array_equal(loaded["charge_monitor_hist"][key], q)


def test_legacy_voltage_only_saturation_is_rechecked_without_changing_saved_data(monkeypatch, tmp_path):
    res, p, _, _ = controlled_run(monkeypatch)
    res["t_conv"], res["converged"] = res["hist"]["t"][-1], True
    T.save_results(res, p, str(tmp_path / "legacy"))
    # 旧 NPZ と同じく、新しい電荷観測履歴・判定メタデータ・設定を持たない。
    import json
    filename = tmp_path / "legacy.npz"
    with np.load(filename, allow_pickle=False) as d:
        old = {k: d[k].copy() for k in d.files
               if not k.startswith("qmonitor_") and k != "convergence_meta_json"}
    settings = json.loads(str(old["params_json"]))
    settings.pop("steady_charge_rtol")
    settings.pop("steady_charge_atol")
    old["params_json"] = np.array(json.dumps(settings))
    np.savez(filename, **old)
    before = filename.read_bytes()
    loaded, _ = T.load_results(filename)
    assert not loaded["converged"] and np.isnan(loaded["t_conv"])
    assert loaded["previous_t_conv"] == res["t_conv"] and loaded["convergence_note"]
    np.testing.assert_array_equal(loaded["rho_raw"], res["rho_raw"])
    np.testing.assert_array_equal(loaded["charge_hist"]["right"], res["charge_hist"]["right"])
    assert filename.read_bytes() == before
    with pytest.raises(ValueError, match="飽和判定"):
        T.probe_vertical_ions(loaded, p)
