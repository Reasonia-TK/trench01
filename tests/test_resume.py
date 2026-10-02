"""再開・旧 NPZ・CPU/CUDA・RF の分割実行を、連続実行と照合する。"""
import copy
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T


def params(**changes):
    base = replace(T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                   mask_t=4, n_vac=8, bias="dc", sigma_s=1e-14,
                   n_batches=12, n_per_batch=128, max_steps=600,
                   probe_ions=128, probe_trajectories=8)
    return replace(base, **changes)


def same_result(actual, expected):
    for key in ("rho_raw", "rho", "phi", "mean_phi", "Ex", "Ey"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for family in ("hist", "charge_hist", "charge_monitor_hist"):
        for key, values in expected[family].items():
            np.testing.assert_array_equal(actual[family][key], values, err_msg=f"{family}/{key}")
    assert actual["checkpoint"] == expected["checkpoint"]
    assert actual["converged"] == expected["converged"]


@pytest.mark.parametrize("mode", ["floating", "fixed", "dielectric"])
def test_cpu_resume_matches_continuous_calculation_and_does_not_mutate_source(mode):
    p = params(field_mode="fixed", side_bc="pillar") if mode == "fixed" else params()
    if mode == "dielectric":
        p = replace(p, mask_type="dielectric")
    full = T.run(p, log=lambda _: None)
    part = T.run(replace(p, n_batches=5), log=lambda _: None)
    before = copy.deepcopy(part)
    progress = []
    resumed = T.run(replace(p, n_batches=7), resume=part, log=lambda _: None, progress=progress.append)
    same_result(resumed, full)
    same_result(part, before)
    assert resumed["start_batch"] == 5
    assert progress[0]["batch"] == 6 and progress[-1]["n_max"] == 12
    assert all(v["start_batch"] == 5 for v in progress)


def circuit():
    t = np.arange(64)*4e-10/64
    s = np.sin(2*np.pi*t/4e-10)
    return dict(t=t, up=t*0+10, uw=-80-35*s+10, um=-40-17*s+10,
                V1=40+17*s, us=-80-35*s+10, i=t*0, E=40+17*s,
                cached=False, ied_computed=True, ied_model="instant",
                sheath=None, n_periods=1, elements={}, s=1e-5, tau_i=1e-9)


@pytest.mark.parametrize("mode", ["energy", "velocity", "circuit", "circuit_rf"])
def test_rf_saved_and_legacy_rng_resume_match_continuous_calculation(monkeypatch, tmp_path, mode):
    p = params(bias="rf", rf_freq=2.5e9, ied_model="instant", sigma_s=0)
    if mode in T.CIRCUIT_FIELDS:
        p = replace(p, field_mode=mode, side_bc="pillar")
    c = circuit()
    if mode == "velocity":
        c.update(vn=np.full(64, 18000.), vx=np.linspace(-500, 500, 64), vz=np.zeros(64))
    calls = []
    monkeypatch.setattr(T.PC, "solve_circuit_cached", lambda *a, **kw: calls.append(1) or c)
    full = T.run(p, log=lambda _: None)
    part = T.run(replace(p, n_batches=5), log=lambda _: None)
    T.save_results(part, p, str(tmp_path / "rf"))
    loaded, saved = T.load_results(tmp_path / "rf.npz")
    assert loaded["checkpoint"] == part["checkpoint"]
    assert len(calls) == 2

    def no_recalculation(*args, **kwargs):
        raise AssertionError("再開時に RF 回路や IED を再計算しました")

    monkeypatch.setattr(T.PC, "solve_circuit_cached", no_recalculation)
    resumed = T.run(replace(saved, n_batches=7), resume=loaded, log=lambda _: None)
    same_result(resumed, full)
    # 同じ旧 NPZ 相当のデータから、乱数状態なしでも入射抽出のみで復元できる。
    legacy = {k: v for k, v in loaded.items() if k != "checkpoint"}
    logs = []
    resumed_legacy = T.run(replace(saved, n_batches=7), resume=legacy, log=logs.append)
    same_result(resumed_legacy, full)
    assert any("乱数状態の復元が完了" in line for line in logs)


@pytest.mark.parametrize("mode", ["fixed", "circuit_rf"])
def test_cuda_resume_matches_continuous_calculation(monkeypatch, mode):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA GPU がありません")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA ドライバを利用できません")
    p = params(backend="cuda", field_mode=mode, side_bc="pillar")
    if mode == "circuit_rf":
        p = replace(p, bias="rf", rf_freq=2.5e9, ied_model="instant")
        monkeypatch.setattr(T.PC, "solve_circuit_cached", lambda *a, **kw: circuit())
    full = T.run(p, log=lambda _: None)
    part = T.run(replace(p, n_batches=5), log=lambda _: None)
    resumed = T.run(replace(p, n_batches=7), resume=part, log=lambda _: None)
    same_result(resumed, full)
    assert resumed["backend_used"] == "cuda"


def plateau_tracer(monkeypatch):
    batches = [0]

    def make(p, vac, log):
        right = T.build_geometry(p)[3]

        def trace(*args, **kwargs):
            batches[0] += 1
            ions, electrons = np.zeros(vac.shape), np.zeros(vac.shape)
            ions[right, p.floor_t+1] = 1
            if batches[0] > 4:
                electrons[right, p.floor_t+1] = 1
            return ions, electrons, np.zeros(2), 0

        return trace, "cpu"

    monkeypatch.setattr(T, "make_particle_tracer", make)
    monkeypatch.setattr(T, "check_steady", lambda *a: (True, "bottom center", 0.0, 1.0))
    return batches


@pytest.mark.parametrize("new_hold, expected_batch", [(2, 20), (3, 30)])
def test_steady_streak_is_preserved_or_reset_when_criterion_changes(monkeypatch, new_hold, expected_batch):
    batches = plateau_tracer(monkeypatch)
    p = params(field_mode="fixed", sigma_s=0, flux=1e18, until_steady=True,
               max_batches=15, steady_window=5, steady_hold=2)
    part = T.run(p, log=lambda _: None)
    assert part["checkpoint"]["steady_n_ok"] == 1
    resumed = T.run(replace(p, max_batches=40, steady_hold=new_hold), resume=part, log=lambda _: None)
    assert resumed["converged"] and len(resumed["hist"]["t"]) == expected_batch
    assert batches[0] == expected_batch
    if new_hold == 2:
        batches[0] = 0
        full = T.run(replace(p, max_batches=40), log=lambda _: None)
        same_result(resumed, full)


def test_legacy_missing_charge_monitors_waits_for_new_history(monkeypatch):
    plateau_tracer(monkeypatch)
    p = params(field_mode="fixed", sigma_s=0, flux=1e18, until_steady=True,
               max_batches=15, steady_window=5, steady_hold=2)
    part = T.run(p, log=lambda _: None)
    legacy = {k: v for k, v in part.items() if k not in ("checkpoint", "charge_monitor_hist")}
    legacy["steady_criterion_version"] = 0
    progress = []
    resumed = T.run(replace(p, max_batches=40), resume=legacy, log=lambda _: None, progress=progress.append)
    assert len(resumed["hist"]["t"]) == 30 and resumed["converged"]
    assert np.isnan(resumed["charge_monitor_hist"]["bottom"][:14]).all()
    assert np.isfinite(resumed["charge_monitor_hist"]["bottom"][14:]).all()
    assert progress[0]["steady_samples_remaining"] == 8
    assert next(v for v in progress if v["batch"] == 20)["check"] is None
    np.testing.assert_array_equal(resumed["hist"]["t"][:15], part["hist"]["t"])


@pytest.mark.parametrize("changes", [dict(trench_offset=1), dict(dt_batch=2e-4), dict(flux=2e20),
                                     dict(rf_volt=200), dict(n_per_batch=256), dict(seed=2)])
def test_resume_rejects_changes_to_physics_and_particles(changes):
    p = params(n_batches=1)
    part = T.run(p, log=lambda _: None)
    with pytest.raises(ValueError, match="保存結果と同じ"):
        T.run(replace(p, **changes), resume=part, log=lambda _: None)


def test_resume_uses_saved_float_values_after_gui_unit_round_trip():
    p = params(n_batches=1)
    part = T.run(p, log=lambda _: None)
    entered = replace(p, dt_batch=100*1e-6, dx=(p.dx*1e9)*1e-9)
    restored = T.resume_params(part, entered)
    assert restored.dt_batch == p.dt_batch and restored.dx == p.dx


def test_resume_rejects_invalid_cap_history_and_checkpoint():
    p = params(n_batches=3)
    part = T.run(p, log=lambda _: None)
    with pytest.raises(ValueError, match="通算"):
        T.run(replace(p, until_steady=True, max_batches=3), resume=part, log=lambda _: None)
    damaged = copy.deepcopy(part)
    damaged["hist"]["t"][-1] += p.dt_batch
    with pytest.raises(ValueError, match="連続した計算履歴"):
        T.run(p, resume=damaged, log=lambda _: None)
    damaged = copy.deepcopy(part)
    damaged["checkpoint"]["batches_completed"] = 2
    with pytest.raises(ValueError, match="バッチ数が一致"):
        T.run(p, resume=damaged, log=lambda _: None)


def test_stop_during_resume_keeps_the_last_batch_and_can_resume_again():
    p = params(n_batches=3)
    part = T.run(p, log=lambda _: None)
    stop = threading.Event()
    interrupted = T.run(p, resume=part, stop=stop,
                        progress=lambda _: stop.set(), log=lambda _: None)
    assert interrupted["stopped"] and len(interrupted["hist"]["t"]) == 4
    resumed = T.run(replace(p, n_batches=8), resume=interrupted, log=lambda _: None)
    full = T.run(replace(p, n_batches=12), log=lambda _: None)
    same_result(resumed, full)


@pytest.mark.parametrize("option, count", [("--n_batches", 5), ("--max_batches", 12)])
def test_cli_resume_inherits_saved_settings_and_adds_only_requested_batches(tmp_path, monkeypatch, option, count):
    p = params(n_batches=3, until_steady=True, max_batches=3, steady_window=5)
    part = T.run(p, log=lambda _: None)
    T.save_results(part, p, str(tmp_path / "source"))
    before = (tmp_path / "source.npz").read_bytes()
    monkeypatch.setattr(sys, "argv", ["trench_charging_2d.py", "--resume_result", str(tmp_path / "source.npz"),
                                      option, "2" if option == "--n_batches" else "12"])
    T.main()
    resumed, saved = T.load_results(tmp_path / "source_continued.npz")
    assert len(resumed["hist"]["t"]) == count and saved.nx == p.nx and saved.sigma_s == p.sigma_s
    assert saved.until_steady == (option == "--max_batches")
    if not saved.until_steady:
        assert saved.n_batches == 2
    assert (tmp_path / "source.npz").read_bytes() == before
    np.testing.assert_array_equal(resumed["hist"]["t"][:3], part["hist"]["t"])
