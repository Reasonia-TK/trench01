"""新しい境界条件と、電荷を更新しない飽和後診断の物理・保存テスト。"""
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T


def small_params(**changes):
    return replace(T.Params(), nx=20, trench_w=6, trench_d=12, floor_t=4,
                   mask_t=4, n_vac=8, bias="dc", sigma_s=0, n_batches=3,
                   n_per_batch=200, max_steps=800, probe_ions=120,
                   probe_trajectories=8, **changes)


@pytest.mark.parametrize("offset", [-6, 0, 6])
def test_shifted_opening_and_masks(offset):
    p = small_params(trench_offset=offset)
    solid, mask, i0, i1 = T.build_geometry(p)
    assert i0 == 7 + offset and i1 == 13 + offset
    assert not solid[i0:i1, p.floor_t:].any()
    assert solid[i0 - 1, :20].all() and solid[i1, :20].all()
    assert np.array_equal(mask, solid & (np.arange(solid.shape[1])[None, :] >= 16))


@pytest.mark.parametrize("offset", [-7, 7, 0.5])
def test_shift_rejects_missing_wall(offset):
    with pytest.raises(ValueError):
        T.build_geometry(small_params(trench_offset=offset))


def test_nonzero_dirichlet_is_linear_in_homogeneous_medium():
    p = small_params(field_mode="fixed", substrate_voltage=-12, top_voltage=8)
    solid = np.zeros((p.nx, 25), dtype=bool)
    fields = T.setup_fields(p, solid, np.zeros_like(solid))
    y = (np.arange(25) + 0.5) / 25
    expected = p.substrate_voltage + (p.top_voltage - p.substrate_voltage) * y
    np.testing.assert_allclose(fields["external"], np.broadcast_to(expected, solid.shape), atol=1e-11)
    ex, ey = T.compute_field(fields["external"], ~solid, p)
    np.testing.assert_allclose(ex, 0, atol=1e-4)
    np.testing.assert_allclose(ey, -(p.top_voltage-p.substrate_voltage)/(25*p.dx), rtol=1e-12)


def test_fixed_mask_stays_fixed_with_charge_and_leakage():
    p = small_params(field_mode="fixed", substrate_voltage=-30, mask_voltage=-10)
    p.sigma_s = 1e-14
    res = T.run(p, log=lambda _: None)
    np.testing.assert_allclose(res["phi"][res["mask"]], -10, atol=1e-10)
    assert np.all(res["rho_raw"][res["mask"]] == 0)
    assert np.any(np.abs(res["rho"][res["mask"]]) > 0)  # 誘導電荷は表示・保存する
    assert np.any(np.abs(res["rho_raw"][~res["mask"]]) > 0)
    for side, ix in (("left", res["i0"]-1), ("right", res["i1"])):
        assert len(res["charge_hist"][side]) == p.n_batches
        np.testing.assert_allclose(res["charge_hist"][side][-1],
                                   res["rho_raw"][ix, p.floor_t:p.floor_t+p.trench_d].sum()*p.dx**2)


def test_pillar_boundary_reproduces_linear_dielectric_voltage():
    p = small_params(field_mode="fixed", side_bc="pillar", substrate_voltage=-20, mask_voltage=10)
    # SiO2 の平板と等電位の上部電極なら、ピラーの線形電位と内部電位が一致する。
    ny = p.floor_t + p.trench_d + p.mask_t + p.n_vac
    solid = np.zeros((p.nx, ny), dtype=bool)
    top = p.floor_t + p.trench_d
    solid[:, :top+p.mask_t] = True
    mask = np.zeros_like(solid); mask[:, top:top+p.mask_t] = True
    fields = T.setup_fields(p, solid, mask)
    expected = -20 + 30*(np.arange(top)+0.5)/top
    np.testing.assert_allclose(fields["external"][:, :top], np.broadcast_to(expected, (p.nx, top)), atol=1e-10)


def test_left_and_right_fixed_boundaries_are_distinct():
    p = small_params(field_mode="fixed", side_bc="fixed", left_voltage=20, right_voltage=-20)
    solid, mask, _, _ = T.build_geometry(p)
    f = T.setup_fields(p, solid, mask)
    assert f["external"][0, 8] > 0
    assert f["external"][-1, 8] < 0
    np.testing.assert_allclose(f["external"], -f["external"][::-1], atol=1e-10)
    ny = solid.shape[1]
    assert f["A"][(p.nx-1)*ny+8, 8] == 0  # SiO2 両端は周期結合を持たない
    assert f["A"][(p.nx-1)*ny+ny-1, ny-1] != 0  # 上部真空は周期境界を維持


def fake_circuit():
    return dict(t=np.arange(4)*1e-8, uw=np.array([-80., -100., -80., -60.]),
                um=np.array([-40., -50., -40., -30.]), up=np.full(4, 10.),
                V1=np.array([50., 60., 50., 40.]), E=np.full(4, 60.),
                ied_model="instant", cached=False)


def test_circuit_phase_is_periodic_and_gauge_invariant():
    c = fake_circuit()
    v = T.circuit_voltages(c, 45)
    assert v == dict(substrate=-100, mask=-55, top=0, sheath=55)
    assert T.circuit_voltages(c, 405) == v
    for k in ("uw", "um", "up"):
        c[k] = c[k] + 123
    assert T.circuit_voltages(c, 45) == v


def test_circuit_mode_applies_phase_and_avoids_double_acceleration(monkeypatch):
    p = small_params(field_mode="circuit", side_bc="pillar", rf_phase=90)
    p.bias = "rf"
    monkeypatch.setattr(T.PC, "solve_circuit_cached", lambda *args, **kwargs: fake_circuit())
    monkeypatch.setattr(T.PC, "summary", lambda c: "test circuit")
    original = T.inject
    energies = []

    def capture(p, n, ny, rng, ion_energy, ion_velocity):
        energies.append(ion_energy(rng, n))
        assert ion_velocity is None
        return original(p, n, ny, rng, ion_energy, ion_velocity)

    monkeypatch.setattr(T, "inject", capture)
    res = T.run(p, log=lambda _: None)
    np.testing.assert_allclose(res["phi"][res["mask"]], -60)
    assert res["voltages"]["substrate"] == -110
    assert all(np.all(e == p.electron_temp_eV/2) for e in energies)


def empty_steady_result(p):
    solid, mask, i0, i1 = T.build_geometry(p)
    z = np.zeros(solid.shape)
    return dict(params=asdict(p), solid=solid, mask=mask, i0=i0, i1=i1, rho_raw=z.copy(),
                phi=z.copy(), rho=z.copy(), Ex=z.copy(), Ey=z.copy(), external_phi=z.copy(),
                t_conv=0.001, stopped=False, circuit=None)


def test_vertical_probe_is_exactly_vertical_and_does_not_charge():
    p = small_params()
    res = empty_steady_result(p)
    before = res["rho_raw"].copy()
    probe = T.probe_vertical_ions(res, log=lambda _: None)
    np.testing.assert_array_equal(before, res["rho_raw"])
    for label in ("charged", "uncharged"):
        d = probe[label]
        assert d["bottom_fraction"] == 1 and d["lost"] == 0 and d["escaped"] == 0
        assert len(d["paths"]) == p.probe_trajectories
        for points in d["paths"]:
            np.testing.assert_array_equal(points[:, 0], np.full(len(points), points[0, 0]))
            np.testing.assert_allclose(points[-1, 1], p.floor_t*p.dx, atol=1e-18)
        np.testing.assert_array_equal(d["angle_deg"], np.zeros(p.probe_ions))
        np.testing.assert_allclose(d["bottom_flux"], p.flux)


def test_charge_deflects_vertical_ions_and_preserves_particle_accounting():
    p = small_params()
    res = empty_steady_result(p)
    res["rho_raw"][res["i0"]-1, p.floor_t:p.floor_t+p.trench_d] = 1e7
    before = res["rho_raw"].copy()
    probe = T.probe_vertical_ions(res, log=lambda _: None)
    np.testing.assert_array_equal(res["rho_raw"], before)
    charged = probe["charged"]
    assert np.any(np.abs(charged["angle_deg"]) > 0.1)
    assert charged["hits"].sum() + charged["escaped"] + charged["lost"] == p.probe_ions
    assert probe["uncharged"]["bottom_fraction"] == 1


def test_probe_requires_convergence_and_supports_cancellation():
    p = small_params()
    res = empty_steady_result(p)
    res["t_conv"] = np.nan
    with pytest.raises(ValueError, match="飽和"):
        T.probe_vertical_ions(res, log=lambda _: None)
    res["t_conv"] = .001
    stop = threading.Event(); stop.set()
    probe = T.probe_vertical_ions(res, log=lambda _: None, stop=stop)
    assert probe["cancelled"]


def test_probe_phase_changes_external_field_without_changing_stored_charge():
    p = small_params(field_mode="circuit", side_bc="pillar")
    p.bias = "rf"
    res = empty_steady_result(p)
    res["circuit"] = fake_circuit()
    original = res["rho_raw"].copy()
    probe = T.probe_vertical_ions(res, phase=90, log=lambda _: None)
    assert probe["phase"] == 90
    np.testing.assert_allclose(probe["charged"]["phi"][res["mask"]], -60)
    np.testing.assert_array_equal(res["rho_raw"], original)
    assert res["params"]["rf_phase"] == 0


def test_driven_leakage_responds_to_voltage_without_particle_charge():
    p = small_params(field_mode="fixed", side_bc="pillar", substrate_voltage=-20, mask_voltage=-10)
    p.flux = 1e-10
    p.n_batches = 1
    p.sigma_s = 1e-14
    res = T.run(p, log=lambda _: None)
    assert np.max(np.abs(res["rho_raw"])) > 1e6  # 指定電位の表面電流で電荷が移動
    np.testing.assert_allclose(res["phi"][res["mask"]], -10)


def test_real_steady_result_roundtrip_and_repeated_probe(tmp_path):
    p = small_params(until_steady=True, max_batches=20, steady_window=5, steady_hold=1, flux=1e12)
    res = T.run(p, log=lambda _: None)
    assert np.isfinite(res["t_conv"])
    res["ion_probe"] = T.probe_vertical_ions(res, log=lambda _: None)
    files = T.save_results(res, p, str(tmp_path / "steady"))
    assert all(Path(f).stat().st_size > 0 for f in files)
    saved, q = T.load_results(tmp_path / "steady.npz")
    assert q == p
    again = T.probe_vertical_ions(saved, log=lambda _: None)
    for label in ("charged", "uncharged"):
        np.testing.assert_array_equal(again[label]["hits"], res["ion_probe"][label]["hits"])
        for orig, loaded in zip(res["ion_probe"][label]["paths"], saved["ion_probe"][label]["paths"]):
            np.testing.assert_array_equal(orig, loaded)
    with np.load(tmp_path / "steady.npz", allow_pickle=False) as d:
        assert all(d[k].dtype != object for k in d.files)


def test_extreme_shift_and_shallow_trench_have_valid_observations():
    p = small_params(trench_offset=-6)
    p.trench_d = 1
    res = T.run(p, log=lambda _: None)
    for ix, iy in res["probes"].values():
        assert 0 <= ix < p.nx and p.floor_t-1 <= iy < res["solid"].shape[1]
        assert res["solid"][ix, iy]
