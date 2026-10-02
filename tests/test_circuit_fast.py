"""CPU JIT の物理・参照実装との照合と、瞬時電場での不要な IED 計算の防止。"""
import builtins
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from scipy.special import ellipk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plasma_circuit as PC
import sheath_ied as SH
import trench_charging_2d as T


@pytest.fixture(scope="module")
def fast():
    pytest.importorskip("numba")
    import sheath_fast
    return sheath_fast


def test_elliptic_integral_matches_scipy_over_all_scattering_parameters(fast):
    parameters = np.r_[0., np.geomspace(1e-16, .9999, 300)]
    actual = np.array([fast.elliptic_k(x) for x in parameters])
    np.testing.assert_allclose(actual, ellipk(parameters), rtol=2e-15)


@pytest.mark.parametrize("voltage", [0., 100., 5000.])
def test_poisson_solver_reproduces_analytic_quadratic_potential(fast, voltage):
    N, L, Te, ns = 128, .002, 3., 3e16
    dy = L/N
    y = np.linspace(0, L, N+1)
    expected = -voltage*(1-y/L)**2
    ke = SH.E_CHARGE/SH.EPS0
    ni = ns*np.exp(expected/Te)+2*voltage/(ke*L**2)
    phi = expected+.1*np.sin(np.pi*y/L)
    work = [np.empty(N-1) for _ in range(3)]
    fast.solve_phi(phi, ni, Te, ns, ke, dy, -voltage, *work)
    np.testing.assert_allclose(phi, expected, atol=1e-8)
    np.testing.assert_allclose((phi[2:]-2*phi[1:-1]+phi[:-2])/dy**2+ke*(ni[1:-1]-ns*np.exp(phi[1:-1]/Te)),
                               0, atol=.03)


def test_scattering_matches_reference_velocities_and_rng_consumption(fast):
    M = 40*T.AMU
    v = np.random.default_rng(22).normal(0, 20000, (1000, 3))
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    expected, cx, over = SH.nk_scatter(v, r1, M, SH.K_B*300, 12.)
    vx, vy, vz = v.T.copy()
    ncx, nover = fast.scatter(vx, vy, vz, np.arange(len(v)), len(v), r2, M, SH.K_B*300,
                             12., SH.E_CHARGE, SH.NK_BETA0, SH.NK_A)
    np.testing.assert_allclose(np.column_stack([vx, vy, vz]), expected, rtol=1e-11, atol=1e-8)
    assert ncx == cx.sum() and nover == over
    assert r1.bit_generator.state == r2.bit_generator.state


def test_elastic_equal_mass_scattering_conserves_two_particle_energy(fast):
    M = 40*T.AMU
    initial = np.tile([10000., -20000., 3000.], (1000, 1))
    vx, vy, vz = initial.T.copy()
    fast.scatter(vx, vy, vz, np.arange(len(vx)), len(vx), np.random.default_rng(6), M, 0.,
                 12., SH.E_CHARGE, SH.NK_BETA0, 0.)
    after = np.column_stack([vx, vy, vz])
    neutral_after = initial-after
    np.testing.assert_allclose((after**2+neutral_after**2).sum(axis=1), (initial**2).sum(axis=1), rtol=2e-15)


@pytest.mark.parametrize("pressure", [0., 1.])
@pytest.mark.parametrize("wave", ["sine", "pulse"])
def test_complete_pic_retains_counts_fields_energy_and_velocity_distributions(fast, pressure, wave):
    p = replace(T.Params(), rf_wave=wave, rf_freq=2e6, ied_model="instant")
    c = PC.solve_circuit(p)
    kwargs = dict(ion_temp=.5, pressure=pressure, ppc=6, warmup=2e-7, min_collect=2e-7)
    a = SH.ion_energies(c["t"], c["V1"], c["elements"], p.flux, engine="numpy", **kwargs)
    b = SH.ion_energies(c["t"], c["V1"], c["elements"], p.flux, engine="numba", **kwargs)
    assert a[0].size > 0
    assert a[0].shape == b[0].shape
    np.testing.assert_allclose(a[0], b[0], rtol=1e-7, atol=1e-6)
    for key in ("vn", "vx", "vz"):
        np.testing.assert_allclose(a[1][key], b[1][key], rtol=1e-6, atol=1e-4)
    for key in ("N", "dt", "steps", "ions", "periods", "collisions_per_ion", "cx_fraction", "beta_over"):
        assert a[2][key] == b[2][key]
    np.testing.assert_allclose(a[2]["phi"], b[2]["phi"], atol=1e-7)


def test_dc_collisionless_energy_includes_bohm_entry_energy(fast):
    el = PC.circuit_elements(T.Params())
    t = np.arange(64)*1e-6/64
    E, vel, info = SH.ion_energies(t, np.full(64, 100.), el, 1e20, ppc=20, engine="numba")
    assert info["ions"] > 1000
    assert np.isfinite(E).all() and (vel["vn"] > 0).all()
    assert E.mean() == pytest.approx(100.+el["Te"]/2, abs=.5)


def test_auto_falls_back_only_when_numba_is_unavailable(monkeypatch):
    original = builtins.__import__
    def absent(name, *args, **kwargs):
        if name == "sheath_fast":
            raise ImportError("test no numba")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", absent)
    el = PC.circuit_elements(T.Params())
    t = np.arange(64)*1e-7/64
    logs = []
    _, _, info = SH.ion_energies(t, np.full(64, 20.), el, 1e20, ppc=2, warmup=1e-8,
                               min_collect=1e-8, engine="auto", log=logs.append)
    assert info["engine"] == "numpy" and any("NumPy" in s for s in logs)
    with pytest.raises(RuntimeError, match="Numba"):
        SH.ion_energies(t, np.full(64, 20.), el, 1e20, engine="numba")


def test_waveform_cache_cannot_satisfy_a_full_ied_request(monkeypatch, tmp_path):
    monkeypatch.setattr(PC, "CACHE_DIR", str(tmp_path))
    p = replace(T.Params(), ied_model="instant")
    wave = PC.solve_circuit_cached(p, include_ied=False)
    assert not wave["ied_computed"] and wave["E"].size == 0 and not wave["cached"]
    with pytest.raises(ValueError, match="IED"):
        PC.ion_energy_sampler(wave)
    assert PC.solve_circuit_cached(p, include_ied=False)["cached"]
    full = PC.solve_circuit_cached(p)
    assert full["ied_computed"] and full["E"].size == 4096 and not full["cached"]
    np.testing.assert_array_equal(wave["V1"], full["V1"])
    reused = PC.solve_circuit_cached(p, include_ied=False)
    assert reused["cached"] and reused["ied_computed"]
    assert PC.cache_key(p) != PC.cache_key(p, include_ied=False)


def test_fixed_phase_run_skips_sheath_and_can_save_reload_waveform_only_result(monkeypatch, tmp_path):
    def unwanted(*args, **kwargs):
        raise AssertionError("瞬時電場では IED を計算しない")
    monkeypatch.setattr(SH, "ion_energies", unwanted)
    p = replace(T.Params(), field_mode="circuit", side_bc="pillar", rf_freq=400000., rf_volt=5000.,
                rf_wave="pulse", rf_duty=.2, rf_rise=1e-8, nx=20, trench_w=6, trench_d=12,
                floor_t=4, mask_t=4, n_vac=8, n_batches=2, n_per_batch=100, max_steps=800)
    logs = []
    res = T.run(p, use_cache=False, log=logs.append)
    assert not res["circuit"]["ied_computed"]
    assert res["circuit"]["sheath"] is None
    expected = PC.solve_circuit(replace(p, ied_model="instant"))
    assert res["voltages"] == T.circuit_voltages(expected, p.rf_phase)
    files = T.save_results(res, p, str(tmp_path/"wave"))
    assert any(name.endswith("_rf.png") for name in files)
    restored, params = T.load_results(str(tmp_path/"wave.npz"))
    assert params.ied_model == "sheath"
    assert not restored["circuit"]["ied_computed"]
    assert "IED は使用・計算しません" in PC.summary(restored["circuit"])
