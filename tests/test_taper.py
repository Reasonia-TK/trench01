"""テーパーの形状、段差の電荷・リーク、ビーム透過、再開と CUDA を検証する。"""
import copy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trench_charging_2d as T

TR_ANGLE = np.degrees(np.arctan(.25))
MASK_ANGLE = np.degrees(np.arctan(.5))


def params(**changes):
    base = replace(T.Params(), nx=40, trench_w=16, trench_d=20, floor_t=4,
                   mask_t=12, n_vac=10, trench_taper_deg=TR_ANGLE, mask_taper_deg=MASK_ANGLE,
                   bias="dc", sigma_s=0, n_batches=3, n_per_batch=256,
                   max_steps=800, probe_ions=224, probe_trajectories=12)
    return replace(base, **changes)


def empty_result(p):
    solid, mask, i0, i1 = T.build_geometry(p)
    z = np.zeros(solid.shape)
    return dict(params=asdict(p), solid=solid, mask=mask, i0=i0, i1=i1,
                rho_raw=z, t_conv=.001, stopped=False, circuit=None)


@pytest.mark.parametrize("mask_t,offset", [(0, -4), (12, 0), (12, 4)])
def test_zero_angles_preserve_original_rectangular_geometry_and_monitors(mask_t, offset):
    p = params(trench_taper_deg=0, mask_taper_deg=0, mask_t=mask_t, trench_offset=offset)
    solid, mask, i0, i1 = T.build_geometry(p)
    expected = np.zeros_like(solid)
    expected[:, :p.floor_t+p.trench_d+p.mask_t] = True
    expected[i0:i1, p.floor_t:] = False
    np.testing.assert_array_equal(solid, expected)
    np.testing.assert_array_equal(mask, expected & (np.arange(solid.shape[1]) >= p.floor_t+p.trench_d))
    regions = T.charge_monitors(solid, mask, i0, i1, p)
    for side, col in (("left", i0-1), ("right", i1)):
        x, y = np.unravel_index(regions[side], solid.shape)
        np.testing.assert_array_equal(x, np.full(p.trench_d, col))
        np.testing.assert_array_equal(y, np.arange(p.floor_t, p.floor_t+p.trench_d))
        px, py = T.wall_profiles(solid, p)[side]
        np.testing.assert_array_equal(px, np.full(p.trench_d+p.mask_t, col))
        np.testing.assert_array_equal(py, np.arange(p.floor_t, p.floor_t+p.trench_d+p.mask_t))


def test_independent_tapers_have_expected_bottom_interface_and_entrance():
    p = params()
    solid, mask, i0, i1 = T.build_geometry(p)
    assert (i0, i1) == (12, 28)
    assert T.opening_widths(p) == pytest.approx(dict(bottom=6, interface=16, top=28))
    assert T.opening_span(solid, p.floor_t) == (17, 23)
    assert T.opening_span(solid, p.floor_t+p.trench_d-1) == (12, 28)
    assert T.opening_span(solid, p.floor_t+p.trench_d) == (12, 28)
    assert T.opening_span(solid, p.floor_t+p.trench_d+p.mask_t-1) == (6, 34)
    np.testing.assert_array_equal(solid, solid[::-1])
    np.testing.assert_array_equal(mask, mask[::-1])
    widths = (~solid[:, p.floor_t:p.floor_t+p.trench_d+p.mask_t]).sum(axis=0)
    assert (np.diff(widths) >= 0).all()
    only_trench = T.build_geometry(replace(p, mask_taper_deg=0))[0]
    only_mask = T.build_geometry(replace(p, trench_taper_deg=0))[0]
    assert T.opening_span(only_trench, 35) == (12, 28)
    assert T.opening_span(only_mask, 4) == (12, 28)


def test_shift_translates_both_tapers_without_changing_width_or_connectivity():
    solid, mask, _, _ = T.build_geometry(params())
    shifted, shifted_mask, i0, i1 = T.build_geometry(params(trench_offset=3))
    assert (i0, i1) == (15, 31)
    np.testing.assert_array_equal(shifted, np.roll(solid, 3, axis=0))
    np.testing.assert_array_equal(shifted_mask, np.roll(mask, 3, axis=0))
    from scipy.ndimage import label
    inner = ~shifted[:, 4:36]
    assert label(inner)[1] == 1


@pytest.mark.parametrize("changes", [dict(trench_taper_deg=25), dict(mask_taper_deg=-40),
    dict(mask_taper_deg=50), dict(trench_taper_deg=-40), dict(trench_offset=8),
    dict(trench_taper_deg=90), dict(mask_taper_deg=-90),
    dict(trench_taper_deg=np.nan), dict(mask_taper_deg=np.inf)])
def test_closed_opening_outside_domain_and_invalid_angles_are_rejected(changes):
    with pytest.raises(ValueError):
        T.build_geometry(params(**changes))


def test_charge_monitors_include_horizontal_steps_and_partition_sidewalls():
    p = params()
    solid, mask, i0, i1 = T.build_geometry(p)
    regions = T.charge_monitors(solid, mask, i0, i1, p)
    left = np.unravel_index(regions["left"], solid.shape)
    assert len(np.unique(left[0])) > 1
    # 垂直に入射して段差へ衝突した全イオンが、SiO2 左右壁・底面かマスクで集計される。
    probe = T.probe_vertical_ions(empty_result(p), log=lambda _: None)["uncharged"]
    hits = probe["hits"].ravel()
    assert (hits[regions["left"]].sum()+hits[regions["right"]].sum()
            +hits[regions["bottom"]].sum()+probe["hits"][mask].sum()) == p.probe_ions
    for side in ("left", "right"):
        bands = [regions[f"{side} band {band}"] for band in (1, 2, 3)]
        np.testing.assert_array_equal(np.sort(np.concatenate(bands)), regions[side])
    charges = T.sample_charge_monitors(probe["hits"]/p.dx**2, regions, p.dx)
    assert charges["left"] == pytest.approx(40)
    assert charges["right"] == pytest.approx(40)
    assert charges["bottom"] == pytest.approx(48)


def test_surface_leakage_remains_connected_and_conserves_charge_on_steps():
    p = params(sigma_s=1e-14, mask_type="dielectric")
    solid, mask, i0, i1 = T.build_geometry(p)
    fields = T.setup_fields(p, solid, mask)
    leak = T.build_leakage(solid, fields["solve"], i0, i1, p)
    from scipy.sparse.csgraph import connected_components
    assert connected_components(leak["G"], directed=False)[0] == 1
    np.testing.assert_allclose(leak["G"].sum(axis=0), 0, atol=1e-20)
    q = np.zeros(len(leak["flat"]))
    b0, b1 = T.opening_span(solid, p.floor_t)
    source = np.flatnonzero(leak["flat"] == ((b0+b1)//2)*solid.shape[1]+p.floor_t-1)[0]
    q[source] = 1e-12
    after = leak["M"] @ q
    assert after.sum() == pytest.approx(q.sum(), rel=1e-12)
    assert np.count_nonzero(np.abs(after) > 1e-18) > 2
    x, y = np.unravel_index(leak["flat"][leak["region"]], solid.shape)
    assert x.min() < i0 or x.max() >= i1 or np.unique(x).size > 2
    assert (y < p.floor_t+p.trench_d//2).all()


@pytest.mark.parametrize("trench_angle,mask_angle,mask_t,bottom_width,entrance_width,fraction", [
    (TR_ANGLE, MASK_ANGLE, 12, 6, 28, 6/28),
    (-TR_ANGLE, MASK_ANGLE, 12, 26, 28, 16/28),
    (TR_ANGLE, -MASK_ANGLE, 12, 6, 4, 1),
    (TR_ANGLE, 0, 0, 6, 16, 6/16)])
def test_vertical_beam_matches_ballistic_aperture_and_flux_conservation(
        trench_angle, mask_angle, mask_t, bottom_width, entrance_width, fraction):
    p = params(trench_taper_deg=trench_angle, mask_taper_deg=mask_angle, mask_t=mask_t,
               probe_ions=entrance_width*8)
    res = empty_result(p)
    before = res["rho_raw"].copy()
    probe = T.probe_vertical_ions(res, log=lambda _: None)
    np.testing.assert_array_equal(res["rho_raw"], before)
    for label in ("charged", "uncharged"):
        record = probe[label]
        assert record["bottom_fraction"] == fraction
        assert record["escaped"] == record["lost"] == 0
        assert record["hits"].sum() == p.probe_ions
        assert len(record["bottom_flux"]) == bottom_width
        assert record["bottom_flux"].sum()/p.flux == pytest.approx(entrance_width*fraction)
        for path in record["paths"]:
            np.testing.assert_array_equal(path[:, 0], np.full(len(path), path[0, 0]))
        np.testing.assert_array_equal(record["angle_deg"], np.zeros(p.probe_ions))
    fig = T.plot_ion_probe(probe, res, p)
    assert len(fig.axes[2].lines[0].get_xdata()) == bottom_width


@pytest.mark.parametrize("mode", ["fixed", "floating", "dielectric"])
def test_tapered_charge_run_save_load_and_resume_match_continuous(tmp_path, mode):
    p = params(sigma_s=1e-14)
    if mode == "fixed":
        p = replace(p, field_mode="fixed", side_bc="pillar", substrate_voltage=-20, mask_voltage=-10)
    elif mode == "dielectric":
        p = replace(p, mask_type="dielectric")
    full = T.run(p, log=lambda _: None)
    part = T.run(replace(p, n_batches=1), log=lambda _: None)
    T.save_results(part, p, str(tmp_path/"part"))
    saved, saved_p = T.load_results(tmp_path/"part.npz")
    assert saved_p.trench_taper_deg == p.trench_taper_deg and saved_p.mask_taper_deg == p.mask_taper_deg
    continued = T.run(replace(p, n_batches=2), resume=saved, log=lambda _: None)
    for key in ("rho_raw", "rho", "phi", "Ex", "Ey"):
        np.testing.assert_array_equal(continued[key], full[key])
    for family in ("hist", "charge_hist", "charge_monitor_hist"):
        for key in full[family]:
            np.testing.assert_array_equal(continued[family][key], full[family][key])
    regions = T.charge_monitors(full["solid"], full["mask"], full["i0"], full["i1"], p)
    charges = T.sample_charge_monitors(full["rho_raw"], regions, p.dx)
    assert full["charge_hist"]["left"][-1] == charges["left"]
    assert full["charge_hist"]["right"][-1] == charges["right"]
    for key in ("trench_taper_deg", "mask_taper_deg"):
        with pytest.raises(ValueError, match=key):
            T.run(replace(p, **{key: 0}), resume=saved, log=lambda _: None)
    if mode == "fixed":
        np.testing.assert_allclose(full["phi"][full["mask"]], p.mask_voltage, atol=1e-10)
    T.plot_results(full, p, tmp_path/"taper", pyplot=False)
    assert (tmp_path/"taper_fields.png").is_file()


def test_probe_rejects_different_taper():
    p = params()
    for key in ("trench_taper_deg", "mask_taper_deg"):
        with pytest.raises(ValueError, match="同じ設定"):
            T.probe_vertical_ions(empty_result(p), replace(p, **{key: 0}))


@pytest.mark.parametrize("reflect,recording", [(False, False), (False, True), (True, True)])
def test_cpu_catches_the_first_wall_face_when_both_end_cells_are_vacuum(reflect, recording):
    _assert_corner_contact(T.trace_cpu, reflect, recording)


def _assert_corner_contact(tracer, reflect, recording):
    p = params(max_steps=1)
    solid, _, _, _ = T.build_geometry(p)
    vac = ~solid
    corners = solid[:, :-1] & vac[:, 1:] & np.roll(vac, -1, axis=0)[:, :-1]
    cx, cy = np.argwhere(corners)[0]
    # 上側の真空 → 固体の角 → 右側の真空。終点セルだけでは壁を通過してしまう。
    x, y = np.array([(cx+.92)*p.dx]), np.array([(cy+1.02)*p.dx])
    phi = np.zeros(solid.shape)
    phi[cx, cy] = 1e6
    barrier = (phi, np.ones_like(phi)) if reflect else None
    record = {} if recording else None
    ci, _, _, lost = tracer(x, y, np.array([1000.]), np.array([-1000.]),
                            np.ones(1), np.zeros(1, dtype=np.int8),
                            np.zeros_like(phi), np.zeros_like(phi), vac, p, barrier, record=record)
    if reflect:
        assert lost == 1 and ci.sum() == 0
        np.testing.assert_array_equal(record["final_velocity"], [[1000., 1000.]])
        np.testing.assert_allclose(record["endpoints"], [[(cx+.92)*p.dx, (cy+1.02)*p.dx]], atol=1e-20)
    else:
        assert lost == 0 and ci[cx, cy] == 1
        if recording:
            assert record["hit_cells"].tolist() == [[cx, cy]]
            assert record["endpoints"][0, 1] == pytest.approx((cy+1)*p.dx, abs=1e-20)


@pytest.fixture(scope="module")
def cuda_ready():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA GPU がありません")
    except cp.cuda.runtime.CUDARuntimeError as e:
        pytest.skip(str(e))
    return True


@pytest.mark.parametrize("reflect,recording", [(False, False), (False, True), (True, True)])
def test_cuda_catches_the_first_wall_face_when_both_end_cells_are_vacuum(cuda_ready, reflect, recording):
    import trench_cuda as C
    solid = T.build_geometry(params())[0]
    _assert_corner_contact(C.CudaTracer(~solid), reflect, recording)


@pytest.mark.parametrize("wall_model", ["absorb", "barrier"])
def test_cuda_traces_tapered_walls_and_steps_identically_to_cpu(cuda_ready, wall_model):
    import trench_cuda as C
    p = params(wall_model=wall_model, field_mode="fixed", substrate_voltage=-30, mask_voltage=-10)
    solid, mask, i0, i1 = T.build_geometry(p)
    fields = T.setup_fields(p, solid, mask)
    rho = np.zeros(solid.shape)
    regions = T.charge_monitors(solid, mask, i0, i1, p)
    rho.ravel()[regions["left"]] = 2e6
    rho.ravel()[regions["right"]] = -1e6
    phi = fields["solve"]((-rho/T.EPS0).ravel()).reshape(solid.shape)+fields["external"]
    ex, ey = T.compute_field(phi, ~solid, p)
    particles = T.inject(p, p.n_per_batch, solid.shape[1], np.random.default_rng(22))
    barrier = (phi, np.where(mask, 1, fields["eps"]/(1+fields["eps"]))) if wall_model == "barrier" else None
    results, records = [], []
    for tracer in (T.trace_cpu, C.CudaTracer(~solid)):
        record = {}
        results.append(tracer(*(a.copy() for a in particles), ex, ey, ~solid, p, barrier, record=record))
        records.append(record)
    for actual, expected in zip(results[1], results[0]):
        np.testing.assert_array_equal(actual, expected)
    for key in ("status", "hit_cells"):
        np.testing.assert_array_equal(records[1][key], records[0][key])
    for actual, expected in zip(records[1]["paths"], records[0]["paths"]):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


def test_rf_cycle_with_taper_and_cuda_resume(monkeypatch, cuda_ready, tmp_path):
    p = params(bias="rf", field_mode="circuit_rf", side_bc="pillar", rf_freq=5e8,
               sigma_s=1e-14, n_batches=3)
    t = np.arange(256)/256/p.rf_freq
    c = dict(t=t, uw=-60-20*np.sin(2*np.pi*t*p.rf_freq), um=-30-10*np.sin(2*np.pi*t*p.rf_freq),
             up=t*0, V1=30+10*np.sin(2*np.pi*t*p.rf_freq), E=np.array([]),
             us=-60-20*np.sin(2*np.pi*t*p.rf_freq), i=t*0,
             cached=False, ied_model="instant", sheath=None, n_periods=1, elements={},
             ied_computed=False, s=1e-5, tau_i=1e-9)
    monkeypatch.setattr(T.PC, "solve_circuit_cached", lambda *args, **kw: copy.deepcopy(c))
    monkeypatch.setattr(T.PC, "summary", lambda c: "test RF circuit")
    cpu = T.run(p, log=lambda _: None)
    gpu_p = replace(p, backend="cuda")
    part = T.run(replace(gpu_p, n_batches=1), log=lambda _: None)
    T.save_results(part, gpu_p, str(tmp_path/"rf_taper"))
    saved, _ = T.load_results(tmp_path/"rf_taper.npz")
    gpu = T.run(replace(gpu_p, n_batches=2), resume=saved, log=lambda _: None)
    for key in ("rho_raw", "phi", "mean_phi", "Ex", "Ey"):
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=1e-11, atol=1e-9)
    for key in cpu["charge_hist"]:
        np.testing.assert_array_equal(gpu["charge_hist"][key], cpu["charge_hist"][key])
    assert gpu["backend_used"] == "cuda" and gpu["checkpoint"]["phase_rng_state"] is not None
