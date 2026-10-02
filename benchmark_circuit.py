#!/usr/bin/env python3
"""同じ設定・分解能で NumPy / CPU JIT / 波形のみを比較する。キャッシュは使わない。"""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

import plasma_circuit as PC
from trench_charging_2d import Params


def timed(p, **kwargs):
    start = time.perf_counter()
    result = PC.solve_circuit(p, **kwargs)
    return result, time.perf_counter()-start


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="GUI の設定 JSON")
    parser.add_argument("--repeats", type=int, default=2, help="JIT のウォーム実行回数")
    parser.add_argument("--out", type=Path, default=Path("docs/circuit_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats は 1 以上です")
    raw = args.config.read_bytes()
    p = Params(**json.loads(raw.decode("utf-8-sig")))
    if p.ied_model != "sheath":
        parser.error("NumPy/JIT の比較には ied_model=sheath の設定を指定してください")
    print("参照版を測定中 (キャッシュなし・設定 JSON と同じ分解能)…", flush=True)
    reference, ref_seconds = timed(p, sheath_engine="numpy")
    print(f"NumPy: {ref_seconds:.3f} 秒", flush=True)
    wave, wave_seconds = timed(p, include_ied=False)
    first, first_seconds = timed(p, sheath_engine="numba")
    print(f"JIT のプロセス内初回: {first_seconds:.3f} 秒 / 波形のみ: {wave_seconds:.3f} 秒", flush=True)
    times = []
    for _ in range(args.repeats):
        fast, seconds = timed(p, sheath_engine="numba")
        times.append(seconds)
        print(f"JIT: {seconds:.3f} 秒", flush=True)
    wave_error = {key: float(np.max(np.abs(reference[key]-fast[key])))
                  for key in ("V1", "us", "uw", "um", "up", "i")}
    distributions = {}
    for key in ("E", "vn", "vx", "vz", "angle_deg"):
        a, b = (PC.ion_angles(reference), PC.ion_angles(fast)) if key == "angle_deg" else (reference[key], fast[key])
        distributions[key] = dict(reference_mean=float(a.mean()), jit_mean=float(b.mean()),
                                   cdf_max_difference=float(ks_2samp(a, b).statistic),
                                   reference_quantiles=np.quantile(a, [.05, .5, .95]).tolist(),
                                   jit_quantiles=np.quantile(b, [.05, .5, .95]).tolist())
    metadata = lambda c: {k: c["sheath"][k] for k in ("N", "dt", "steps", "ions", "stored", "periods",
                                                   "collisions_per_ion", "cx_fraction", "beta_over")}
    import numba
    report = dict(config=args.config.name, config_sha256=hashlib.sha256(raw).hexdigest(), params=asdict(p),
                  environment=dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                                   platform=platform.platform(), cpu=platform.processor()),
                  cache_used=False, reference_seconds=ref_seconds, jit_process_first_seconds=first_seconds,
                  jit_warm_seconds=times, jit_median_seconds=float(np.median(times)),
                  full_ied_speedup=ref_seconds/np.median(times), waveform_only_seconds=wave_seconds,
                  waveform_only_max_V1_difference=float(np.max(np.abs(wave["V1"]-reference["V1"]))),
                  waveform_max_difference=wave_error, reference_sheath=metadata(reference), jit_sheath=metadata(fast),
                  energy_mean_relative_difference=abs(fast["E"].mean()-reference["E"].mean())/reference["E"].mean(),
                  distributions=distributions,
                  notes="倍精度・粒子数・格子・時間刻み・Newton 許容値は同じ。衝突を含む分布は丸め誤差により乱数消費が変わる場合があり、ビット一致ではなく分布を比較。プロセス内初回は既存 JIT ディスクキャッシュを利用する場合がある。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for c, label, color in ((reference, "NumPy reference", "#64748b"), (fast, "CPU JIT", "#2563eb")):
        axs[0].plot(c["t"]*1e6, c["V1"], label=label, color=color)
        bins = np.linspace(0, max(reference["E"].max(), fast["E"].max())*1.01, 100)
        axs[1].hist(c["E"], bins=bins, density=True, histtype="step", label=label, color=color)
        a = np.sort(PC.ion_angles(c))
        axs[2].plot(a, np.arange(1, len(a)+1)/len(a), label=label, color=color)
    axs[0].set(xlabel="time [us]", ylabel="wafer sheath voltage [V]", title="Circuit waveform")
    axs[1].set(xlabel="ion energy [eV]", ylabel="probability density", title="Ion energy distribution")
    axs[2].set(xlabel="angle from normal [deg]", ylabel="cumulative probability", title="Ion angle distribution", xlim=(0, 5))
    for ax in axs:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"IED 高速化 {report['full_ied_speedup']:.2f} 倍 / 結果: {args.out}", flush=True)


if __name__ == "__main__":
    main()
