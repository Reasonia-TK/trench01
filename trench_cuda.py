"""任意依存 CuPy を使う CUDA 粒子追跡。通常の CPU 実行では CuPy を import しない。"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import trench_rf as RF


@lru_cache(maxsize=1)
def _cupy():
    # カーネルキャッシュをプロジェクト内にまとめる (ユーザー指定があれば尊重)。
    os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "cupy"))
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CUDA 依存がありません。Python 3.10 以上で uv sync --extra cuda を実行してください") from exc
    return cp


class CudaTracer:
    """形状とカーネルを再利用する。電場と粒子を転送し、GPU で移動・反射をまとめて計算。"""

    def __init__(self, vac, device=0):
        self.cp = cp = _cupy()
        self.device = int(device)
        if self.device < 0 or self.device >= cp.cuda.runtime.getDeviceCount():
            raise RuntimeError(f"CUDA デバイス {self.device} は利用できません")
        self.vac = np.asarray(vac, dtype=bool).copy()
        with cp.cuda.Device(self.device):
            self.gpu_vac = cp.asarray(self.vac, dtype=cp.uint8)
            self.dummy = cp.zeros(1, dtype=cp.float64)
            self.dummy_int = cp.zeros(1, dtype=cp.int32)
            code = Path(__file__).with_name("trench_trace.cu").read_text(encoding="utf-8")
            self.kernel = cp.RawKernel(code, "trace_particles", options=("--std=c++17", "--fmad=false"))
            self.kernel.compile()
            props = cp.cuda.runtime.getDeviceProperties(self.device)
            name = props["name"]
            self.name = name.decode() if isinstance(name, bytes) else str(name)

    def __call__(self, x, y, vx, vy, qm, spc, Ex, Ey, vac, p, barrier=None, record=None, stop=None, rf=None):
        if vac.shape != self.vac.shape or not np.array_equal(vac, self.vac):
            raise ValueError("CUDA 追跡器の形状が変わっています。追跡器を作り直してください")
        if not (0 < p.cfl <= 1) or p.max_steps < 1:
            raise ValueError("CUDA 追跡には 0 < cfl <= 1、max_steps >= 1 が必要です")
        cp = self.cp
        with cp.cuda.Device(self.device):
            return self._trace(x, y, vx, vy, qm, spc, Ex, Ey, p, barrier, record, stop, rf)

    def _trace(self, x, y, vx, vy, qm, spc, Ex, Ey, p, barrier, record, stop, rf):
        cp = self.cp
        n = len(x)
        nx, ny = self.vac.shape
        if any(np.shape(v) != (n,) for v in (x, y, vx, vy, qm, spc)):
            raise ValueError("粒子配列の長さが一致しません")
        if any(np.shape(v) != (nx, ny) for v in (Ex, Ey)):
            raise ValueError("電場の形状が一致しません")
        if np.any((spc != 0) & (spc != 1)):
            raise ValueError("粒子種は 0 (イオン) または 1 (電子) にしてください")
        particles = cp.asarray(np.column_stack((x, y, vx, vy, qm)), dtype=cp.float64)
        gpu_ex, gpu_ey = (cp.asarray(np.ascontiguousarray(v, dtype=np.float64)) for v in (Ex, Ey))
        if rf is None:
            rf_ex = rf_ey = rf_phi = rf_wave = clocks = self.dummy
            period = dt_max = 1.0
            n_phase, bound = 0, np.zeros(2)
        else:
            RF.validate(rf, n, self.vac.shape)
            rf_ex, rf_ey, rf_phi, rf_wave = (cp.asarray(np.ascontiguousarray(rf[k], dtype=np.float64))
                                           for k in ("Ex", "Ey", "phi", "wave"))
            clocks = cp.asarray(np.column_stack((rf["initial_time"] % rf["period"], np.zeros(n))), dtype=cp.float64)
            period, dt_max, n_phase = rf["period"], rf["dt_max"], len(rf["wave"])
            bound = np.abs(rf["wave"]).max(axis=0)
        status = cp.full(n, 3, dtype=cp.int8)  # 1=吸収, 2=上端流出, 3=追跡中/打ち切り
        hits = cp.full(n, -1, dtype=cp.int32)
        if barrier is None:
            gpu_phi = gpu_wface = self.dummy
        else:
            if any(np.shape(v) != (nx, ny) for v in barrier):
                raise ValueError("壁面電位の形状が一致しません")
            gpu_phi, gpu_wface = (cp.asarray(np.ascontiguousarray(v, dtype=np.float64)) for v in barrier)
        path_stride = p.max_steps + 1
        if record is not None:
            if p.probe_trajectories < 1:
                raise ValueError("表示する軌道数は 1 以上にしてください")
            selected = np.unique(np.linspace(0, n-1, min(p.probe_trajectories, n), dtype=int))
            slots = np.full(n, -1, dtype=np.int32)
            slots[selected] = np.arange(len(selected))
            gpu_slots = cp.asarray(slots)
            endpoints = cp.asarray(np.column_stack((x, y)), dtype=cp.float64)
            # 書き込んだ部分だけを後でホストへコピーし、未初期化領域は結果に含めない。
            paths = cp.empty((len(selected), path_stride, 2), dtype=cp.float64)
            lengths = cp.ones(len(selected), dtype=cp.int32)
            if n:
                paths[:, 0] = endpoints[cp.asarray(selected)]
        else:
            endpoints = paths = self.dummy
            gpu_slots = lengths = self.dummy_int
        host_status = np.full(n, 3, dtype=np.int8)
        cancelled = False
        for offset in range(0, p.max_steps, 256):
            if not n or not np.any(host_status == 3):
                break
            if stop is not None and stop.is_set():
                cancelled = True
                break
            # 短いチャンクに分け、診断の停止要求をカーネル間で受け取る。
            self.kernel(((n+127)//128,), (128,), (
                particles, gpu_ex, gpu_ey, self.gpu_vac, gpu_phi, gpu_wface, status, hits,
                endpoints, gpu_slots, paths, lengths,
                rf_ex, rf_ey, rf_phi, rf_wave, clocks,
                np.int32(n), np.int32(nx), np.int32(ny), np.int32(min(256, p.max_steps-offset)),
                np.int32(offset == 0), np.int32(barrier is not None), np.int32(record is not None),
                np.int32(bool(p.trench_taper_deg or (p.mask_t > 0 and p.mask_taper_deg))),
                np.int32(path_stride), np.float64(p.dx), np.float64(p.cfl),
                np.int32(n_phase), np.float64(period), np.float64(dt_max),
                np.float64(bound[0]), np.float64(bound[1])))
            host_status = cp.asnumpy(status)  # 同期はチャンクごと。粒子ステップごとの CPU 呼び出しは不要。
        host_hits = cp.asnumpy(hits)
        counts = [np.bincount(host_hits[(host_status == 1) & (spc == k)], minlength=nx*ny).reshape(nx, ny)
                  for k in (0, 1)]
        escaped = np.bincount(spc[host_status == 2], minlength=2).astype(np.int64)
        lost = int(np.count_nonzero(host_status == 3))
        if record is not None:
            cells = np.full((n, 2), -1, dtype=int)
            absorbed = host_status == 1
            cells[absorbed] = np.column_stack((host_hits[absorbed] // ny, host_hits[absorbed] % ny))
            host_lengths = cp.asnumpy(lengths)
            host_particles = cp.asnumpy(particles)
            record.update(selected=selected, status=host_status, hit_cells=cells,
                          initial_x=np.asarray(x).copy(), endpoints=cp.asnumpy(endpoints),
                          final_velocity=host_particles[:, 2:4].copy(),
                          paths=[cp.asnumpy(paths[k, :length]) for k, length in enumerate(host_lengths)],
                          cancelled=cancelled or (stop is not None and stop.is_set()))
            if rf is not None:
                timing = cp.asnumpy(clocks)
                record.update(launch_phase=rf["initial_time"] % period / period * 360,
                              final_phase=timing[:, 0] / period * 360, flight_time=timing[:, 1])
        return counts[0], counts[1], escaped, lost
