"""RF 周期の粒子時計と、周期的な電極電圧の補間。"""
import numpy as np


def coefficients(rf, at):
    """周期端をつなぐ線形補間。wave は平均電位からの差分 [基板, マスク]。"""
    wave = rf["wave"]
    f = (np.asarray(at) % rf["period"]) / rf["period"] * len(wave)
    i = np.floor(f).astype(np.int64) % len(wave)
    w = (f - np.floor(f))[..., None]
    return wave[i] * (1 - w) + wave[(i + 1) % len(wave)] * w


def launch_times(rng, n, period):
    """1 周期を n 区間に層化し、位置・速度との相関を避けて順序を混ぜる。"""
    times = (np.arange(n) + rng.random(n)) * period / n
    rng.shuffle(times)
    return times


def validate(rf, n, shape):
    if (not np.isfinite(rf["period"]) or rf["period"] <= 0
            or not np.isfinite(rf["dt_max"]) or rf["dt_max"] <= 0):
        raise ValueError("RF 周期と時間刻みの上限は正の有限値にしてください")
    for key in ("phi", "Ex", "Ey"):
        if np.shape(rf[key]) != (2, *shape) or not np.isfinite(rf[key]).all():
            raise ValueError("RF 電場の基底の形状または数値が不正です")
    wave = np.asarray(rf["wave"])
    if wave.ndim != 2 or wave.shape[1] != 2 or len(wave) < 2 or not np.isfinite(wave).all():
        raise ValueError("RF 電圧の波形が不正です")
    if np.shape(rf["initial_time"]) != (n,) or not np.isfinite(rf["initial_time"]).all():
        raise ValueError("粒子の入射時刻の形状または数値が不正です")
