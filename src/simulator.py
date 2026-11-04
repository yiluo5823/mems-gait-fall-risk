# -*- coding: utf-8 -*-
"""足底柔性压力与IMU多模态时序仿真，包含类间重叠和噪声。"""
import numpy as np
from config import GRID, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


def _gaussian2d(h, w, cx, cy, sx, sy, amp=1.0):
    y, x = np.ogrid[:h, :w]
    return amp * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))


def _footprint(side="L", intensity=1.0):
    h = w = GRID
    cx, cy = (w * 0.35, h * 0.55) if side == "L" else (w * 0.65, h * 0.55)
    base = _gaussian2d(h, w, cx, cy, 1.2, 1.8, intensity)
    heel = _gaussian2d(h, w, cx, h * 0.82, 0.8, 0.7, 0.35 * intensity)
    return np.clip(base + heel, 0, None)


def _walk_cycle(phase, asymmetry=0.0, noise=0.08):
    left = _footprint("L", max(0, np.sin(phase * 2 * np.pi) ** 2))
    right = _footprint("R", max(0, np.sin((phase + 0.5) * 2 * np.pi) ** 2))
    if asymmetry:
        right *= 1 - asymmetry
    frame = left + right
    frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0, None)


def _imu_walk(n, phase0=0.0, stumble=False, fall=False):
    t = np.arange(n)
    ax = 0.3 * np.sin(2 * np.pi * t / 40 + phase0) + rng.normal(0, 0.05, n)
    ay = 0.2 * np.cos(2 * np.pi * t / 40) + rng.normal(0, 0.05, n)
    az = 1.0 + 0.1 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.03, n)
    if stumble:
        k = min(n // 2, n - 1)
        ax[k : k + 5] += rng.uniform(1.5, 2.5)
        ay[k : k + 5] += rng.uniform(-2, -1)
    if fall:
        k = min(int(n * 0.6), n - 3)
        ax[k:] = rng.normal(0, 0.02, n - k)
        ay[k:] = rng.normal(0, 0.02, n - k)
        az[k:] = 0.15 + rng.normal(0, 0.01, n - k)
    return np.stack([ax, ay, az], axis=1).astype(np.float32)


def generate_sequence(scenario: str, n_frames: int = 80) -> tuple:
    """返回 (pressure[T,H,W], imu[T,3])"""
    seq, imu = [], None

    if scenario == "normal":
        imu = _imu_walk(n_frames)
        for i in range(n_frames):
            seq.append(_walk_cycle((i % 40) / 40, 0.02 + rng.uniform(0, 0.03), 0.07))

    elif scenario == "risk_gait":
        imu = _imu_walk(n_frames)
        for i in range(n_frames):
            asym = 0.18 + 0.12 * np.sin(i / 8) + rng.uniform(0, 0.08)
            seq.append(_walk_cycle((i % 50) / 50, asym, 0.11))

    elif scenario == "stumble_recovery":
        imu = _imu_walk(n_frames, stumble=True)
        for i in range(n_frames):
            if i < 25:
                seq.append(_walk_cycle((i % 40) / 40, 0.05, 0.08))
            elif i < 38:
                f = _walk_cycle(0.2, 0.35 + rng.uniform(0, 0.1), 0.18) * (1.6 + rng.uniform(0, 0.3))
                f += _gaussian2d(GRID, GRID, GRID / 2, GRID / 2, 2.5, 2.5, 0.5)
                seq.append(f)
            else:
                t = (i - 38) / max(1, n_frames - 38)
                seq.append(_footprint("L", 0.4 + 0.6 * t) + _footprint("R", 0.4 + 0.6 * t))

    elif scenario == "sudden_fall":
        imu = _imu_walk(n_frames, fall=True)
        for i in range(n_frames):
            if i < 28:
                seq.append(_walk_cycle((i % 40) / 40, 0.03, 0.07))
            elif i < 36:
                seq.append(_gaussian2d(GRID, GRID, GRID / 2, GRID / 2, 3.5, 3.5, 2.2 + rng.uniform(0, 0.5))
                                + rng.normal(0, 0.12, (GRID, GRID)))
            else:
                static = _gaussian2d(GRID, GRID, GRID / 2, GRID / 2, 3.2, 2.8, 1.8)
                seq.append(static + rng.normal(0, 0.04, (GRID, GRID)))

    elif scenario == "squat":
        imu = _imu_walk(n_frames)
        for i in range(n_frames):
            if i < 38:
                seq.append(_walk_cycle((i % 40) / 40, 0.03, 0.07))
            else:
                t = min(1, (i - 38) / 14)
                seq.append(np.clip((_footprint("L", 0.7) + _footprint("R", 0.7)) * (1.1 + 0.4 * t), 0, None))

    else:
        raise ValueError(scenario)

    # 传感器漂移、量化噪声
    arr = np.array(seq, dtype=np.float32)
    arr += rng.normal(0, 0.04, arr.shape)
    arr = np.clip(arr, 0, None)
    return arr, imu


def cop_trajectory(pressure_seq: np.ndarray) -> np.ndarray:
    T, H, W = pressure_seq.shape
    cop = np.zeros((T, 2))
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(T):
        p = pressure_seq[t]
        s = p.sum() + 1e-8
        cop[t, 0] = (p * xx).sum() / s
        cop[t, 1] = (p * yy).sum() / s
    return cop


def extract_features(seq: np.ndarray) -> dict:
    T = len(seq)
    total = seq.reshape(T, -1).sum(axis=1)
    left = seq[:, :, : GRID // 2].sum(axis=(1, 2))
    right = seq[:, :, GRID // 2 :].sum(axis=(1, 2))
    asym = np.abs(left - right) / (left + right + 1e-8)
    cop = cop_trajectory(seq)
    cop_speed = np.linalg.norm(np.diff(cop, axis=0), axis=1)

    from scipy.signal import welch
    if T > 16:
        f, psd = welch(total, fs=10, nperseg=min(32, T))
        stride_freq = f[np.argmax(psd[1:]) + 1] if len(f) > 1 else 0
    else:
        stride_freq = 0

    impact = total.max()
    tail = total[-max(5, T // 5) :]
    static_ratio = (tail.std() / (tail.mean() + 1e-8)) if len(tail) > 2 else 1
    recovery = _detect_recovery(total)

    return {
        "mean_pressure": float(total.mean()),
        "pressure_std": float(total.std()),
        "asymmetry_mean": float(asym.mean()),
        "asymmetry_max": float(asym.max()),
        "cop_speed_mean": float(cop_speed.mean()) if len(cop_speed) else 0,
        "cop_speed_max": float(cop_speed.max()) if len(cop_speed) else 0,
        "stride_freq": float(stride_freq),
        "impact_peak": float(impact),
        "static_stability": float(1 / (static_ratio + 0.1)),
        "recovery_score": float(recovery),
        "duration": T,
    }


def _detect_recovery(total: np.ndarray) -> float:
    if len(total) < 20:
        return 0
    peak_idx = np.argmax(total)
    if peak_idx > len(total) * 0.55:
        after = total[peak_idx:]
        if len(after) > 5 and after[-1] > after[: min(5, len(after))].mean() * 0.8:
            return 1.0
    return 0.0


def feature_vector(feat: dict) -> np.ndarray:
    keys = [
        "mean_pressure", "pressure_std", "asymmetry_mean", "asymmetry_max",
        "cop_speed_mean", "cop_speed_max", "stride_freq",
        "impact_peak", "static_stability", "recovery_score",
    ]
    return np.array([feat[k] for k in keys], dtype=np.float32)


SCENARIO_LABELS = {
    "normal": 0,
    "risk_gait": 1,
    "stumble_recovery": 2,
    "sudden_fall": 3,
    "squat": 4,
}

LABEL_NAMES = ["正常行走", "跌倒倾向", "失稳-可自救", "突发跌倒", "蹲下/坐下"]


def build_dataset(n_per_class: int = 200, hard: bool = True) -> tuple:
    """hard=True 时注入边界样本与标签噪声"""
    from features_advanced import extract_advanced_features

    X, y = [], []
    for scen, label in SCENARIO_LABELS.items():
        for _ in range(n_per_class):
            n_frames = int(rng.integers(65, 95))
            seq, imu = generate_sequence(scen, n_frames)
            # 边界混淆：倾向样本偶尔像正常
            if hard and scen == "risk_gait" and rng.random() < 0.12:
                seq, imu = generate_sequence("normal", n_frames)
            if hard and scen == "stumble_recovery" and rng.random() < 0.08:
                seq, _ = generate_sequence("squat", n_frames)
                imu = _imu_walk(n_frames, stumble=True)
            X.append(extract_advanced_features(seq, imu))
            y.append(label)

    X, y = np.array(X), np.array(y)
    return X, y
