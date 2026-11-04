# -*- coding: utf-8 -*-
"""时频、小波、非线性与IMU特征。"""
import numpy as np
from scipy import stats
from scipy.signal import welch, hilbert
import pywt

from simulator import cop_trajectory


def wavelet_energy(seq_1d: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    coeffs = pywt.wavedec(seq_1d, wavelet, level=level)
    return np.array([np.sum(c ** 2) for c in coeffs])


def spectral_features(seq_1d: np.ndarray, fs: float = 10.0) -> dict:
    f, psd = welch(seq_1d, fs=fs, nperseg=min(64, max(16, len(seq_1d) // 2)))
    psd = psd + 1e-12
    peak_f = f[np.argmax(psd)]
    entropy = -np.sum(psd * np.log(psd))
    return {"spec_peak_freq": float(peak_f), "spec_entropy": float(entropy), "spec_centroid": float(np.sum(f * psd) / psd.sum())}


def sample_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """简化样本熵（步态复杂度）"""
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return 0.0
    if r is None:
        r = 0.2 * x.std()
    if r < 1e-8:
        return 0.0
    n = len(x)

    def _phi(mm):
        patterns = np.array([x[i : i + mm] for i in range(n - mm)])
        C = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= r, axis=0) - 1
        C = C[C > 0]
        if len(C) == 0:
            return 0
        return np.mean(C / (n - mm))

    a, b = _phi(m), _phi(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(b / a))


def extract_advanced_features(seq: np.ndarray, imu: np.ndarray = None) -> np.ndarray:
    """seq: (T,H,W), imu: (T,3) optional"""
    from simulator import extract_features, feature_vector

    base = feature_vector(extract_features(seq))
    total = seq.reshape(len(seq), -1).sum(axis=1)
    cop = cop_trajectory(seq)
    cop_sp = np.linalg.norm(np.diff(cop, axis=0), axis=1)

    wav = wavelet_energy(total)
    spec = spectral_features(total)
    sampen = sample_entropy(total)
    hilb_amp = np.abs(hilbert(total - total.mean()))
    hilb_mean = float(hilb_amp.mean())

    extra = [
        sampen,
        hilb_mean,
        float(stats.skew(total)),
        float(stats.kurtosis(total)),
        float(cop_sp.std()) if len(cop_sp) else 0,
        spec["spec_peak_freq"],
        spec["spec_entropy"],
        spec["spec_centroid"],
        *wav.tolist(),
    ]

    if imu is not None and len(imu) > 5:
        acc_mag = np.linalg.norm(imu, axis=1)
        extra.extend([
            float(acc_mag.mean()),
            float(acc_mag.std()),
            float(acc_mag.max()),
            spectral_features(acc_mag)["spec_peak_freq"],
        ])
    else:
        extra.extend([0.0] * 4)

    return np.concatenate([base, np.array(extra, dtype=np.float32)])
