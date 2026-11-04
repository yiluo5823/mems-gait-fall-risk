# -*- coding: utf-8 -*-
"""日常体征仿真与阈值预警"""
import numpy as np
from config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED + 1)

def simulate_vitals_days(n_days: int = 14, profile: str = "normal") -> dict:
    """生成多日体征时序"""
    days = np.arange(n_days)
    if profile == "normal":
        sbp = rng.normal(128, 8, n_days)
        dbp = rng.normal(78, 5, n_days)
        hr = rng.normal(72, 6, n_days)
        spo2 = rng.normal(97, 1, n_days)
    elif profile == "hypertension":
        sbp = rng.normal(165, 12, n_days)
        dbp = rng.normal(95, 8, n_days)
        hr = rng.normal(78, 8, n_days)
        spo2 = rng.normal(96, 1.5, n_days)
    else:  # fall_risk — 体位性低血压倾向
        sbp = rng.normal(118, 15, n_days)
        sbp[-3:] -= 25  # 近期下降
        dbp = rng.normal(70, 8, n_days)
        hr = rng.normal(88, 10, n_days)
        hr[-2:] = 105
        spo2 = rng.normal(95, 2, n_days)

    return {
        "days": days,
        "sbp": np.clip(sbp, 80, 220),
        "dbp": np.clip(dbp, 40, 130),
        "hr": np.clip(hr, 45, 150),
        "spo2": np.clip(spo2, 85, 100),
    }

def check_vitals(sbp, dbp, hr, spo2) -> dict:
    alerts = []
    if sbp >= 180 or sbp < 90:
        alerts.append("血压异常")
    if dbp >= 110:
        alerts.append("舒张压过高")
    if hr > 120 or hr < 50:
        alerts.append("心率异常")
    if spo2 < 92:
        alerts.append("血氧偏低")
    return {
        "bp_alert": "血压异常" in alerts or "舒张压过高" in alerts,
        "hr_alert": "心率异常" in alerts,
        "spo2_alert": "血氧偏低" in alerts,
        "alerts": alerts,
        "sbp": sbp, "dbp": dbp, "hr": hr, "spo2": spo2,
    }
