# -*- coding: utf-8 -*-
"""分场景决策引擎：预警等级与干预动作"""
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class AlertAction:
    level: str          # 绿/黄/橙/红
    scenario: str
    title: str
    message: str
    actions: List[str]
    auto_call_120: bool
    notify_family: bool
    bystander_mode: bool

def classify_scenario(pred_label: int, proba: np.ndarray, vitals_alert: bool) -> str:
    names = ["正常行走", "跌倒倾向", "失稳-可自救", "突发跌倒", "蹲下/坐下"]
    base = names[pred_label]
    if vitals_alert and pred_label in (0, 1):
        return "体征异常叠加风险"
    return base

def decide(pred_label: int, proba: np.ndarray, vitals: dict) -> AlertAction:
    """根据分类结果与体征生成干预方案"""
    conf = proba.max()
    hr_alert = vitals.get("hr_alert", False)
    bp_alert = vitals.get("bp_alert", False)
    vitals_alert = hr_alert or bp_alert

    if pred_label == 0 and not vitals_alert:
        return AlertAction(
            "绿", "正常行走", "状态良好",
            "步态稳定，体征正常。继续保持适度活动。",
            ["记录今日步态评分", "同步健康周报给家属"],
            False, False, False,
        )

    if pred_label == 1 or vitals_alert:
        return AlertAction(
            "黄", "跌倒倾向" if pred_label == 1 else "体征异常",
            "跌倒风险升高",
            "检测到步态不对称或体征波动，建议放慢速度、使用扶手。",
            ["推送语音提醒", "通知家属「请关注」", "生成今日风险报告"],
            False, True, False,
        )

    if pred_label == 2:
        return AlertAction(
            "橙", "失稳-可自救",
            "失稳预警 — 请扶稳",
            "检测到短暂失稳，您可能仍可自主恢复。请立即扶稳或坐下。",
            ["手环震动 + 语音播报", "30秒未恢复 → 通知家属", "不自动拨打120"],
            False, True, False,
        )

    if pred_label == 3:
        return AlertAction(
            "红", "突发跌倒",
            "紧急警报 — 跌倒确认",
            f"检测到突发倒地（置信度 {conf:.0%}）。即将启动紧急协议。",
            [
                "60秒倒计时取消窗口",
                "自动拨打120",
                "推送GPS与病史摘要",
                "短信通知全部紧急联系人",
                "开启录音录像",
                "启动周围人求助模式",
            ],
            True, True, True,
        )

    if pred_label == 4:
        return AlertAction(
            "蓝", "蹲下/坐下",
            "姿态变化（非紧急）",
            "检测到蹲下或坐下动作，与跌倒模式不同。无需紧急呼叫。",
            ["记录事件", "若长时间无活动将升级提醒"],
            False, False, False,
        )

    return AlertAction("绿", "未知", "监测中", "系统分析中…", [], False, False, False)

def emergency_timeline() -> List[dict]:
    """跌倒后紧急协议时间线（用于可视化）"""
    return [
        {"t": 0, "event": "跌倒检测", "icon": "!"},
        {"t": 3, "event": "App弹窗确认", "icon": "App"},
        {"t": 60, "event": "倒计时结束", "icon": "T"},
        {"t": 61, "event": "自动拨打120", "icon": "120"},
        {"t": 62, "event": "通知紧急联系人", "icon": "家"},
        {"t": 63, "event": "推送定位+病史", "icon": "GPS"},
        {"t": 65, "event": "周围人求助广播", "icon": "播"},
        {"t": 66, "event": "显示授权求助码", "icon": "码"},
    ]
