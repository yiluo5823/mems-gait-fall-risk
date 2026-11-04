# -*- coding: utf-8 -*-
"""全局配置：项目路径、配色与仿真参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "src"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
GIFS = RESULTS / "gifs"
VIDEOS = RESULTS / "videos"
DATA = RESULTS / "data"

for p in (FIGURES, GIFS, VIDEOS, DATA):
    p.mkdir(parents=True, exist_ok=True)

# 8×8 柔性压力阵列
GRID = 8
FPS_ANIM = 8
DPI = 200
DPI_POSTER = 300

# 专业医疗科技配色
COLORS = {
    "primary": "#0D9488",      # 青绿
    "secondary": "#6366F1",    # 靛蓝
    "accent": "#F59E0B",       # 琥珀
    "danger": "#EF4444",
    "warning": "#F97316",
    "success": "#10B981",
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "bg_light": "#F8FAFC",
    "text": "#1E293B",
    "text_light": "#94A3B8",
    "gradient_start": "#0D9488",
    "gradient_end": "#6366F1",
}

RISK_COLORS = {"低": "#10B981", "中": "#F59E0B", "高": "#EF4444"}
SCENARIO_COLORS = {
    "正常行走": "#10B981",
    "跌倒倾向": "#F59E0B",
    "失稳-可自救": "#F97316",
    "突发跌倒": "#EF4444",
    "蹲下/坐下": "#6366F1",
}

RANDOM_SEED = 42
