# -*- coding: utf-8 -*-
"""PPT 封面海报"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from config import FIGURES, COLORS
from style import setup_style, save_fig

setup_style()

def fig_poster():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    grad = LinearSegmentedColormap.from_list("g", [COLORS["primary"], COLORS["secondary"]], N=256)
    for i in range(200):
        ax.axvline(i * 0.08, ymin=0, ymax=0.35, color=grad(i / 200), alpha=0.15, lw=2)

    ax.text(8, 6.2, "MEMS足底压力步态监测", ha="center", fontsize=34, fontweight="bold", color="white")
    ax.text(8, 4.9, "面向老年人的五类步态场景识别\n与跌倒风险分级预警仿真",
            ha="center", fontsize=14, color="#94A3B8", linespacing=1.6)
    ax.text(8, 3.6, "足底压力阵列  |  IMU融合  |  可解释特征  |  分级决策",
            ha="center", fontsize=11, color=COLORS["primary"])

    tags = ["多模态融合", "小波时频", "Stacking+MLP", "分场景决策", "体征监测", "鲁棒性评估"]
    for i, t in enumerate(tags):
        x = 1.2 + (i % 3) * 4.8
        y = 2.2 - (i // 3) * 0.9
        rect = FancyBboxPatch((x, y), 4, 0.65, boxstyle="round,pad=0.02",
                               facecolor=COLORS["bg_card"], edgecolor=COLORS["primary"], alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 2, y + 0.32, t, ha="center", va="center", color="white", fontsize=10)

    ax.text(8, 0.5, "仿真验证 · 算法驱动 · 全周期闭环", ha="center", fontsize=10, color="#64748B")
    save_fig(fig, FIGURES / "00_PPT封面海报.png")

if __name__ == "__main__":
    fig_poster()
