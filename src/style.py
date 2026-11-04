# -*- coding: utf-8 -*-
"""matplotlib 全局美观样式"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import COLORS, DPI

DPI_POSTER = 200

def setup_style():
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI_POSTER,
        "figure.facecolor": COLORS["bg_light"],
        "axes.facecolor": "white",
        "axes.edgecolor": "#E2E8F0",
        "axes.labelcolor": COLORS["text"],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.color": COLORS["text_light"],
        "ytick.color": COLORS["text_light"],
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.8,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E8F0",
    })

def gradient_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "medical", [COLORS["gradient_start"], COLORS["secondary"], COLORS["accent"]], N=256
    )

def pressure_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "pressure", ["#0F172A", "#0D9488", "#34D399", "#FBBF24", "#EF4444"], N=256
    )

def save_fig(fig, path, tight=True):
    try:
        if tight:
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none", dpi=DPI_POSTER)
        else:
            fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=DPI_POSTER)
        print(f"  saved: {path}")
    finally:
        plt.close(fig)
        import gc
        gc.collect()
