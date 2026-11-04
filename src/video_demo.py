# -*- coding: utf-8 -*-
"""生成产品完整流程演示 MP4（流式写入，低内存）"""
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
import imageio

from config import VIDEOS, COLORS
from style import setup_style, pressure_cmap
from simulator import generate_sequence
from models import predict_from_sequence
from decision_engine import decide
from vitals import check_vitals

setup_style()

W, H = 960, 544  # 可被 16 整除，避免 ffmpeg 缩放
FPS = 20
DPI = 96


def _frame_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    if img.shape[1] != W or img.shape[0] != H:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((W, H), Image.Resampling.LANCZOS))
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _write_scene(writer, draw_func, duration_sec):
    n = max(1, int(duration_sec * FPS))
    for _ in range(n):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        draw_func(fig)
        writer.append_data(_frame_from_fig(fig))


def scene_title(fig, title, subtitle=""):
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", fontsize=22, fontweight="bold", color="white")
    if subtitle:
        ax.text(0.5, 0.38, subtitle, ha="center", fontsize=12, color="#94A3B8")
    ax.text(0.5, 0.12, "MEMS足底压力步态监测与跌倒风险预警", ha="center", fontsize=10, color=COLORS["primary"])


def scene_architecture(fig):
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.axis("off")
    ax.text(0.5, 0.95, "系统架构", ha="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
    layers = ["柔性传感", "AI算法", "分场景决策", "App干预"]
    for i, L in enumerate(layers):
        x = 0.08 + i * 0.23
        rect = FancyBboxPatch((x, 0.4), 0.18, 0.35, boxstyle="round,pad=0.02", transform=ax.transAxes,
                               facecolor=COLORS["primary"] if i % 2 == 0 else COLORS["secondary"], alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + 0.09, 0.57, L, ha="center", va="center", color="white", fontsize=11, fontweight="bold", transform=ax.transAxes)


def scene_pressure(fig, seq, frame_idx, title):
    fig.patch.set_facecolor(COLORS["bg_dark"])
    gs = GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cmap = pressure_cmap()
    fi = min(frame_idx, len(seq) - 1)
    ax1.imshow(seq[fi], cmap=cmap, vmin=0, vmax=max(seq.max(), 0.1))
    ax1.set_title("柔性压力阵列 8x8", color="white", fontsize=11)
    ax1.axis("off")
    total = seq.reshape(len(seq), -1).sum(axis=1)
    ax2.plot(total, color=COLORS["accent"], lw=2)
    ax2.axvline(fi, color=COLORS["danger"], ls="--", lw=2)
    ax2.set_facecolor(COLORS["bg_card"])
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_title("压力时序", color="white", fontsize=11)
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)


def scene_alert(fig, action):
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    cmap_lvl = {"绿": COLORS["success"], "黄": "#F59E0B", "橙": "#F97316", "红": COLORS["danger"], "蓝": COLORS["secondary"]}
    color = cmap_lvl.get(action.level, COLORS["primary"])
    rect = FancyBboxPatch((0.12, 0.18), 0.76, 0.64, boxstyle="round,pad=0.03",
                           facecolor=COLORS["bg_card"], edgecolor=color, linewidth=4)
    ax.add_patch(rect)
    ax.text(0.5, 0.72, f"[{action.level}警] {action.title}", ha="center", fontsize=18, fontweight="bold", color=color)
    ax.text(0.5, 0.58, action.message, ha="center", fontsize=11, color="white")
    y = 0.46
    for act in action.actions[:5]:
        ax.text(0.5, y, "> " + act, ha="center", fontsize=9, color="#94A3B8")
        y -= 0.055


def scene_emergency(fig, step):
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    steps = ["跌倒确认", "60s倒计时", "拨打120", "通知家属", "周围人求助"]
    ax.text(0.5, 0.88, "紧急协议执行中", ha="center", fontsize=18, fontweight="bold", color=COLORS["danger"])
    for i, s in enumerate(steps):
        active = i <= step
        c = COLORS["danger"] if active else "#475569"
        ax.add_patch(Circle((0.15 + i * 0.17, 0.5), 0.035, facecolor=c, transform=ax.transAxes))
        ax.text(0.15 + i * 0.17, 0.38, s, ha="center", fontsize=8, color="white" if active else "#64748B", transform=ax.transAxes)


def scene_vitals(fig):
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax = fig.add_axes([0.12, 0.15, 0.76, 0.7])
    days = np.arange(14)
    sbp = 128 + np.cumsum(np.random.randn(14) * 2)
    sbp[-1] = 178
    ax.plot(days, sbp, "o-", color=COLORS["danger"], lw=2)
    ax.axhspan(90, 140, alpha=0.15, color=COLORS["success"], label="正常")
    ax.scatter([13], [178], s=200, c=COLORS["danger"], marker="*", zorder=5)
    ax.set_title("模块A: 血压异常预警", fontweight="bold")
    ax.set_xlabel("天数")
    ax.grid(alpha=0.3)


def build_demo_video(bundle, output_path):
    print("\n[视频] 生成演示 MP4 (流式)...")
    output_path = str(output_path)
    writer = imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p")

    _write_scene(writer, lambda f: scene_title(f, "康护智行", "跌倒全周期智能监护系统 · 仿真演示"), 2.5)
    _write_scene(writer, scene_architecture, 2)

    seq_n = generate_sequence("normal", 40)
    for fi in [5, 15, 25, 35]:
        _write_scene(writer, lambda f, fi=fi: scene_pressure(f, seq_n, fi, "场景1: 正常行走监测"), 0.5)

    seq_r = generate_sequence("risk_gait", 40)
    pred, proba, _ = predict_from_sequence(seq_r, bundle)
    act_r = decide(pred, proba, check_vitals(135, 82, 75, 97))
    _write_scene(writer, lambda f: scene_pressure(f, seq_r, 20, "场景2: 跌倒倾向识别"), 1)
    _write_scene(writer, lambda f: scene_alert(f, act_r), 2.5)

    seq_s = generate_sequence("stumble_recovery", 50)
    pred, proba, _ = predict_from_sequence(seq_s, bundle)
    act_s = decide(pred, proba, {"hr_alert": False, "bp_alert": False})
    _write_scene(writer, lambda f: scene_pressure(f, seq_s, 32, "场景3: 失稳-可自救"), 1)
    _write_scene(writer, lambda f: scene_alert(f, act_s), 2.5)

    seq_f = generate_sequence("sudden_fall", 50)
    pred, proba, _ = predict_from_sequence(seq_f, bundle)
    act_f = decide(pred, proba, {"hr_alert": True, "bp_alert": False})
    _write_scene(writer, lambda f: scene_pressure(f, seq_f, 40, "场景4: 突发跌倒"), 1)
    _write_scene(writer, lambda f: scene_alert(f, act_f), 2)
    for step in range(5):
        _write_scene(writer, lambda f, s=step: scene_emergency(f, s), 0.8)

    _write_scene(writer, scene_vitals, 2)
    _write_scene(writer, lambda f: scene_title(f, "感谢观看", "算法仿真验证 · 全周期闭环 · 软干预非仅预警"), 2)

    writer.close()
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    dur = (2.5 + 2 + 2 + 1 + 2.5 + 1 + 2.5 + 1 + 2 + 4 + 2 + 2)  # approx
    print(f"  saved: {output_path} ({size_mb:.1f} MB, ~{dur:.0f}s)")
