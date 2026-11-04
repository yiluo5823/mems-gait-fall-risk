# -*- coding: utf-8 -*-
"""足底压力、步态分类与风险决策可视化。"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.gridspec import GridSpec
import seaborn as sns
import imageio

from config import FIGURES, GIFS, COLORS, SCENARIO_COLORS, GRID
from style import setup_style, pressure_cmap, save_fig
from simulator import generate_sequence, cop_trajectory, LABEL_NAMES, SCENARIO_LABELS
from decision_engine import emergency_timeline
from vitals import simulate_vitals_days, check_vitals
from models import predict_from_sequence

setup_style()
CMAP_PRESSURE = pressure_cmap()


def fig_tools_and_difficulty():
    """展示项目使用的工具与技术模块。"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax.text(7, 7.5, "技术工具与项目模块", ha="center", fontsize=17, fontweight="bold")

    tools = [
        ("Python / NumPy / SciPy", "仿真与信号处理", COLORS["primary"]),
        ("PyWavelets + scikit-learn", "小波特征 · Stacking · MLP", COLORS["secondary"]),
        ("Matplotlib / Seaborn", "科研级可视化输出", COLORS["accent"]),
        ("PyTorch 1D-CNN（扩展）", "深度学习对比实验", "#8B5CF6"),
        ("App 原型设计", "全周期交互流程", COLORS["bg_dark"]),
    ]
    for i, (name, desc, c) in enumerate(tools):
        y = 5.8 - i * 0.95
        rect = FancyBboxPatch((0.8, y), 5.5, 0.75, boxstyle="round,pad=0.02", facecolor="white", edgecolor=c, lw=2)
        ax.add_patch(rect)
        ax.text(1.1, y + 0.48, name, fontsize=11, fontweight="bold", color=c)
        ax.text(1.1, y + 0.15, desc, fontsize=9, color=COLORS["text_light"])

    levels = [
        ("基础层", "单模态压力特征与分类", False),
        ("融合层", "压力、IMU、体征的多模态融合", True),
        ("应用层", "分场景决策与响应策略", True),
    ]
    for i, (lv, desc, hl) in enumerate(levels):
        y = 5.5 - i * 1.3
        c = COLORS["primary"] if hl else "#CBD5E1"
        rect = FancyBboxPatch((7.2, y), 6, 1.0, boxstyle="round,pad=0.02",
                               facecolor=c if hl else "white", edgecolor=c, lw=2.5 if hl else 1, alpha=0.9 if hl else 0.5)
        ax.add_patch(rect)
        ax.text(7.5, y + 0.65, lv + ("  <-- 已实现" if hl else ""), fontsize=12, fontweight="bold",
                color="white" if hl else COLORS["text"])
        ax.text(7.5, y + 0.25, desc, fontsize=9, color="white" if hl else COLORS["text_light"])

    save_fig(fig, FIGURES / "15_技术工具与项目模块.png")


def fig_multimodal_architecture():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_dark"])
    ax.text(6.5, 6.5, "多模态融合算法架构", ha="center", fontsize=16, fontweight="bold", color="white")

    boxes = [
        (0.5, 4.5, "压力阵列\n8x8", COLORS["primary"]),
        (0.5, 2.8, "IMU\n三轴", COLORS["primary"]),
        (0.5, 1.1, "体征\nBP/HR/SpO2", "#10B981"),
        (3.2, 3.5, "特征工程\n时域/频域/小波/样本熵", COLORS["secondary"]),
        (6.5, 3.5, "Stacking\nRF+GBDT+MLP", COLORS["accent"]),
        (9.5, 4.2, "五类场景\n识别", "#F472B6"),
        (9.5, 2.2, "决策引擎\n黄橙红", COLORS["danger"]),
        (12.0, 3.2, "App\n软干预", "white"),
    ]
    for x, y, txt, c in boxes:
        fc = COLORS["bg_card"] if c != "white" else COLORS["bg_card"]
        tc = "white" if c != "white" else COLORS["primary"]
        ec = c if c != "white" else "white"
        rect = FancyBboxPatch((x, y), 2.2, 1.2, boxstyle="round,pad=0.03", facecolor=fc, edgecolor=ec, lw=2)
        ax.add_patch(rect)
        ax.text(x + 1.1, y + 0.6, txt, ha="center", va="center", fontsize=9, color=tc, fontweight="bold")

    for (x1, y1), (x2, y2) in [((2.7, 5.1), (3.2, 4.2)), ((2.7, 3.4), (3.2, 3.8)), ((2.7, 1.7), (3.2, 3.2)),
                                  ((5.4, 4.1), (6.5, 4.1)), ((8.7, 4.8), (9.5, 4.8)), ((8.7, 2.8), (9.5, 2.8)),
                                  ((11.7, 4.2), (12.0, 3.8))]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))
    save_fig(fig, FIGURES / "16_多模态融合架构图.png")


def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax.text(7, 7.55, "居家老年人跌倒全周期智能监护系统", ha="center", fontsize=17, fontweight="bold", color=COLORS["bg_dark"])

    modules = [
        (0.8, 5.2, 3.8, 1.6, "A 日常监测", "血压·心率·血氧\n慢病异常预警", "#10B981"),
        (5.1, 5.2, 3.8, 1.6, "B 跌倒前", "倾向识别·分场景预警\n可自救/突发无意识", "#F97316"),
        (9.4, 5.2, 3.8, 1.6, "C 跌倒后", "120·家属·周围人\n伦理求助辅助", COLORS["danger"]),
        (3.5, 2.0, 7, 2.2, "感知与算法层", "柔性足底压力 + IMU + 体征  |  小波/频谱特征  |  多模态融合分类", COLORS["secondary"]),
        (4.5, 0.4, 5, 1.2, "App / 小程序", "分级软干预 · 紧急协议 · 健康周报", COLORS["bg_dark"]),
    ]
    for x, y, w, h, t, s, c in modules:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                               facecolor="white", edgecolor=c, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.35, t, ha="center", fontsize=12, fontweight="bold", color=c)
        ax.text(x + w / 2, y + h / 2 - 0.15, s, ha="center", fontsize=9, color=COLORS["text"], linespacing=1.5)
    save_fig(fig, FIGURES / "01_系统架构图.png")


def fig_pressure_heatmap_grid():
    scenarios = list(SCENARIO_LABELS.keys())
    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.25)
    fig.suptitle("五类场景典型足底压力分布（柔性 8×8 阵列仿真）", fontsize=15, fontweight="bold", y=0.98)
    vmax = 2.5
    for idx, scen in enumerate(scenarios):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        seq, _ = generate_sequence(scen, 70)
        frame = seq[len(seq) // 2]
        im = ax.imshow(frame, cmap=CMAP_PRESSURE, vmin=0, vmax=vmax)
        title = LABEL_NAMES[SCENARIO_LABELS[scen]]
        ax.set_title(title, fontsize=12, fontweight="bold", color=SCENARIO_COLORS.get(title, COLORS["primary"]))
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cax, label="归一化压力")
    save_fig(fig, FIGURES / "02_五场景压力热力图.png")


def fig_pressure_3d_surface():
    from mpl_toolkits.mplot3d import Axes3D
    seq, _ = generate_sequence("sudden_fall", 50)
    frame = seq[40]
    fig = plt.figure(figsize=(8, 5.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.arange(GRID), np.arange(GRID))
    surf = ax.plot_surface(X, Y, frame, cmap=CMAP_PRESSURE, linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title("突发跌倒瞬间 — 压力三维分布", fontsize=13, fontweight="bold", pad=16)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("压力")
    fig.colorbar(surf, shrink=0.5, label="a.u.")
    save_fig(fig, FIGURES / "17_压力三维曲面.png")


def fig_wavelet_spectrogram():
    seq, _ = generate_sequence("risk_gait", 90)
    total = seq.reshape(len(seq), -1).sum(axis=1)
    import pywt
    coeffs, freqs = pywt.cwt(total, np.arange(1, 32), "morl", sampling_period=0.1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1.2]})
    axes[0].plot(total, color=COLORS["primary"], lw=1.5)
    axes[0].set_title("跌倒倾向场景 — 足底总压力时序", fontweight="bold")
    axes[0].grid(alpha=0.3)
    im = axes[1].imshow(np.abs(coeffs), aspect="auto", cmap="magma", origin="lower",
                        extent=[0, len(total) / 10, freqs[-1], freqs[0]])
    axes[1].set_title("Morlet 小波时频图", fontweight="bold")
    axes[1].set_xlabel("时间 (s)"); axes[1].set_ylabel("尺度")
    fig.colorbar(im, ax=axes[1], label="能量")
    save_fig(fig, FIGURES / "18_小波时频分析.png")


def fig_cop_trajectories():
    fig, ax = plt.subplots(figsize=(10, 7))
    for scen in ["normal", "risk_gait", "stumble_recovery", "sudden_fall"]:
        seq, _ = generate_sequence(scen, 80)
        cop = cop_trajectory(seq)
        title = LABEL_NAMES[SCENARIO_LABELS[scen]]
        c = SCENARIO_COLORS.get(title, COLORS["primary"])
        ax.plot(cop[:, 0], cop[:, 1], lw=2.5, label=title, color=c, alpha=0.88)
        ax.scatter(cop[0, 0], cop[0, 1], s=80, c=c, marker="o", edgecolors="white", zorder=5)
        ax.scatter(cop[-1, 0], cop[-1, 1], s=120, c=c, marker="*", edgecolors="white", zorder=5)
    ax.set_xlabel("COP 横向"); ax.set_ylabel("COP 纵向")
    ax.set_title("压力中心轨迹对比（生物力学可解释性）", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.35); ax.set_aspect("equal")
    save_fig(fig, FIGURES / "03_COP轨迹对比.png")


def fig_ml_results(bundle):
    cm = bundle["confusion"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5,
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
                cbar_kws={"label": "样本数"}, annot_kws={"size": 11})
    ax.set_xlabel("预测", fontsize=11); ax.set_ylabel("真实", fontsize=11)
    ax.set_title(f"混淆矩阵 — 多模态融合模型\n准确率 {bundle['accuracy']:.1%}  |  宏 F1 {bundle['f1_macro']:.3f}",
                 fontweight="bold", pad=12)
    save_fig(fig, FIGURES / "04_混淆矩阵.png")

    imp = bundle["feature_importance"]
    names = bundle["feature_names"][: len(imp)]
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(order)))
    ax.barh([names[i] for i in order], imp[order], color=colors, height=0.7)
    ax.set_xlabel("重要性"); ax.set_title("Top-15 特征重要性（随机森林）", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, FIGURES / "05_特征重要性.png")

    y_test, y_proba, y_bin = bundle["y_test"], bundle["y_proba"], bundle["y_bin"]
    n_classes = bundle["n_classes"]
    fig, ax = plt.subplots(figsize=(9, 7))
    from sklearn.metrics import roc_curve, auc as sk_auc
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, label=f"{LABEL_NAMES[i]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("多分类 ROC (One-vs-Rest)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.35)
    save_fig(fig, FIGURES / "06_ROC曲线.png")

    # PR 曲线（宏平均展示一类）
    fig, ax = plt.subplots(figsize=(9, 6))
    from sklearn.metrics import precision_recall_curve, average_precision_score
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(r, p, lw=2, label=f"{LABEL_NAMES[i]} AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall 曲线", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.35)
    save_fig(fig, FIGURES / "19_PR曲线.png")

    # 性能卡片
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    metrics = [
        ("准确率", f"{bundle['accuracy']:.1%}"),
        ("宏 F1", f"{bundle['f1_macro']:.3f}"),
        ("特征维度", f"{len(bundle['feature_names'])}"),
        ("样本量", f"{len(bundle['y'])}"),
    ]
    for i, (k, v) in enumerate(metrics):
        x = 0.08 + i * 0.23
        rect = FancyBboxPatch((x, 0.15), 0.2, 0.7, boxstyle="round,pad=0.03", transform=ax.transAxes,
                               facecolor="white", edgecolor=COLORS["primary"], lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.1, 0.58, v, ha="center", transform=ax.transAxes, fontsize=20, fontweight="bold", color=COLORS["primary"])
        ax.text(x + 0.1, 0.28, k, ha="center", transform=ax.transAxes, fontsize=10)
    save_fig(fig, FIGURES / "07_模型性能卡片.png")


def fig_ablation(bundle):
    ab = bundle["ablation"]
    names = [ab[k]["name"] for k in ab]
    accs = [ab[k]["accuracy"] for k in ab]
    f1s = [ab[k]["f1"] for k in ab]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    w = 0.35
    b1 = ax.bar(x - w / 2, accs, w, label="准确率", color=COLORS["primary"], edgecolor="white")
    b2 = ax.bar(x + w / 2, f1s, w, label="宏 F1", color=COLORS["secondary"], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("消融实验：多模态融合 vs 基线", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.35)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.1%}", ha="center", fontsize=9)
    save_fig(fig, FIGURES / "20_消融实验对比.png")


def fig_learning_curve(bundle):
    ts, tr, val = bundle["learning_curve"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ts, tr.mean(axis=1), "o-", color=COLORS["primary"], lw=2, label="训练准确率")
    ax.fill_between(ts, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1), alpha=0.15, color=COLORS["primary"])
    ax.plot(ts, val.mean(axis=1), "s-", color=COLORS["accent"], lw=2, label="验证准确率")
    ax.fill_between(ts, val.mean(1) - val.std(1), val.mean(1) + val.std(1), alpha=0.15, color=COLORS["accent"])
    ax.set_xlabel("训练样本比例"); ax.set_ylabel("准确率")
    ax.set_title("学习曲线（5折交叉验证）", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.35)
    save_fig(fig, FIGURES / "21_学习曲线.png")


def fig_decision_flowchart():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax.text(6.5, 8.5, "分场景决策引擎（闭环软干预）", ha="center", fontsize=16, fontweight="bold")
    nodes = [
        (6.5, 7.3, "多模态输入\n压力+IMU+体征", "#E0F2FE"),
        (6.5, 6.0, "融合分类器", COLORS["secondary"]),
        (2.5, 4.5, "正常/倾向\n黄预警", "#FEF9C3"),
        (6.5, 4.5, "失稳-可自救\n橙预警", "#FFEDD5"),
        (10.5, 4.5, "突发跌倒\n红预警", "#FEE2E2"),
        (2.5, 2.5, "健康建议\n家属关注", "white"),
        (6.5, 2.5, "语音扶稳\n30s升级", "white"),
        (10.5, 2.5, "120+联系人\n周围人模式", "white"),
        (6.5, 0.8, "App 执行", COLORS["primary"]),
    ]
    for x, y, t, fc in nodes:
        tc = "white" if fc in (COLORS["secondary"], COLORS["primary"]) else COLORS["text"]
        rect = FancyBboxPatch((x - 1.3, y - 0.45), 2.6, 0.9, boxstyle="round,pad=0.02",
                               facecolor=fc if isinstance(fc, str) and fc.startswith("#") else fc,
                               edgecolor=COLORS["text_light"], lw=1.5)
        ax.add_patch(rect)
        ax.text(x, y, t, ha="center", va="center", fontsize=9, fontweight="bold", color=tc)
    save_fig(fig, FIGURES / "08_决策引擎流程图.png")


def fig_emergency_timeline():
    timeline = emergency_timeline()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-2, 72); ax.set_ylim(0, 3); ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax.text(35, 2.85, "跌倒后紧急协议时间线", ha="center", fontsize=15, fontweight="bold")
    for item in timeline:
        t, ev, icon = item["t"], item["event"], item["icon"]
        c = COLORS["danger"] if t >= 60 else COLORS["primary"]
        ax.add_patch(Circle((t, 1.25), 2.0, facecolor=c, edgecolor="white", lw=2, alpha=0.92))
        ax.text(t, 1.25, icon, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(t, 0.35, f"+{t}s", ha="center", fontsize=8, color=COLORS["text_light"])
        ax.text(t, -0.05, ev, ha="center", fontsize=8, fontweight="bold", rotation=20)
    ax.annotate("", xy=(68, 1.25), xytext=(0, 1.25), arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=2))
    save_fig(fig, FIGURES / "09_紧急协议时间线.png")


def fig_vitals_dashboard():
    v = simulate_vitals_days(21, "fall_risk")
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, hspace=0.38, wspace=0.28)
    fig.suptitle("模块 A · 日常体征监测与异常预警", fontsize=15, fontweight="bold")
    plots = [
        (gs[0, 0], v["days"], v["sbp"], "收缩压 mmHg", COLORS["danger"], 90, 140),
        (gs[0, 1], v["days"], v["hr"], "心率 bpm", COLORS["secondary"], 60, 100),
        (gs[1, 0], v["days"], v["spo2"], "血氧 %", COLORS["primary"], 92, 100),
        (gs[1, 1], v["days"], v["dbp"], "舒张压 mmHg", COLORS["accent"], 60, 90),
    ]
    for spec, x, y, title, color, lo, hi in plots:
        ax = fig.add_subplot(spec)
        ax.fill_between(x, y, alpha=0.2, color=color)
        ax.plot(x, y, "o-", color=color, lw=2.2, markersize=4, markeredgecolor="white", markeredgewidth=0.8)
        ax.axhspan(lo, hi, alpha=0.12, color=COLORS["success"])
        last = check_vitals(v["sbp"][-1], v["dbp"][-1], v["hr"][-1], v["spo2"][-1])
        if last["alerts"]:
            ax.scatter([x[-1]], [y[-1]], s=280, c=COLORS["danger"], marker="*", zorder=5, edgecolors="white", lw=1.5)
        ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.35)
    save_fig(fig, FIGURES / "10_体征监测仪表盘.png")


def fig_app_mockup():
    screens = [
        ("首页", ["今日风险: 中", "血压 132/80", "心率 78", ""], COLORS["primary"], False),
        ("橙警 · 可自救", ["失稳预警", "请扶稳或坐下", "30s 后通知家属"], "#EA580C", False),
        ("红警 · 跌倒", ["紧急警报", "60s 后拨打 120", ""], COLORS["danger"], True),
        ("周围人求助", ["授权求助码", "录像已开启", "请协助联系120"], COLORS["secondary"], False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
    fig.patch.set_facecolor(COLORS["bg_light"])
    for ax, (title, lines, color, sos) in zip(axes, screens):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.06, 0.04), 0.88, 0.92, boxstyle="round,pad=0.02",
                                    facecolor=COLORS["bg_dark"], edgecolor="#475569", lw=3))
        ax.add_patch(FancyBboxPatch((0.08, 0.8), 0.84, 0.12, boxstyle="round,pad=0.01", facecolor=color, edgecolor="none"))
        ax.text(0.5, 0.86, title, ha="center", color="white", fontsize=11, fontweight="bold")
        for i, line in enumerate(lines):
            if line:
                ax.text(0.5, 0.68 - i * 0.14, line, ha="center", color="white", fontsize=9)
        if sos:
            ax.add_patch(Circle((0.5, 0.18), 0.09, facecolor=COLORS["danger"], edgecolor="white", lw=2))
            ax.text(0.5, 0.18, "SOS", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    fig.suptitle("App / 小程序界面原型", fontsize=14, fontweight="bold", y=1.0)
    save_fig(fig, FIGURES / "11_App界面原型.png")


def fig_ethics():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg_light"])
    ax.text(0.5, 0.94, "「扶不扶」伦理困境与系统应对", ha="center", fontsize=16, fontweight="bold", transform=ax.transAxes)
    for i, t in enumerate(["路人顾虑：责任不清", "不敢扶 · 延误救援", "缺乏病史信息"]):
        ax.text(0.08, 0.72 - i * 0.11, "[问题] " + t, fontsize=11, color=COLORS["danger"], transform=ax.transAxes)
    for i, t in enumerate(["授权求助二维码", "录音录像时间戳", "规范急救图示", "现场语音说明"]):
        ax.text(0.52, 0.72 - i * 0.11, "[系统] " + t, fontsize=11, color=COLORS["success"], transform=ax.transAxes)
    rect = FancyBboxPatch((0.12, 0.08), 0.76, 0.16, boxstyle="round,pad=0.02", transform=ax.transAxes,
                           facecolor="white", edgecolor=COLORS["secondary"], lw=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.16, "目标：降低协助风险 · 缩短救援时间 · 不替代专业急救",
            ha="center", fontsize=10, transform=ax.transAxes, color=COLORS["text"])
    save_fig(fig, FIGURES / "12_扶不扶伦理与系统应对.png")


def fig_scenario_comparison(bundle):
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("五类场景实时识别（多模态融合模型）", fontsize=14, fontweight="bold")
    profiles = ["normal", "fall_risk", "fall_risk", "hypertension", "normal"]
    for ax, scen, prof in zip(axes, SCENARIO_LABELS.keys(), profiles):
        seq, imu = generate_sequence(scen, 75)
        pred, proba, _ = predict_from_sequence(seq, imu, bundle, prof)
        true_name = LABEL_NAMES[SCENARIO_LABELS[scen]]
        pred_name = LABEL_NAMES[pred]
        total = seq.reshape(len(seq), -1).sum(axis=1)
        c = SCENARIO_COLORS.get(true_name, COLORS["primary"])
        ax.plot(total, color=c, lw=1.8); ax.fill_between(range(len(total)), total, alpha=0.18, color=c)
        ok = "OK" if pred_name == true_name else "~"
        ax.set_ylabel(true_name, fontsize=9)
        ax.text(0.99, 0.82, f"预测: {pred_name} {ok}  {proba.max():.0%}", transform=ax.transAxes, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95), fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("时间帧")
    save_fig(fig, FIGURES / "13_五场景识别对比.png")


def fig_risk_gauge():
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})
    for lo, hi, color in [(0, 0.33, "#10B981"), (0.33, 0.66, "#F59E0B"), (0.66, 1.0, "#EF4444")]:
        t = np.linspace(np.pi * (1 - hi), np.pi * (1 - lo), 40)
        ax.fill_between(t, 0.55, 1.0, color=color, alpha=0.75)
    ax.annotate("中", xy=(np.pi * 0.62, 0.72), fontsize=32, fontweight="bold", ha="center", color="#F59E0B")
    ax.set_title("今日跌倒风险仪表盘", fontweight="bold", pad=18, fontsize=13)
    ax.set_yticklabels([]); ax.set_xticklabels([])
    save_fig(fig, FIGURES / "14_风险等级仪表盘.png")


def gif_pressure(scenario, filename, n=55):
    seq, _ = generate_sequence(scenario, n)
    vmax = float(np.percentile(seq, 98)) + 0.1
    frames = []
    cmap = CMAP_PRESSURE
    title = LABEL_NAMES[SCENARIO_LABELS[scenario]]
    step = max(1, n // 45)
    for i in range(0, n, step):
        fig, ax = plt.subplots(figsize=(4.5, 4.8), dpi=90, facecolor=COLORS["bg_dark"])
        im = ax.imshow(seq[i], cmap=cmap, vmin=0, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"{title}  |  帧 {i+1}/{n}", color="white", fontsize=13, fontweight="bold", pad=10)
        txt = ax.text(0.02, 0.02, "8x8 Flexible Pressure Sensor", transform=ax.transAxes,
                      color="#94A3B8", fontsize=8)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground=COLORS["bg_dark"])])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white")
        cb.outline.set_edgecolor("#475569")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    path = GIFS / filename
    imageio.mimsave(str(path), frames, duration=0.1, loop=0)
    print(f"  saved: {path}")


def generate_all_figures(bundle):
    from poster import fig_poster
    print("\n[可视化] 生成项目图表...")
    fig_poster()
    fig_tools_and_difficulty()
    fig_multimodal_architecture()
    fig_system_architecture()
    fig_pressure_heatmap_grid()
    fig_pressure_3d_surface()
    fig_wavelet_spectrogram()
    fig_cop_trajectories()
    fig_ml_results(bundle)
    fig_ablation(bundle)
    fig_learning_curve(bundle)
    fig_decision_flowchart()
    fig_emergency_timeline()
    fig_vitals_dashboard()
    fig_app_mockup()
    fig_ethics()
    fig_scenario_comparison(bundle)
    fig_risk_gauge()

    print("\n[可视化] 生成 GIF...")
    gif_pressure("normal", "01_正常行走.gif")
    gif_pressure("risk_gait", "02_跌倒倾向.gif")
    gif_pressure("stumble_recovery", "03_失稳可自救.gif")
    gif_pressure("sudden_fall", "04_突发跌倒.gif")
    gif_pressure("squat", "05_蹲下坐下.gif")
