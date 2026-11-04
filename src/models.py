# -*- coding: utf-8 -*-
"""多模态融合模型：Stacking、MLP与可选PyTorch 1D-CNN。"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline

from config import DATA, RANDOM_SEED
from simulator import build_dataset, LABEL_NAMES, generate_sequence, SCENARIO_LABELS
from features_advanced import extract_advanced_features
from vitals import simulate_vitals_days, check_vitals


def _vitals_feature_vector(profile: str = "normal") -> np.ndarray:
    v = simulate_vitals_days(7, profile)
    last = check_vitals(v["sbp"][-1], v["dbp"][-1], v["hr"][-1], v["spo2"][-1])
    return np.array([
        v["sbp"][-1], v["dbp"][-1], v["hr"][-1], v["spo2"][-1],
        v["sbp"].std(), v["hr"].std(),
        float(last["bp_alert"]), float(last["hr_alert"]), float(last["spo2_alert"]),
    ], dtype=np.float32)


PROFILE_MAP = {0: "normal", 1: "fall_risk", 2: "fall_risk", 3: "hypertension", 4: "normal"}


def build_multimodal_dataset(n_per_class: int = 160) -> tuple:
    Xg, y = build_dataset(n_per_class=n_per_class, hard=True)
    Xv = np.array([_vitals_feature_vector(PROFILE_MAP[int(l)]) for l in y])
    # 加入体征随机扰动
    rng = np.random.default_rng(RANDOM_SEED)
    Xv += rng.normal(0, 0.5, Xv.shape)
    return np.hstack([Xg, Xv]), y, Xg, Xv


def _make_stacking():
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=14, class_weight="balanced", random_state=RANDOM_SEED)),
        ("gb", GradientBoostingClassifier(n_estimators=120, max_depth=5, random_state=RANDOM_SEED)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=250, early_stopping=True,
                              random_state=RANDOM_SEED, learning_rate_init=0.001)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_SEED),
        cv=3,
        passthrough=False,
    )


def _train_single(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", _make_stacking())])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)
    return {
        "name": name,
        "pipeline": pipe,
        "X_test": X_te,
        "y_test": y_te,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_te, y_pred),
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
    }


def _try_pytorch_cnn(X, y):
    """可选 1D-CNN（若已安装 PyTorch）"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        return None

    device = torch.device("cpu")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)
    sc = StandardScaler()
    X_trs = sc.fit_transform(X_tr)
    X_tes = sc.transform(X_te)

    class CNN1D(nn.Module):
        def __init__(self, n_in, n_out=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
                nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_out),
            )

        def forward(self, x):
            return self.net(x)

    model = CNN1D(X_trs.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_trs[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    model.train()
    losses = []
    for epoch in range(40):
        ep_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader))

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(X_tes[:, None, :], dtype=torch.float32)), dim=1).numpy()
        pred = proba.argmax(1)

    return {
        "name": "1D-CNN (PyTorch)",
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "y_test": y_te,
        "y_pred": pred,
        "y_proba": proba,
        "train_losses": losses,
        "scaler": sc,
        "model": model,
    }


def train_and_evaluate(n_per_class: int = 220, save_model: bool = True):
    """训练并评估多模态模型。

    参数化合成数据用于验证软件链路，不代表真实临床性能。
    """
    print("  构建多模态数据集（压力+IMU小波 + 体征）...")
    X_mm, y, X_gait, X_vit = build_multimodal_dataset(n_per_class=n_per_class)

    results = {}
    results["rf_baseline"] = _train_single(X_gait, y, "随机森林(基础特征)")
    results["fusion"] = _train_single(X_mm, y, "多模态融合+Stacking+MLP")

    cnn_res = _try_pytorch_cnn(X_mm, y)
    if cnn_res:
        results["cnn"] = cnn_res
        print("  PyTorch 1D-CNN 已训练")
    else:
        print("  PyTorch 未安装，跳过可选的 1D-CNN 对比实验")

    main = results["fusion"]
    pipe = main["pipeline"]
    clf = pipe.named_steps["clf"]
    rf = clf.named_estimators_["rf"]

    # 学习曲线
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_mm, y, cv=3, n_jobs=1,
        train_sizes=np.linspace(0.3, 1.0, 5), scoring="accuracy",
    )

    y_test = main["y_test"]
    y_proba = main["y_proba"]
    n_classes = len(LABEL_NAMES)
    y_bin = label_binarize(y_test, classes=range(n_classes))

    bundle = {
        "main_model": pipe,
        "results": results,
        "X_mm": X_mm,
        "y": y,
        "feature_importance": rf.feature_importances_,
        "feature_names": _feature_names(),
        "accuracy": main["accuracy"],
        "f1_macro": main["f1_macro"],
        "report": classification_report(y_test, main["y_pred"], target_names=LABEL_NAMES, output_dict=True),
        "confusion": confusion_matrix(y_test, main["y_pred"]),
        "y_test": y_test,
        "y_pred": main["y_pred"],
        "y_proba": y_proba,
        "y_bin": y_bin,
        "n_classes": n_classes,
        "learning_curve": (train_sizes, train_scores, val_scores),
        "ablation": {k: {"accuracy": v["accuracy"], "f1": v["f1_macro"], "name": v["name"]} for k, v in results.items()},
    }

    if save_model:
        with open(DATA / "model_bundle.pkl", "wb") as f:
            pickle.dump(bundle, f)
    return bundle


def _feature_names():
    base = ["平均压力", "压力波动", "不对称均值", "不对称峰值", "COP速度均值", "COP速度峰值",
            "步频", "冲击峰值", "静态稳定性", "恢复评分"]
    adv = ["样本熵", "Hilbert幅值", "偏度", "峰度", "COP速度波动",
           "谱峰频", "谱熵", "谱质心", "小波E1", "小波E2", "小波E3", "小波E4",
           "IMU均值", "IMU波动", "IMU峰值", "IMU主频"]
    vit = ["收缩压", "舒张压", "心率", "血氧", "血压波动", "心率波动", "血压告警", "心率告警", "血氧告警"]
    return base + adv + vit


def predict_from_sequence(seq, imu, bundle, profile="normal") -> tuple:
    v = _vitals_feature_vector(profile)
    g = extract_advanced_features(seq, imu)
    vec = np.hstack([g, v]).reshape(1, -1)
    pipe = bundle["main_model"]
    pred = pipe.predict(vec)[0]
    proba = pipe.predict_proba(vec)[0]
    return int(pred), proba, g
