#!/usr/bin/env python3
"""一键运行合成数据训练、评估和可视化流程。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import FIGURES, GIFS, DATA  # noqa: E402
from models import train_and_evaluate  # noqa: E402
from visuals import generate_all_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEMS足底压力步态监测与跌倒风险预警仿真"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=220,
        help="每类参数化合成样本数，默认220",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="只训练和评估，不重新生成图表与GIF",
    )
    parser.add_argument(
        "--no-model-file",
        action="store_true",
        help="不保存pickle模型文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 10:
        raise SystemExit("--samples-per-class 必须至少为10")

    started = time.time()
    print("=" * 64)
    print("MEMS足底压力步态监测与跌倒风险预警仿真")
    print(f"每类合成样本数: {args.samples_per_class}")
    print("注意: 所有数据均为参数化合成数据，不代表临床性能")
    print("=" * 64)

    bundle = train_and_evaluate(
        n_per_class=args.samples_per_class,
        save_model=not args.no_model_file,
    )

    if not args.skip_figures:
        generate_all_figures(bundle)

    summary = {
        "data_type": "parameterized synthetic data",
        "samples_per_class": args.samples_per_class,
        "total_samples": args.samples_per_class * 5,
        "feature_count": len(bundle["feature_names"]),
        "test_accuracy": float(bundle["accuracy"]),
        "macro_f1": float(bundle["f1_macro"]),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n运行完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.skip_figures:
        print(f"图表目录: {FIGURES}")
        print(f"GIF目录: {GIFS}")


if __name__ == "__main__":
    main()
