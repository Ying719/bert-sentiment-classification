# -*- coding: utf-8 -*-
"""
prepare_data.py —— 第一步：准备数据

运行方式：
    cd v2/src
    python prepare_data.py

它做三件事：
    1. 清洗原始数据（去重、过滤）
    2. 切成 train / val / test 三份并落盘到 v2/data/
    3. 执行泄漏自检并把报告写进 split_report.json

这个脚本一生只需要跑一次（除非你改了数据或划分比例）。
之后 train.py 和 evaluate.py 都读它产出的固定文件。
"""

import argparse
import sys
from pathlib import Path

# 把 src 目录加进 import 路径，这样脚本无论从哪运行都能 import 到同目录的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG
from data import build_datasets
from utils import build_logger


def main():
    parser = argparse.ArgumentParser(description="准备并固定 train/val/test 划分")
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有的划分文件，强制重新划分（改过数据或比例时才需要）",
    )
    args = parser.parse_args()

    CFG.ensure_dirs()
    logger = build_logger("prepare", CFG.output_dir / "prepare_data.log")

    logger.info("=" * 62)
    logger.info("步骤 1/4  数据准备")
    logger.info("=" * 62)

    splits = build_datasets(force=args.force, logger=logger)

    logger.info("-" * 62)
    logger.info("数据准备完成。下一步：python train.py")


if __name__ == "__main__":
    main()
