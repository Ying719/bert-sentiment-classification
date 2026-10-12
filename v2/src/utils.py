# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具函数

放三类东西：
  1. set_seed()      固定随机种子，让实验可复现
  2. get_device()    自动选择 GPU / CPU
  3. Logger          同时打印到屏幕和写进文件的日志器
  4. get_tokenizer() 统一的 tokenizer 加载入口
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from config import CFG


# ---------------------------------------------------------------------------
# 1. 随机种子固化
# ---------------------------------------------------------------------------
def set_seed(seed: int = None) -> None:
    """
    固定 python / numpy / torch 三层的随机源。

    为什么必须做这件事？
        BERT 微调里有大量随机性：权重初始化、Dropout、数据打乱顺序。
        不固定种子的话，同一个脚本跑两次结果能差 1~2 个点，
        你就没法判断"我这次的改动到底有没有用"。
        答辩时被问"你的结果可复现吗"，这就是答案。
    """
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 下面两行让 cuDNN 使用确定性算法（速度略降，但结果可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. 设备选择
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    """返回一句人类可读的设备描述，用于日志开头。"""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {name} ({total:.1f} GB 显存)"
    return "CPU（未检测到可用 GPU，训练会较慢）"


# ---------------------------------------------------------------------------
# 3. 日志器：同时输出到控制台和文件
# ---------------------------------------------------------------------------
def build_logger(name: str, log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """
    旧代码全靠 print()，跑完之后什么都不留痕。
    这里改成同时打印和落盘，配合 outputs/ 里的图表，
    就是一份完整的实验记录——答辩和写 README 全靠它。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()          # 避免重复添加 handler 导致日志打印两遍
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    # 控制台
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 4. Tokenizer 统一入口
# ---------------------------------------------------------------------------
def get_tokenizer(model_path: str = None):
    """
    加载 BERT 分词器。优先从本地模型目录加载，取不到再回退到在线拉取。

    为什么要这样写？
        线上部署时可能没有网络，本地已保存的 vocab.txt 就能直接用；
        本地还没有模型（首次训练）时，走镜像下载。
    """
    from transformers import BertTokenizer

    model_path = model_path or str(CFG.model_name)
    try:
        return BertTokenizer.from_pretrained(model_path)
    except Exception:
        return BertTokenizer.from_pretrained(CFG.model_name)


# ---------------------------------------------------------------------------
# 5. JSON 读写小工具
# ---------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 6. 计时器（上下文管理器）
# ---------------------------------------------------------------------------
class Timer:
    """with Timer() as t: ... 之后用 t.elapsed 拿到秒数。用于给各阶段计时。"""

    def __enter__(self):
        self._start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._start
        return False

    def __str__(self):
        return f"{self.elapsed:.1f}s"
