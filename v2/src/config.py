# -*- coding: utf-8 -*-
"""
config.py —— 全局配置中心

为什么要有这个文件？
    旧代码把 BATCH_SIZE、EPOCHS 这些参数散落在每个脚本的顶部，
    改一次参数要翻好几个文件，还容易改漏。
    现在所有可调参数集中在这里，其他文件统一 `from config import CFG`。

改动这个文件只需要注意一件事：
    凡是带"（可调）"注释的项，都可以放心改；
    带"（勿改）"注释的项改了会让已保存的模型失效。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 先设置 HuggingFace 国内镜像。
# 必须在 import transformers 之前设置，否则会去连 huggingface.co 而超时。
# 这是原代码就有的处理，保留。
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 离线模式：bert-base-chinese 的缓存四件套（config/vocab/tokenizer_config/权重）
# 已经下载完整，强制离线加载可以避免 huggingface_hub 联网做 etag 检查——
# 2026-10-02 实测镜像无响应时会无限挂起（卡了 4 分钟 GPU 占用 0%）。
#
# ⚠️ 换用新的预训练模型（如 hfl/chinese-macbert-base）时：
#    先把下面两行注释掉 → 跑一次让它从镜像下载 → 下载完再恢复这两行。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 项目根目录（v2 的上一级，也就是原项目目录）
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ======================= 一、路径配置 =======================
    # 原始数据。注意：第一版的全部内容已归档到 V1/ 目录下，
    # 数据文件现在位于 V1/data/，所以这里指向那儿（不拷贝，直接读，避免占空间）。
    raw_large_data: Path = ROOT / "V1" / "data" / "large_data.csv"   # 10800 条，含重复
    raw_small_data: Path = ROOT / "V1" / "data" / "data.csv"         # 102 条，人工构造的小样本

    # v2 自己的目录
    data_dir: Path = ROOT / "v2" / "data"               # 固定划分后的数据落在这里
    output_dir: Path = ROOT / "v2" / "outputs"          # 训练产物（模型/日志/图）
    model_dir: Path = ROOT / "v2" / "outputs" / "model"  # 最终模型

    # 第一版的模型（已随 V1 归档）保留用于对比，不删
    legacy_model_dir: Path = ROOT / "V1" / "model" / "sentiment_model"

    # ======================= 二、数据配置 =======================
    # （可调）文本最大长度。中文评论绝大多数在 128 字以内，128 足够。
    # 调大会变慢且几乎无收益，调小可能截断长评论。
    max_len: int = 128

    # （可调）最短文本长度：少于该字数的样本直接丢弃（噪声样本）
    min_len: int = 4

    # （可调）三路划分比例，三者之和必须为 1.0
    # train 用于训练，val 用于早停和选模型，test 用于最终汇报（只跑一次！）
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # （勿改）固定随机种子。固定后每次划分出的数据集完全一致，
    # 保证实验可复现——这是论文/答辩要求的基本素养。
    seed: int = 42

    # ======================= 三、模型配置 =======================
    # （可调）预训练模型名称。bert-base-chinese 是 1.02 亿参数的通用中文 BERT。
    # 换成更强模型的两个推荐（想再提升几个点就改这里）：
    #   "hfl/chinese-roberta-wwm-ext"   —— 哈工大全词掩码 RoBERTa，通常 +1~2 个点
    #   "hfl/chinese-macbert-base"      —— 哈工大 MacBERT，纠错式预训练，通常 +1~2 个点
    # 注意：换模型后必须重新训练，旧权重不能用。
    model_name: str = "bert-base-chinese"

    # （勿改）分类类别数。好评/差评 二分类。
    num_labels: int = 2

    # ======================= 四、训练超参数 =======================
    # （可调）批次大小。显存不够就往下调（8 或 4），显存够就往上调（32）。
    # 经验：BERT-base 每样本约吃 0.3~0.5GB 显存（max_len=128, fp16）。
    batch_size: int = 32

    # （可调）评估/推理时的批次大小。推理不给梯度，可以开得比训练大得多。
    # 这就是"分批推理提速"的关键——旧代码每条跑一次前向，这里一次跑 128 条。
    eval_batch_size: int = 128

    # （可调）训练轮数。BERT 微调通常 3~5 轮。
    # 已经配了早停，所以设大一点也没关系（比如 10），它会自己停。
    epochs: int = 5

    # （可调）学习率。BERT 微调的黄金区间是 1e-5 ~ 5e-5。
    # 太大（>1e-4）会训崩，太小（<1e-5）收敛不动。
    learning_rate: float = 2e-5

    # （可调）权重衰减，抑制过拟合。0.01 是 BERT 微调的常用值。
    weight_decay: float = 0.01

    # （可调）学习率预热比例：前 10% 的步数里学习率从 0 线性升到 2e-5。
    # 作用是不让模型一开始就被大梯度冲乱——这是旧代码完全没有的。
    warmup_ratio: float = 0.1

    # （可调）梯度累积步数。真实批次 = batch_size × grad_accum_steps。
    # 显存不够时可以 batch_size=8 + grad_accum_steps=4，等效于 batch_size=32。
    grad_accum_steps: int = 1

    # （可调）早停耐心值：验证集指标连续 N 轮不提升就停止训练。
    early_stopping_patience: int = 2

    # （可调）是否启用混合精度训练（FP16）。
    # 有 NVIDIA 显卡时开启可提速约 2 倍并省一半显存；CPU 上自动忽略。
    use_fp16: bool = True

    # （可调）梯度裁剪阈值，防止梯度爆炸。1.0 是标准值。
    max_grad_norm: float = 1.0

    # ======================= 五、推理与校准配置 =======================
    # （可调）判定"情感不明显"的置信度阈值。
    # 注意与旧代码的区别：旧代码用 "置信度≥85% 且 概率差≥0.5" 两个魔改条件，
    # 这里只保留一个有统计学意义的阈值，且它对应的分位数会在评估阶段算出来。
    #
    # 取值依据（evaluate.py 输出的覆盖率-准确率表）：
    #   阈值 0.80 -> 覆盖 94.9% 样本，准确率 93.29%
    #   阈值 0.90 -> 覆盖 90.8% 样本，准确率 94.31%
    #   阈值 0.95 -> 覆盖 83.6% 样本，准确率 95.80%   <- 采用这一档
    # 含义：只对最有把握的 83.6% 样本给结论，此时准确率 95.80%；
    #       剩下 16.4% 交由人工复核。这是"选择性预测"，比硬答更可信。
    #
    # 已知取舍：极短文本（如"太贵了"）置信度约 0.947，会落入"情感不明显"。
    # 这是有意为之 —— 短文本信息量少，说"不确定"比硬判更谨慎。
    # 若要覆盖更多短评可下调到 0.90，但整体准确率会降到 94.31%。
    uncertain_threshold: float = 0.95

    # （可调）是否启用温度缩放校准（Temperature Scaling）。
    # 未校准的神经网络普遍"过度自信"——明明只有 70% 把握却报 99%。
    # 温度缩放用一个标量 T 把置信度拉到符合真实准确率。
    # T 由 evaluate.py 在验证集上拟合，写入 calibration.json。
    use_calibration: bool = True

    # ======================= 六、绘图中文字体 =======================
    # Windows 自带黑体，避免中文显示成方块
    font_family: str = "SimHei"

    def ensure_dirs(self) -> None:
        """确保所有输出目录存在。任何脚本启动时都应先调用它。"""
        for d in (self.data_dir, self.output_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 以下四个属性是"固定划分文件"的路径。
    # 这些文件只生成一次（由 prepare_data.py），之后 train/val/test
    # 全都读它们——这样就从流程上彻底杜绝了"训练集和测试集串味"。
    # ------------------------------------------------------------------
    @property
    def train_file(self) -> Path:
        return self.data_dir / "train.csv"

    @property
    def val_file(self) -> Path:
        return self.data_dir / "val.csv"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "test.csv"

    @property
    def split_report(self) -> Path:
        return self.data_dir / "split_report.json"


# 全局单例。其他文件统一 from config import CFG
CFG = Config()
