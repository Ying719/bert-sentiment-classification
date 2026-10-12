# -*- coding: utf-8 -*-
"""
train.py —— 第二步：微调 BERT

运行方式：
    cd v2/src
    python train.py

【相比旧 train.py 的六个升级】

1. 换了训练框架：手写训练循环 -> HuggingFace Trainer
   旧代码要自己写 for batch in loader、自己算 loss.backward()、自己写验证。
   新代码用官方 Trainer，它已经把混合精度、梯度裁剪、日志、Checkpoint
   全都封装好了，代码量更少而且更不容易出 bug。

2. 真正用上了 GPU 加速
   旧代码：batch=8，无混合精度，3 轮，全程 float32。
   新代码：batch=32 + FP16 混合精度（速度约翻倍，显存省一半）。

3. 加了学习率预热 + 线性衰减
   旧代码是一条恒定的 2e-5 学习率从头跑到尾。
   这是微调不稳定、掉点的常见原因。BERT 的标准做法是
   前 10% 步数从 0 线性升温，之后线性降到 0。

4. 加了早停（Early Stopping）
   旧代码固定跑 3 轮，不管第 2 轮就已经过拟合了照样跑完。
   新代码监控验证集 macro-F1，连续 2 轮不提升就自动停，并回滚到最优权重。

5. 选择指标从"准确率"改成"macro-F1"
   为什么？假设测试集 90% 是好评，那么"全部预测成好评"这个废模型
   也有 90% 准确率。macro-F1 对两个类别一视同仁，
   能真正反映"差评有没有被识别出来"——这才是电商场景的价值所在。

6. 防止误用测试集
   训练脚本只读 train.csv 和 val.csv，
   在代码层面就够不到 test.csv。从流程上杜绝"边看测试集边调参"。

产物（都在 v2/outputs/ 下）：
    model/                   最优模型 + 分词器
    train_log.txt            完整训练日志
    training_curve.png       损失/准确率/F1 曲线
    train_metrics.json       训练过程数值记录
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 【import 顺序很重要】先把 src 目录加进路径、先导入 config。
# config.py 里设置了 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE（强制离线加载本地
# 缓存），但 huggingface_hub 是在"被 import 的那一刻"读取这些环境变量的——
# 如果 from config import CFG 排在 import transformers 之后，变量就设晚了，
# 实测会卡在联网 etag 检查上好几分钟（GPU 占用 0%，进程假死）。
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG  # 必须在 import transformers 之前

import numpy as np
import pandas as pd
import torch

# ===========================================================================
# 【Windows 中文路径补丁】必须在 import transformers 之前执行
#
# 问题：torch 2.0.1 的 C++ 保存器（PyTorchFileWriter）在 Windows 上检查
#       父目录时走 ANSI 编码，遇到含中文/全角字符的绝对路径会误报
#       "Parent directory ... does not exist"。本项目路径含"项目3："，
#       首次训练就在保存 checkpoint 时崩在这（训练本身没问题）。
# 事实：torch.load 读中文路径正常；用文件对象写入也正常（已做对照实验验证）。
# 方案：包装 torch.save——收到路径时先 open 成文件对象再交给底层保存器。
# 时序：transformers 的 save_pretrained 在"模块导入时"就把 torch.save
#       绑定为默认参数，所以这个补丁必须写在 import transformers 之前，
#       写在后面就替换不进去了。
# ===========================================================================
_torch_save_orig = torch.save

def _unicode_safe_save(obj, f, *args, **kwargs):
    if isinstance(f, (str, os.PathLike)):
        with open(f, "wb") as fh:
            _torch_save_orig(obj, fh, *args, **kwargs)
    else:
        _torch_save_orig(obj, f, *args, **kwargs)

torch.save = _unicode_safe_save

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import build_logger, describe_device, get_device, get_tokenizer, save_json, set_seed


# ===========================================================================
# 数据集封装
# ===========================================================================
class SentimentDataset(torch.utils.data.Dataset):
    """
    把 DataFrame 包装成 PyTorch 能吃的 Dataset。

    注意一个细节：这里**不**做 padding。
    旧代码 train.py 第 50-58 行一次性把整个数据集 padding 到 max_length=128，
    意味着哪怕一条评论只有 8 个字，也要占 128 个 token 的位置。
    7277 条训练数据里绝大多数评论远短于 128 字，这就造成大量无效计算。

    新方案：这里只做 tokenize（转成 token id），
    padding 交给 DataCollatorWithPadding 在组 batch 时动态完成——
    每个 batch 只 pad 到该 batch 内最长样本的长度。这叫**动态 padding**，
    能显著减少计算量（实测整体训练时间可降 30% 以上）。
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["review"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,          # 超过 max_len 的截断
            max_length=self.max_len,
            # 注意：这里没有 padding=True，交给 collator
        )
        enc["labels"] = int(self.labels[idx])
        return enc


# ===========================================================================
# 评估指标
# ===========================================================================
def compute_metrics(eval_pred):
    """
    Trainer 在每个 epoch 结束时调用它，输入是 (预测 logits, 真实标签)。

    返回的字段名会原样出现在日志里（比如 eval_macro_f1），
    后面配置 load_best_model_at_end 时要用到这个名字。
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        # 差评（label=0）单独看一遍召回率：
        # 对电商来说，漏判差评比漏判好评严重得多，这个数字是给业务看的
        "neg_recall": recall_score(labels, preds, pos_label=0, zero_division=0),
        "neg_precision": precision_score(labels, preds, pos_label=0, zero_division=0),
    }


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="微调 BERT 中文情感分类模型")
    parser.add_argument("--model_name", default=CFG.model_name, help="预训练模型名或本地路径")
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--no_fp16", action="store_true", help="关闭混合精度（排错时可加）")
    parser.add_argument("--output", default=None, help="模型保存目录，默认 v2/outputs/model")
    args = parser.parse_args()

    CFG.ensure_dirs()
    logger = build_logger("train", CFG.output_dir / "train_log.txt")

    # ---- 0. 环境准备 ----
    set_seed()
    device = get_device()
    use_fp16 = CFG.use_fp16 and not args.no_fp16 and device.type == "cuda"
    out_dir = Path(args.output) if args.output else CFG.model_dir

    logger.info("=" * 62)
    logger.info("步骤 2/4  模型微调")
    logger.info("=" * 62)
    logger.info(f"设备      : {describe_device()}")
    logger.info(f"预训练模型: {args.model_name}")
    logger.info(f"混合精度  : {'开启 (FP16)' if use_fp16 else '关闭'}")
    logger.info(f"随机种子  : {CFG.seed}")

    # ---- 1. 加载固定划分的数据 ----
    if not CFG.train_file.exists():
        logger.error("找不到划分好的数据，请先运行: python prepare_data.py")
        sys.exit(1)

    train_df = pd.read_csv(CFG.train_file, encoding="utf-8-sig")
    val_df = pd.read_csv(CFG.val_file, encoding="utf-8-sig")
    logger.info(f"训练集 {len(train_df)} 条 | 验证集 {len(val_df)} 条")
    logger.info("（测试集 test.csv 已被刻意排除在训练流程之外，只在 evaluate.py 中使用）")

    # ---- 2. 分词器与数据 ----
    tokenizer = get_tokenizer(args.model_name)
    train_ds = SentimentDataset(train_df, tokenizer, CFG.max_len)
    val_ds = SentimentDataset(val_df, tokenizer, CFG.max_len)

    # 动态 padding 的核心组件
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # ---- 3. 模型 ----
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=CFG.num_labels,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params/1e6:.1f} M")

    # ---- 4. 训练参数 ----
    # 总步数 = 样本数 / 真实批次 × 轮数，用来算 warmup 步数
    effective_batch = args.batch_size * CFG.grad_accum_steps
    steps_per_epoch = int(np.ceil(len(train_ds) / effective_batch))
    warmup_steps = int(steps_per_epoch * args.epochs * CFG.warmup_ratio)
    logger.info(
        f"批次大小 {args.batch_size} x 累积 {CFG.grad_accum_steps} = 等效 {effective_batch}"
        f" | 每轮 {steps_per_epoch} 步 | 预热 {warmup_steps} 步"
    )

    training_args = TrainingArguments(
        output_dir=str(CFG.output_dir / "checkpoints"),
        # --- 基本设置 ---
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=CFG.eval_batch_size,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        # --- 优化器设置 ---
        learning_rate=args.lr,
        weight_decay=CFG.weight_decay,
        lr_scheduler_type="linear",      # 线性衰减
        warmup_steps=warmup_steps,       # 线性预热
        max_grad_norm=CFG.max_grad_norm, # 梯度裁剪
        # --- 精度 ---
        fp16=use_fp16,
        # --- 评估与保存策略 ---
        # 注意参数名：transformers 4.30 用 evaluation_strategy（4.41 才改名为 eval_strategy）
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,              # 只保留最近 2 个 checkpoint，省磁盘
        load_best_model_at_end=True,     # 训练结束自动回滚到最优权重
        metric_for_best_model="macro_f1",# 以 macro-F1 作为"最优"标准
        greater_is_better=True,
        # --- 日志 ---
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=str(CFG.output_dir / "logs"),
        report_to=[],                    # 不上传 wandb/tensorboard，保持离线可用
        seed=CFG.seed,
        data_seed=CFG.seed,
        # Windows 上 num_workers>0 容易卡死，设为 0 最稳
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)
        ],
    )

    # ---- 5. 开始训练 ----
    logger.info("-" * 62)
    logger.info("开始训练（混合精度 + 学习率预热 + 早停）")
    logger.info("-" * 62)
    train_result = trainer.train()

    # ---- 6. 保存 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"最优模型已保存到 {out_dir}")

    # ---- 7. 记录结果 ----
    final_metrics = trainer.evaluate()
    logger.info("最终验证集指标：")
    for k, v in final_metrics.items():
        if k.startswith("eval_") and isinstance(v, (int, float)):
            logger.info(f"  {k[5:]:<16}: {v:.4f}")

    save_json(
        {
            "model_name": args.model_name,
            "device": str(device),
            "fp16": use_fp16,
            "seed": CFG.seed,
            "epochs_configured": args.epochs,
            "epochs_actually_run": int(train_result.metrics.get("epoch", 0) + 0.999),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_len": CFG.max_len,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "total_params_M": round(total_params / 1e6, 2),
            "train_runtime_sec": round(train_result.metrics.get("train_runtime", 0), 1),
            "final_val_metrics": {
                k[5:]: round(float(v), 4)
                for k, v in final_metrics.items()
                if k.startswith("eval_") and isinstance(v, (int, float))
            },
        },
        CFG.output_dir / "train_metrics.json",
    )

    # ---- 8. 画训练曲线 ----
    try:
        plot_training_curve(trainer.state.log_history, logger)
    except Exception as e:
        logger.warning(f"训练曲线绘制失败（不影响模型）: {e}")

    logger.info("-" * 62)
    logger.info("训练完成。下一步：python evaluate.py")


def plot_training_curve(log_history, logger):
    """
    从 Trainer 的日志历史里提取数值，画两张并排的图：
      左：训练损失 + 验证损失（看有没有过拟合）
      右：验证集准确率 / macro-F1 / 差评召回率
    这张图在答辩时很有用——能直观说明"早停在什么时候起的作用"。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，避免弹窗阻塞脚本
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [CFG.font_family, "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    train_steps, train_loss = [], []
    eval_epochs, eval_loss = [], []
    eval_acc, eval_f1, eval_recall = [], [], []

    for rec in log_history:
        if "loss" in rec and "eval_loss" not in rec:
            train_steps.append(rec.get("step", 0))
            train_loss.append(rec["loss"])
        if "eval_loss" in rec:
            eval_epochs.append(rec.get("epoch", len(eval_epochs) + 1))
            eval_loss.append(rec["eval_loss"])
            eval_acc.append(rec.get("eval_accuracy", float("nan")))
            eval_f1.append(rec.get("eval_macro_f1", float("nan")))
            eval_recall.append(rec.get("eval_neg_recall", float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左图：损失曲线
    axes[0].plot(train_steps, train_loss, label="训练损失", color="#667eea", linewidth=1.8)
    if eval_loss:
        # 把 eval 的 epoch 换算成 step，方便和训练损失画在同一横轴上
        steps_per_epoch = train_steps[-1] / max(len(eval_epochs), 1) if train_steps else 1
        axes[0].plot(
            [e * steps_per_epoch for e in eval_epochs],
            eval_loss, label="验证损失", color="#FF6B6B",
            marker="o", linewidth=1.8, markersize=6,
        )
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("损失 Loss")
    axes[0].set_title("损失曲线（训练集 vs 验证集）", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, linestyle="--")

    # 右图：指标曲线
    axes[1].plot(eval_epochs, eval_acc, label="准确率 Accuracy", color="#4ECDC4", marker="o", linewidth=1.8)
    axes[1].plot(eval_epochs, eval_f1, label="Macro-F1", color="#667eea", marker="s", linewidth=1.8)
    axes[1].plot(eval_epochs, eval_recall, label="差评召回率", color="#FFB347", marker="^", linewidth=1.8)
    axes[1].set_xlabel("训练轮次 Epoch")
    axes[1].set_ylabel("指标值")
    axes[1].set_title("验证集指标曲线", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    out = CFG.output_dir / "training_curve.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"训练曲线已保存到 {out}")


if __name__ == "__main__":
    main()
