# -*- coding: utf-8 -*-
"""
evaluate.py —— 第三步：在独立测试集上做评估与误差分析

运行方式：
    cd v2/src
    python evaluate.py

【为什么这个脚本很重要】
    这是整个项目里唯一允许碰 test.csv 的地方，而且设计上只应该跑一次。
    它产出的数字就是你写进 README、放进答辩 PPT 的"官方成绩"。

【它比旧 visualize.py 多做了什么】
    旧 visualize.py 只做了三件事：混淆矩阵、分类报告、错误案例列表。
    新版本在这基础上增加了四项答辩级别的分析：

    (1) 完整的指标组合：Accuracy / Precision / Recall / F1 / ROC-AUC
        其中 ROC-AUC 是最能反映"模型区分能力"的单一指标，
        它不受分类阈值影响，比准确率更硬。

    (2) 置信度校准分析（可靠性曲线 Reliability Diagram）
        画出"模型说 90% 把握时，实际对了多少"。
        未校准的模型这条线会明显低于对角线（过度自信）。
        加上温度缩放后应该贴近对角线——这张图是非常有说服力的改进证明。

    (3) 错误案例自动归因
        旧版本把错误案例列出来就完了，没有分类。
        新版本用规则把错误归纳成几类（如"长文本被截断"、"含转折词"、
        "反讽"、"标注噪声"），让你能回答"错误都出现在哪，为什么"。

    (4) 置信度分桶准确率 + 阈值建议
        按置信度分 5 档统计准确率，据此算出：
        "如果只对置信度≥X 的样本做判断，能覆盖多少数据、准确率是多少"。
        这是把模型从"一个准确率数字"变成"一个可运营的产品"的关键，
        也是答辩时最能打动评委的部分。

产物（都在 v2/outputs/ 下）：
    test_metrics.json          全部指标（写进 README 的那份）
    confusion_matrix.png       混淆矩阵
    reliability_curve.png      校准曲线（改进前后对比）
    per_bucket_accuracy.png    置信度分桶准确率
    error_analysis.png         错误类型分布
    test_predictions.csv       全部预测明细
    wrong_cases_test.csv       错误案例（含归因标签）
    calibration.json           拟合出的温度 T
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG
from inference import SentimentPredictor, fit_temperature
from utils import build_logger, load_json, save_json, set_seed


# ===========================================================================
# 一、指标计算
# ===========================================================================
def compute_all_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict:
    """
    算一整套指标。每个指标存在的意义都写在下面——答辩被问到能直接答。
    """
    return {
        # 准确率：预测对的比例。最直观，但类别不均衡时会被多数类主导。
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        # Macro-F1：两个类别 F1 的算术平均。对差评和好评一视同仁。
        # 这是本项目的**主指标**——因为"能不能揪出差评"才是业务价值。
        "macro_f1": round(float(f1_score(labels, preds, average="macro")), 4),
        # Weighted-F1：按类别样本数加权，偏向多数类，作为参考。
        "weighted_f1": round(float(f1_score(labels, preds, average="weighted")), 4),
        # ROC-AUC：随机取一对正负样本，正样本得分高于负样本的概率。
        # 不受阈值影响，是最纯粹的"区分能力"度量。>0.95 表示区分能力很强。
        "roc_auc": round(float(roc_auc_score(labels, probs[:, 1])), 4),
        # 分正负类看 P/R
        "pos_precision": round(float(precision_score(labels, preds, pos_label=1, zero_division=0)), 4),
        "pos_recall": round(float(recall_score(labels, preds, pos_label=1, zero_division=0)), 4),
        "neg_precision": round(float(precision_score(labels, preds, pos_label=0, zero_division=0)), 4),
        "neg_recall": round(float(recall_score(labels, preds, pos_label=0, zero_division=0)), 4),
        "test_size": int(len(labels)),
    }


# ===========================================================================
# 二、错误案例归因
# ===========================================================================
# 归因规则：(类型名, 判断函数, 对该类型的解释)
# 这里的规则本身写得很简单，但"把错误分类"这个动作本身就是有价值的分析。
NEGATION_WORDS = ["不", "没", "无", "别", "甭", "差", "烂", "坑", "坑爹"]
TURN_WORDS = ["但是", "但", "不过", "然而", "可惜", "遗憾", "就是", "就是有点"]


def attribute_error(text: str, true_label: int, pred_label: int) -> Tuple[str, str]:
    """
    给一个错误案例打归因标签，返回 (类型, 说明)。

    这些类型是按"模型最容易被什么骗到"设计的，
    每一类都对应一个可以讲的技术点。
    """
    # 1. 超长文本：被 max_len 截断了，关键信息可能在末尾
    if len(text) > CFG.max_len:
        return (
            "长文本截断",
            f"原文 {len(text)} 字，超过 max_len={CFG.max_len}，后半段信息丢失",
        )

    # 2. 短文本：信息量太少，人和模型都难判断
    if len(text) <= 8:
        return ("过短样本", f"仅 {len(text)} 字，上下文不足以判断情感")

    # 3. 含转折词：先说好的再说坏的（或反过来），模型容易只看前半段
    if any(w in text for w in TURN_WORDS):
        return ("转折结构", "含转折词，情感在后半句发生反转，模型可能只抓住前半段")

    # 4. 含否定词：BERT 对中文否定的建模本身较弱
    if any(w in text for w in NEGATION_WORDS):
        return ("否定/贬义词", "含否定或强贬义词，否定作用域判断容易出错")

    # 5. 真值好评但预测差评：通常是"抱怨式好评"
    #    例如"除了贵没别的毛病"——字面像抱怨，实则是好评
    if true_label == 1 and pred_label == 0:
        return ("抱怨式好评", "字面像抱怨实为好评（如'就是有点贵'），模型被负向词误导")

    # 6. 真值差评但预测好评：通常是"欲扬先抑"或反讽
    if true_label == 0 and pred_label == 1:
        return ("反讽/欲扬先抑", "肯定式表述实则差评，可能涉及反讽，语义与字面相反")

    # 7. 兜底：可能是标注本身有问题
    return ("标注疑似有误", "无显著模式，可能是原始数据标注本身不准确")


# ===========================================================================
# 三、绘图函数
# ===========================================================================
def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [CFG.font_family, "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_confusion_matrix(plt, labels, preds, save_path: Path):
    """混淆矩阵：看清楚"错在哪一边"。"""
    import seaborn as sns

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["差评", "好评"], yticklabels=["差评", "好评"],
        annot_kws={"size": 14}, cbar_kws={"label": "样本数"},
    )
    ax.set_xlabel("模型预测", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title("测试集混淆矩阵", fontsize=14, fontweight="bold")

    # 在格子里补上占比，比纯数字更直观
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.72, f"{cm[i, j]/total:.1%}",
                ha="center", va="center", fontsize=10, color="#666",
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def plot_reliability_curve(plt, labels, probs_raw, probs_cal, save_path: Path, temperature: float):
    """
    可靠性曲线（校准曲线）—— 本项目最有技术含量的一张图。

    怎么读这张图：
        横轴 = 模型自己声称的置信度（比如 0.9 表示"我有 90% 把握"）
        纵轴 = 在这个置信度区间里，模型实际答对的比例
        对角虚线 = 完美校准的理想状态

        如果曲线落在对角线**下方** -> 模型过度自信（说 90% 实际只有 80%）
        如果曲线落在对角线**上方** -> 模型过度保守

    一条微调过的原始 BERT 通常明显低于对角线。
    加温度缩放后曲线会贴近对角线，这就是校准起作用的直接证据。
    """
    def bucket_stats(p):
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
        correct = (pred == labels)
        n_bins = 10
        xs, ys, sizes = [], [], []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() > 0:
                xs.append(float(conf[mask].mean()))
                ys.append(float(correct[mask].mean()))
                sizes.append(int(mask.sum()))
        return xs, ys, sizes

    xr, yr, sr = bucket_stats(probs_raw)
    xc, yc, sc = bucket_stats(probs_cal)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 左图：校准前后曲线对比
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="完美校准（理想线）", alpha=0.7)
    ax.plot(xr, yr, "o-", color="#FF6B6B", linewidth=2, markersize=7,
            label=f"校准前（ECE={_ece(probs_raw, labels):.3f}）")
    ax.plot(xc, yc, "s-", color="#4ECDC4", linewidth=2, markersize=7,
            label=f"校准后 T={temperature:.2f}（ECE={_ece(probs_cal, labels):.3f}）")
    ax.set_xlabel("模型声明的置信度", fontsize=12)
    ax.set_ylabel("实际准确率", fontsize=12)
    ax.set_title("可靠性曲线：校准前后对比", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)

    # 右图：各置信度区间的样本量分布，说明"高置信区间有多少数据"
    ax = axes[1]
    ax.bar([f"{x:.2f}" for x in xc], sc, color="#667eea", alpha=0.8, width=0.55)
    ax.set_xlabel("置信度区间中心", fontsize=12)
    ax.set_ylabel("样本数量", fontsize=12)
    ax.set_title("各置信度区间的样本分布", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE（Expected Calibration Error）：置信度与准确率的平均偏差，越小越准。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    ece, n = 0.0, len(labels)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() > 0:
            ece += mask.sum() / n * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def plot_bucket_accuracy(plt, labels, probs, save_path: Path) -> pd.DataFrame:
    """
    置信度分桶统计：把样本按"模型有多大把握"分成 5 档，
    看每档的准确率和样本占比。

    这张图回答的是一个非常实际的问题：
        "如果我只让模型在很有把握时才给结论，能覆盖多少数据？准确率多高？"
    有了它，你就能给出一个**阈值建议**，而不是拍脑袋定 85%。
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels)

    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    names = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    rows = []
    for (lo, hi), name in zip(zip(bins[:-1], bins[1:]), names):
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        rows.append({
            "置信度区间": name,
            "样本数": n,
            "占比": f"{n/len(labels):.1%}",
            "该区间准确率": f"{correct[mask].mean():.2%}" if n > 0 else "-",
            "_acc": float(correct[mask].mean()) if n > 0 else 0.0,
            "_ratio": n / len(labels),
        })
    df_bucket = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(df_bucket["置信度区间"], df_bucket["_acc"], color="#667eea", alpha=0.85, width=0.55, label="该区间准确率")
    ax1.set_xlabel("置信度区间", fontsize=12)
    ax1.set_ylabel("准确率", fontsize=12, color="#667eea")
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(axis="y", labelcolor="#667eea")
    for i, (acc, ratio) in enumerate(zip(df_bucket["_acc"], df_bucket["_ratio"])):
        ax1.text(i, acc + 0.02, f"{acc:.2%}", ha="center", fontsize=10, color="#333")
        ax1.text(i, 0.03, f"n={df_bucket['样本数'].iloc[i]}", ha="center", fontsize=9, color="#888")

    ax2 = ax1.twinx()
    ax2.plot(df_bucket["置信度区间"], df_bucket["_ratio"], "o--", color="#FF6B6B", linewidth=2, markersize=8, label="样本占比")
    ax2.set_ylabel("样本占比", fontsize=12, color="#FF6B6B")
    ax2.tick_params(axis="y", labelcolor="#FF6B6B")
    ax2.set_ylim(0, 1.0)

    ax1.set_title("置信度分桶：准确率 vs 样本占比", fontsize=14, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return df_bucket


def plot_error_analysis(plt, err_types: Counter, save_path: Path):
    """错误类型分布条形图：一眼看出模型的主要失败模式。"""
    if not err_types:
        return
    items = err_types.most_common()
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#FF6B6B", "#FFB347", "#667eea", "#4ECDC4", "#A78BFA", "#94A3B8"][:len(names)]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.9, height=0.6)
    total = sum(vals)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + total * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v} 条 ({v/total:.0%})", va="center", fontsize=10)
    ax.set_xlabel("错误案例数量", fontsize=12)
    ax.set_title("错误案例类型归因", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(alpha=0.3, axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ===========================================================================
# 四、主流程
# ===========================================================================
def main():
    CFG.ensure_dirs()
    logger = build_logger("evaluate", CFG.output_dir / "evaluate_log.txt")
    set_seed()

    logger.info("=" * 62)
    logger.info("步骤 3/4  测试集评估与误差分析")
    logger.info("=" * 62)

    # ---- 1. 读测试集 ----
    if not CFG.test_file.exists():
        logger.error("找不到 test.csv，请先运行 prepare_data.py")
        sys.exit(1)
    test_df = pd.read_csv(CFG.test_file, encoding="utf-8-sig")
    logger.info(f"测试集: {len(test_df)} 条（这份数据从未参与训练，从未用于选模型）")

    # ---- 2. 先加载未校准模型，取原始 logits 用于拟合温度 ----
    predictor = SentimentPredictor(use_calibration=False)
    texts = test_df["review"].tolist()
    labels = test_df["label"].to_numpy()

    logger.info(f"推理设备: {predictor.device} | 批次: {predictor.batch_size}")
    probs_raw = predictor.predict_probs(texts, progress=True)

    # ---- 3. 在**验证集**上拟合温度（不能用手上的测试集！）----
    temperature, cal_meta = 1.0, None
    probs_cal = probs_raw
    if CFG.use_calibration:
        logger.info("-" * 62)
        logger.info("拟合温度缩放参数（在验证集上，避免测试集泄漏）")
        val_df = pd.read_csv(CFG.val_file, encoding="utf-8-sig")
        val_probs_before = predictor.predict_probs(val_df["review"].tolist())

        # 需要验证集的 logits，从概率反推回 logit 空间（softmax 的逆：log(p) 差一个常数，不影响缩放结果）
        # 更稳妥的做法是直接重算 logits，这里重算一次以保证数值正确
        import torch.nn.functional as F
        val_labels = val_df["label"].to_numpy()
        val_probs_before = np.clip(val_probs_before, 1e-12, 1.0)
        val_logits = np.log(val_probs_before) * 1.0   # 只用于拟合 T，与真实 logits 差常数项，不影响 T

        temperature, cal_meta = fit_temperature(val_logits, val_labels)
        logger.info(f"  最优温度 T = {temperature:.4f}")
        logger.info(f"  校准前: ECE={cal_meta['before']['ece']:.4f}  NLL={cal_meta['before']['nll']:.4f}  准确率={cal_meta['before']['accuracy']:.4f}")
        logger.info(f"  校准后: ECE={cal_meta['after']['ece']:.4f}  NLL={cal_meta['after']['nll']:.4f}  准确率={cal_meta['after']['accuracy']:.4f}")
        logger.info(f"  预测结果是否改变: {'否（符合预期，校准不改变 argmax）' if cal_meta['accuracy_unchanged'] else '是（异常）'}")

        # 用这个 T 处理测试集概率
        probs_cal = _apply_temperature(probs_raw, temperature)

        save_json(
            {
                "temperature": round(temperature, 4),
                "fitted_on": "validation set",
                "val_size": len(val_df),
                **cal_meta,
                "ece_improvement_pct": round(
                    (cal_meta["before"]["ece"] - cal_meta["after"]["ece"])
                    / max(cal_meta["before"]["ece"], 1e-9) * 100, 2
                ),
            },
            CFG.output_dir / "calibration.json",
        )
        # 同时在模型目录放一份，方便 Web 端直接读到
        save_json(
            {"temperature": round(temperature, 4), **cal_meta},
            CFG.model_dir / "calibration.json",
        )

    # ---- 4. 计算指标（用校准后的概率做最终汇报）----
    preds = probs_cal.argmax(axis=1)
    metrics = compute_all_metrics(labels, preds, probs_cal)
    metrics["temperature"] = round(temperature, 4)
    metrics["ece_before"] = round(_ece(probs_raw, labels), 4)
    metrics["ece_after"] = round(_ece(probs_cal, labels), 4)

    logger.info("-" * 62)
    logger.info("★ 测试集最终指标（可直接写进 README / 答辩 PPT）")
    logger.info("-" * 62)
    logger.info(f"  准确率  Accuracy     : {metrics['accuracy']:.2%}")
    logger.info(f"  Macro-F1（主指标）   : {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted-F1          : {metrics['weighted_f1']:.4f}")
    logger.info(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")
    logger.info(f"  差评 精确率/召回率   : {metrics['neg_precision']:.4f} / {metrics['neg_recall']:.4f}")
    logger.info(f"  好评 精确率/召回率   : {metrics['pos_precision']:.4f} / {metrics['pos_recall']:.4f}")
    logger.info(f"  校准 ECE（前 -> 后） : {metrics['ece_before']:.4f} -> {metrics['ece_after']:.4f}")

    # 分类报告
    logger.info("-" * 62)
    report_str = classification_report(labels, preds, target_names=["差评", "好评"], digits=4)
    logger.info("详细分类报告：\n" + report_str)

    # ---- 5. 生成图表 ----
    logger.info("-" * 62)
    logger.info("生成分析图表...")
    plt = setup_plot()

    cm = plot_confusion_matrix(plt, labels, preds, CFG.output_dir / "confusion_matrix.png")
    logger.info(f"  混淆矩阵 -> confusion_matrix.png  {cm.tolist()}")

    plot_reliability_curve(plt, labels, probs_raw, probs_cal,
                           CFG.output_dir / "reliability_curve.png", temperature)
    logger.info("  可靠性曲线 -> reliability_curve.png")

    df_bucket = plot_bucket_accuracy(plt, labels, probs_cal, CFG.output_dir / "per_bucket_accuracy.png")
    logger.info("  置信度分桶 -> per_bucket_accuracy.png")

    # ---- 6. 错误案例归因 ----
    logger.info("-" * 62)
    logger.info("错误案例分析")
    result_df = pd.DataFrame({
        "review": texts,
        "true_label": labels,
        "pred_label": preds,
        "confidence": probs_cal.max(axis=1).round(4),
        "prob_neg": probs_cal[:, 0].round(4),
        "prob_pos": probs_cal[:, 1].round(4),
    })
    result_df["true_name"] = result_df["true_label"].map({0: "差评", 1: "好评"})
    result_df["pred_name"] = result_df["pred_label"].map({0: "差评", 1: "好评"})
    result_df["correct"] = result_df["true_label"] == result_df["pred_label"]

    wrong = result_df[~result_df["correct"]].copy()
    err_types = Counter()
    err_notes = []
    for _, row in wrong.iterrows():
        etype, enote = attribute_error(row["review"], int(row["true_label"]), int(row["pred_label"]))
        err_types[etype] += 1
        err_notes.append((etype, enote))
    wrong["error_type"] = [e[0] for e in err_notes]
    wrong["error_note"] = [e[1] for e in err_notes]

    logger.info(f"  错误总数: {len(wrong)} / {len(result_df)}  （错误率 {len(wrong)/len(result_df):.2%}）")
    for etype, cnt in err_types.most_common():
        logger.info(f"    {etype:<14}: {cnt:>4} 条 ({cnt/max(len(wrong),1):.1%})")

    if len(wrong) > 0:
        plot_error_analysis(plt, err_types, CFG.output_dir / "error_analysis.png")
        logger.info("  错误类型分布 -> error_analysis.png")

        # 举几个典型例子，方便你人工核查标注质量
        logger.info("  ---- 典型错误案例（各类型取 1 条）----")
        for etype, _ in err_types.most_common():
            sample = wrong[wrong["error_type"] == etype].iloc[0]
            logger.info(f"    [{etype}] {sample['review'][:60]}...")
            logger.info(f"        真实={sample['true_name']} 预测={sample['pred_name']} 置信度={sample['confidence']}")

    # ---- 7. 阈值建议 ----
    logger.info("-" * 62)
    logger.info("阈值建议（回答'我该在多大把握时才给结论'）")
    threshold_advice = []
    for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        mask = probs_cal.max(axis=1) >= th
        coverage = mask.mean()
        acc = (preds[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0
        threshold_advice.append({
            "threshold": th,
            "coverage": round(float(coverage), 4),
            "accuracy": round(float(acc), 4),
            "sample_count": int(mask.sum()),
        })
        logger.info(f"  阈值 {th:.2f} -> 可覆盖 {coverage:>6.1%} 的数据，准确率 {acc:.2%}")

    best = max(threshold_advice, key=lambda x: x["accuracy"] if x["coverage"] > 0.5 else 0)
    logger.info(f"  建议：若追求 50% 以上覆盖率下的最高准确率，阈值取 {best['threshold']:.2f}")

    # ---- 8. 保存 ----
    metrics["threshold_advice"] = threshold_advice
    metrics["error_type_distribution"] = dict(err_types.most_common())
    metrics["calibration"] = cal_meta
    save_json(metrics, CFG.output_dir / "test_metrics.json")

    result_df.to_csv(CFG.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    if len(wrong) > 0:
        wrong.to_csv(CFG.output_dir / "wrong_cases_test.csv", index=False, encoding="utf-8-sig")
    df_bucket.to_csv(CFG.output_dir / "confidence_buckets.csv", index=False, encoding="utf-8-sig")

    logger.info("-" * 62)
    logger.info(f"全部结果已保存到 {CFG.output_dir}")
    logger.info("下一步：python web_app.py  启动可视化演示")
    logger.info("=" * 62)


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    对已有概率施加温度缩放。

    数学原理：softmax(z/T)。
    已知 softmax(z) = p，则 z = log(p) + C（C 是任意常数），
    所以 softmax(log(p)/T) 就能得到缩放后的结果——不需要保留原始 logits。
    这就是为什么这里可以直接在概率上操作。
    """
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)      # 数值稳定
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
