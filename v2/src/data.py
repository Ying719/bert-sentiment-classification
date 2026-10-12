# -*- coding: utf-8 -*-
"""
data.py —— 数据处理模块：一次性、彻底地切断数据泄漏

【旧代码的问题】
    旧 train.py 直接读 large_data.csv 并用 train_test_split 切 8:2，
    但用于汇报评估结果的 visualize.py 又去读完整的 large_data.csv。
    两条链路互不知情，导致"评估集里混着训练时见过的样本"，
    准确率虚高，答辩一问就穿帮。

【新方案的核心思路】
    不靠"每次运行时重新随机划分"，而是把划分结果**物化落盘**：
    先由 prepare_data.py 生成 train.csv / val.csv / test.csv 三个固定文件，
    之后训练脚本只读 train.csv 和 val.csv，
    评估脚本只读 test.csv（而且只能跑一次）。

    这样做的价值有两点：
      (1) 物理隔离——训练脚本在代码层面就看不到测试集，想泄漏都做不到；
      (2) 可复现——任何人都能用同样的三个文件复现你的全部指标。

【去重为什么关键】
    large_data.csv 里有 1703 条完全重复的评论。
    如果重复样本跨越了训练/测试边界，就会出现"测试集里那句评论，
    模型训练时已经背过"——这是最常见的泄漏来源。
    所以我们第一步就全局去重。
"""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CFG


# ===========================================================================
# 一、原始数据读取
# ===========================================================================
def load_raw(path=None, text_col: str = "review", label_col: str = "label") -> pd.DataFrame:
    """
    读取原始 CSV，统一列名，只保留 review / label 两列。

    这里额外处理了一个旧代码没注意的坑：
        large_data2.csv 的列顺序是 label,review，而 large_data.csv 是 review,label。
        旧代码用 df['review'] 按列名取值，恰好绕过了这个问题，
        但一旦有人改成按位置取值就会全错。这里显式按列名取，更稳。
    """
    path = CFG.raw_large_data if path is None else path
    # utf-8-sig 能同时正确读取带 BOM 和不带 BOM 的文件
    df = pd.read_csv(path, encoding="utf-8-sig")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} 缺少必需列。实际列名: {list(df.columns)}")

    df = df[[text_col, label_col]].rename(columns={text_col: "review", label_col: "label"})
    return df


# ===========================================================================
# 二、清洗
# ===========================================================================
def clean(df: pd.DataFrame, min_len: int = None) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗流程：补空值 -> 去首尾空白 -> 删空文本 -> 长度过滤 -> 全局去重。

    每一步的删除数量都记录下来，最后写进 split_report.json。
    为什么要把"删了多少"记下来？
        答辩老师很爱问"你的数据预处理做了什么"。
        有一张"原始 10800 条 -> 去重后 9097 条"的漏斗表，
        比任何口头描述都有说服力。
    """
    min_len = CFG.min_len if min_len is None else min_len
    stats = {"raw": len(df)}

    # (1) 缺失值
    df = df.dropna(subset=["review", "label"]).copy()
    stats["after_dropna"] = len(df)

    # (2) 转成字符串并去首尾空白
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0]
    stats["after_strip_empty"] = len(df)

    # (3) 长度过滤：太短的样本通常是"好""还行"这类噪声，学不到东西
    df = df[df["review"].str.len() >= min_len]
    stats["after_min_len"] = len(df)

    # (4) 全局去重 —— 切断泄漏的关键一步
    before = len(df)
    df = df.drop_duplicates(subset=["review"], keep="first")
    stats["after_dedup"] = len(df)
    stats["removed_duplicates"] = before - len(df)

    # (5) 标签必须是 0/1
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    stats["after_label_check"] = len(df)

    # (6) 重置索引，避免后续按 index 取值时错位
    df = df.reset_index(drop=True)

    # 记录长度分布，方便判断 max_len 设得合不合理
    lengths = df["review"].str.len()
    stats["length"] = {
        "mean": round(float(lengths.mean()), 1),
        "p50": int(lengths.quantile(0.50)),
        "p95": int(lengths.quantile(0.95)),
        "p99": int(lengths.quantile(0.99)),
        "max": int(lengths.max()),
        "over_max_len_ratio": round(float((lengths > CFG.max_len).mean()), 4),
    }
    return df, stats


# ===========================================================================
# 三、三路划分（train / val / test）
# ===========================================================================
def split_three_way(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    分层抽样切出 train / val / test 三份，比例来自 config（默认 8:1:1）。

    "分层"（stratify）是什么意思？
        保证三份数据里好评/差评的比例都和总体一致。
        不分层的话，如果随机切出来的测试集正好好评偏多，
        你的准确率就会虚高——那是抽样运气，不是模型能力。

    为什么要两步切？
        第一次切出 test，第二次从剩下的里切出 val。
        两次都带 stratify，比例才严格可控。
    """
    # 第一步：切出 10% 作为测试集（永不参与训练）
    train_val_df, test_df = train_test_split(
        df,
        test_size=CFG.test_ratio,
        random_state=CFG.seed,
        stratify=df["label"],
    )

    # 第二步：从剩下的 90% 里切出 val。
    # 注意这里的分母变了，要重新算比例：val 占整体的 10% -> 占剩余的 10/90
    val_relative = CFG.val_ratio / (CFG.train_ratio + CFG.val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=CFG.seed,
        stratify=train_val_df["label"],
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


# ===========================================================================
# 四、泄漏自检（本次升级的"体检报告"）
# ===========================================================================
def leak_check(splits: Dict[str, pd.DataFrame]) -> Dict:
    """
    主动检查三份数据之间有没有交叉污染。

    检查三件事：
      1. train ∩ test 的文本交集大小（必须为 0）
      2. train ∩ val  的文本交集大小（必须为 0）
      3. val   ∩ test 的文本交集大小（必须为 0）
    另外顺便查一遍"同一句文本出现了不同标签"的情况——
    这类矛盾样本会让模型学不明白，也是数据质量的硬伤。
    """
    sets = {k: set(v["review"]) for k, v in splits.items()}
    report = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sets[a] & sets[b]
        report[f"{a}_∩_{b}"] = len(inter)
        report[f"{a}_∩_{b}_samples"] = list(inter)[:5]  # 留几个例子方便人工核查

    # 标签冲突检查（在完整数据上做，不受划分影响）
    full = pd.concat(splits.values(), ignore_index=True)
    conflict = full.groupby("review")["label"].nunique()
    report["conflicting_labels"] = int((conflict > 1).sum())

    report["all_clean"] = all(
        report[k] == 0 for k in ("train_∩_val", "train_∩_test", "val_∩_test")
    )
    return report


# ===========================================================================
# 五、对外主入口
# ===========================================================================
def build_datasets(force: bool = False, logger=None) -> Dict[str, pd.DataFrame]:
    """
    完整流程：读原始数据 -> 清洗 -> 三路划分 -> 泄漏自检 -> 落盘。

    参数 force:
        False（默认）—— 如果 v2/data/ 下已经有划分好的文件，直接读，不重新划分。
                        这一点非常重要：保证你"评估时用的测试集"永远是同一份。
        True          —— 忽略已有文件，重新走一遍划分（改了比例或数据才需要）。
    """
    from utils import save_json

    CFG.ensure_dirs()
    log = logger.info if logger else print

    # ---- 命中缓存：直接返回 ----
    if not force and all(p.exists() for p in (CFG.train_file, CFG.val_file, CFG.test_file)):
        log("检测到已划分好的数据，直接加载（如需重新划分请加 --force）")
        splits = {
            "train": pd.read_csv(CFG.train_file, encoding="utf-8-sig"),
            "val": pd.read_csv(CFG.val_file, encoding="utf-8-sig"),
            "test": pd.read_csv(CFG.test_file, encoding="utf-8-sig"),
        }
        for k, v in splits.items():
            log(f"  {k:<5}: {len(v):>6} 条")
        return splits

    # ---- 从零构建 ----
    log(f"读取原始数据: {CFG.raw_large_data}")
    raw = load_raw()
    log(f"  原始样本 {len(raw)} 条")

    log("清洗中（去空值 / 长度过滤 / 全局去重）...")
    df, cstats = clean(raw)
    log(
        f"  清洗后 {len(df)} 条"
        f"（删除重复 {cstats['removed_duplicates']} 条、"
        f"过短 {cstats['after_strip_empty'] - cstats['after_min_len']} 条）"
    )

    log(f"按 {CFG.train_ratio:.0%}/{CFG.val_ratio:.0%}/{CFG.test_ratio:.0%} 分层划分...")
    splits = split_three_way(df)
    for k, v in splits.items():
        pos = int((v["label"] == 1).sum())
        neg = int((v["label"] == 0).sum())
        log(f"  {k:<5}: {len(v):>6} 条  （好评 {pos} / 差评 {neg}）")

    log("泄漏自检中...")
    leak = leak_check(splits)
    if leak["all_clean"]:
        log("  ✅ 通过：train / val / test 之间零重叠，无标签冲突")
    else:
        log(f"  ❌ 检出重叠！详情: {leak}")

    # ---- 落盘 ----
    for k, v in splits.items():
        out = getattr(CFG, f"{k}_file")
        v.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"划分结果已固定保存到 {CFG.data_dir}")

    save_json(
        {
            "config": {
                "seed": CFG.seed,
                "ratios": [CFG.train_ratio, CFG.val_ratio, CFG.test_ratio],
                "max_len": CFG.max_len,
                "min_len": CFG.min_len,
                "source_file": str(CFG.raw_large_data),
            },
            "cleaning_funnel": cstats,
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "leak_check": leak,
        },
        CFG.split_report,
    )
    log(f"划分报告已保存到 {CFG.split_report}")
    return splits
