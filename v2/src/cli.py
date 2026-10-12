# -*- coding: utf-8 -*-
"""
cli.py —— 命令行交互式预测

运行方式：
    cd v2/src
    python cli.py

比旧 chat_predict.py 多的东西：
    - 显示两个类别的完整概率分布，而不是只给一个结论
    - 显示置信度，并按阈值提示"无法确定"
    - 支持从文件批量预测并导出 CSV
    - 显示本次推理耗时（可以看到批量推理有多快）
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG
from inference import SentimentPredictor


BANNER = """
================================================================
  BERT 中文情感分析 · 命令行工具  (v2)
================================================================
  直接输入评论回车即可分析
  命令： :file <路径>   批量分析文件并导出 CSV
         :info          查看模型信息
         :quit          退出
----------------------------------------------------------------
"""


def format_result(r, rank=None):
    """把一条预测结果格式化成对齐的文本行。"""
    name = r["label_name"]

    if name == "好评":
        tag, emoji = "好评", "^_^"
    elif name == "差评":
        tag, emoji = "差评", "T_T"
    elif name == "非评论内容":
        tag, emoji = "非评论", "---"
    else:
        tag, emoji = "不明确", "?"

    prefix = f"{rank:>3}. " if rank is not None else ""
    text = r["text"] if len(r["text"]) <= 42 else r["text"][:42] + "..."

    if r["confidence"] is None:
        detail = f"未做情感判断（{r['reject_reason']}）"
    else:
        detail = (
            f"置信度 {r['confidence']*100:>5.1f}%  "
            f"P(好)={r['probs'][1]:.3f} P(差)={r['probs'][0]:.3f}"
        )
    return f"{prefix}{text:<46} {tag:<5} {detail}"


def main():
    parser = argparse.ArgumentParser(description="命令行情感分析")
    parser.add_argument("--model", default=None, help="模型目录，默认 v2/outputs/model")
    args = parser.parse_args()

    print("正在加载模型...")
    predictor = SentimentPredictor(args.model or CFG.model_dir)
    print(f"就绪 | 设备: {predictor.device} | 温度 T={predictor.temperature:.3f}")
    print(BANNER)

    while True:
        try:
            raw = input("评论 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not raw:
            continue

        # ---- 内置命令 ----
        if raw.lower() in (":quit", ":q", "exit", "quit", "退出"):
            print("再见。")
            break

        if raw.lower() == ":info":
            print(f"  模型目录      : {predictor.model_path}")
            print(f"  推理设备      : {predictor.device}")
            print(f"  批次大小      : {predictor.batch_size}")
            print(f"  最大长度      : {CFG.max_len}")
            print(f"  温度校准 T    : {predictor.temperature}")
            print(f"  不确定阈值    : {CFG.uncertain_threshold}")
            if predictor.calibration_meta:
                b = predictor.calibration_meta.get("before", {})
                a = predictor.calibration_meta.get("after", {})
                print(f"  校准前 ECE    : {b.get('ece', '—')}")
                print(f"  校准后 ECE    : {a.get('ece', '—')}")
            print()
            continue

        if raw.startswith(":file"):
            parts = raw.split(maxsplit=1)
            if len(parts) < 2:
                print("  用法: :file <文件路径>\n")
                continue
            path = Path(parts[1].strip('"').strip("'"))
            if not path.exists():
                print(f"  找不到文件: {path}\n")
                continue
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                col = next(
                    (c for c in ("review", "text", "comment", "content") if c in df.columns),
                    df.columns[0],
                )
                texts = df[col].astype(str).tolist()
                print(f"  读取 {len(texts)} 条（列名: {col}），推理中...")
                t0 = time.time()
                results = predictor.predict(texts)
                dt = time.time() - t0
                stats = predictor.summarize(results)
                print(
                    f"  完成，耗时 {dt:.2f}s（{dt/len(texts)*1000:.1f} ms/条）\n"
                    f"  好评 {stats['positive']} | 差评 {stats['negative']} | "
                    f"情感不明显 {stats['uncertain']} | 非评论 {stats['non_comment']} | "
                    f"平均置信度 {stats['avg_confidence']:.3f}"
                )
                out = path.with_name(path.stem + "_预测结果.csv")
                pd.DataFrame(
                    [
                        {
                            "review": r["text"],
                            "预测": r["label_name"],
                            "置信度": r["confidence"] if r["confidence"] is not None else "",
                            "P差评": r["probs"][0],
                            "P好评": r["probs"][1],
                            "拒识原因": r.get("reject_reason", ""),
                        }
                        for r in results
                    ]
                ).to_csv(out, index=False, encoding="utf-8-sig")
                print(f"  已导出: {out}\n")
            except Exception as e:
                print(f"  处理失败: {e}\n")
            continue

        # ---- 普通文本预测 ----
        t0 = time.time()
        r = predictor.predict_one(raw)
        dt = (time.time() - t0) * 1000
        print("  " + format_result(r))
        print(f"  （耗时 {dt:.0f} ms）\n")


if __name__ == "__main__":
    main()
