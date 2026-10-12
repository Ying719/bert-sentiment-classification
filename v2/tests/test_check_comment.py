# -*- coding: utf-8 -*-
"""
test_check_comment.py —— 内容合法性检查（第一层闸门）的回归测试

运行方式：
    cd v2/src
    python ../tests/test_check_comment.py

它验证的是 inference.check_comment()，不加载模型，所以秒级完成。

为什么要专门测这个函数？
    它是纯规则判断，规则写松了会漏掉乱码和指令句，
    写紧了会误伤真评论（例如"什么都好，就是有点贵"含疑问词但是评论）。
    这两类错误都不会在模型指标里体现出来，只能靠用例覆盖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import check_comment  # noqa: E402

# (输入文本, 期望是否能交给模型判断)
CASES = [
    # ---------------- 应该能判（真正的评论） ----------------
    ("这个手机屏幕清晰，续航很长", True),
    ("物流太慢，客服态度差", True),
    ("一般般吧，还行", True),
    ("除了贵点没别的毛病", True),
    ("这家店服务特别好，强烈推荐！", True),
    ("东西很贵，质量还差，再也不来了", True),
    ("这个产品质量很好，物流也快", True),
    ("收到了，和描述一致，满意", True),
    # 含疑问词但实际是评论的易误伤样本 —— 必须能判
    ("什么都好，就是有点贵", True),
    ("不知道怎么说，反正不太满意", True),
    ("怎么用都不顺手，退了", True),

    # ---------------- 日常陈述句：应该拒识（第 8 条闸门） ----------------
    ("我吃饭了", False),
    ("今天上课", False),
    ("我去图书馆了", False),
    ("我睡觉了", False),
    ("下班了", False),
    ("我去打球", False),

    # ---------------- 含情感词但不该被第 8 条拦的评论 ----------------
    ("手机到了，很好用", True),
    ("快递很快", True),
    ("质量不错", True),
    ("这家店服务很好", True),
    # 3 字短评：电商场景常见，含情感词必须放行
    ("太贵了", True),
    ("差评了", True),
    ("退货了", True),
    ("太好了", True),

    # ---------------- 应该拒识（非评论内容） ----------------
    ("1234567", False),
    ("!@#$%^&*", False),
    ("aaa", False),
    ("test", False),
    ("", False),
    ("今天", False),
    ("还行", False),
    ("你好", False),
    ("帮我查一下订单", False),
    ("帮我写一首诗", False),
    ("请翻译这段话", False),
    ("什么是人工智能", False),
    ("什么是机器学习", False),
    ("这个怎么用", False),
    ("为什么会这样", False),
    ("http://www.abc.com/12345", False),
    ("print('hello world')", False),
]


def main() -> int:
    passed = 0
    failed = []

    for text, expected in CASES:
        got, reason = check_comment(text)
        if got == expected:
            passed += 1
        else:
            failed.append((text, expected, got, reason))

    print(f"通过 {passed}/{len(CASES)}")

    if failed:
        print("\n失败用例：")
        for text, expected, got, reason in failed:
            exp = "可判" if expected else "拒识"
            act = "可判" if got else "拒识"
            print(f"  {text!r:34} 期望{exp} 实得{act}  ({reason})")
        return 1

    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
