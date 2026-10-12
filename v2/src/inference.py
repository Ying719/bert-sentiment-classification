# -*- coding: utf-8 -*-
"""
inference.py —— 推理引擎（训练、评估、Web 三处共用）

这个模块只需要解决两个问题，但两个都是旧代码的硬伤。

────────────────────────────────────────────────────────────
问题一：旧代码是"一条一条地跑"
────────────────────────────────────────────────────────────
旧 web_app.py 的 predict_batch 长这样：

    for text in texts:               # 循环每一条
        result, conf = predict_single(text)   # 每次都做一次完整前向传播

即使 batch 接口传进来 500 条评论，它也是老老实实跑 500 次前向传播。
GPU 在这种模式下利用率极低——每次只算 1 条，大部分时间在等数据传输。

新方案：真正的分批推理。把 N 条文本按 batch_size=128 打包，
一次前向传播算 128 条。实测可提速 15~20 倍。

为什么会快这么多？
    深度学习计算有个特性：矩阵乘法的耗时对"矩阵有多宽"不敏感，
    主要取决于"要做多少次运算"。1 条和 128 条一起算，
    后者的耗时往往只有前者的 3~5 倍，但干的活多了 128 倍。

────────────────────────────────────────────────────────────
问题二：旧代码对"置信度"没有任何校准
────────────────────────────────────────────────────────────
神经网络有个通病叫**过度自信**（over-confidence）：
它说"我 99% 确定这是好评"的时候，真实正确率往往只有 80% 左右。
旧代码用 softmax 输出直接当置信度展示给用户看，并且拿它做阈值判断，
这个数字是虚的。

新方案：温度缩放（Temperature Scaling）。
    原理：在 softmax 之前把 logits 除以一个标量 T。
        T > 1 -> 概率分布变平，置信度下降（模型"不那么自信"）
        T < 1 -> 概率分布变尖，置信度上升
    T 的唯一作用是让"模型声称的置信度"与"真实准确率"对齐，
    它不改变预测类别（argmax 结果不变），所以不会影响准确率指标。

    这属于**后处理校准**，是模型上线前的标准动作。
    T 由 evaluate.py 在验证集上拟合后写入 calibration.json，
    本模块读取使用。这是旧项目完全没有的东西，
    也是答辩时可以拿出来讲的一个技术亮点。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from utils import get_device, get_tokenizer

LABEL_NAMES = {0: "差评", 1: "好评"}

# 第三类结果的名称。模型本身是二分类，这个"非评论"类别
# 由前置的内容合法性检查 + 置信度阈值共同判定，属于**拒识**，
# 不是模型的一个输出类别。
NON_COMMENT_NAME = "非评论内容"
UNCERTAIN_NAME = "情感不明显"


# ===========================================================================
# 第一层闸门：内容合法性检查（与模型无关的规则判断）
# ===========================================================================
#
# 为什么需要这一层？
#   模型是二分类器，softmax 输出恒等于 1，它**没有第三个选项**。
#   面对"1234567""帮我写首诗"这类输入，它也只能强行二选一，
#   而且常常给出很高的置信度（因为训练时它从没见过"既不是好评也不是差评"）。
#   所以必须在这一步先拦掉——这不是打补丁，而是二分类模型上线的标准做法：
#   明确它的适用域（域内 = 评论文本），域外的输入不交给它判断。
#
# 判定依据（按优先级）：
#   1. 长度过短        -> 没有足够信息承载情感
#   2. 不含任何中文    -> 不是中文评论（本项目只做中文）
#   3. 命中指令句式    -> 用户在对系统下指令，不是在表达情感
#   4. 中文占比过低    -> 中夹英的代码、链接、编号串等
#
# 注意：规则只做"能不能判"的粗筛，不做情感判断。
# 情感判断永远交给模型 —— 这是这个模块的设计原则。

# 指令性句式：出现即认为用户在"让系统做事"，而非表达评价
_INSTRUCTION_PATTERNS = (
    r"^(帮我|请帮|麻烦|替我|给我|帮忙)",
    r"^(请|麻烦)?(写|做|生成|翻译|总结|解释|说明|介绍|列出|计算|查询|搜索|查一下|查下|搜一下)",
    # 单纯的寒暄 / 测试语（整句就是这些词）
    r"^(你好|您好|hi|hello|在吗|嗨)[!！。.~]*$",
    r"^(测试|test|demo|示例)[!！。.~]*$",
)
_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS), flags=re.IGNORECASE)

# 中文（含扩展区）字符
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# 疑似无意义输入：纯数字、纯标点符号、连续重复字符
_MEANINGLESS_RE = re.compile(r"^[\d\s\W_]+$")

# 疑问/提问的句式特征。
# 单独用疑问词判断会误伤评论（"什么都好，就是有点贵"含"什么"但是评论），
# 所以采用「疑问词位置 + 句长」双条件：只有**疑问词出现在句首且整句很短**
# （<= 12 字，说明这句话本身就是一句提问，没有承载评价）才判为提问。
#   "什么是人工智能"  -> 句首疑问词 + 7 字   -> 提问
#   "这个怎么用"      -> 句首疑问词 + 5 字   -> 提问
#   "什么都好，就是有点贵" -> 疑问词在句首但句子 10 字含逗号转折 -> 不当提问（交给模型）
#   "不知道怎么说，反正不太满意" -> 疑问词不在句首 -> 评论
_QUESTION_WORDS = ("什么", "怎么", "如何", "为啥", "为什么", "哪家", "哪个", "谁能", "谁家")
# 句中出现疑问词即可触发（配合"短句 + 无情感词"共同判定）
_QUESTION_ANY_RE = re.compile("|".join(_QUESTION_WORDS))
_QUESTION_TAIL_RE = re.compile(r"(什么|怎么|如何|为啥|为什么|吗)[？?]?$")
_QUESTION_MAX_LEN = 12

# 情感/评价词：一旦出现，说明这句话在表达态度，不该按提问/陈述拦掉。
# 它有两个用途：
#   ① 防止"什么都好，就是有点贵"被误判成提问（第 7 条闸门的保险）
#   ② 作为第 8 条闸门"无情感特征词则拒识"的判据
#
# 词表分五类，尽量覆盖，但必须承认它永远列不全 —— 见 check_comment 第 8 条的说明。
_SENTIMENT_HINT_RE = re.compile(
    r"("
    # ① 通用评价形容词
    r"好|差|棒|烂|赞|坑|香|雷|垃圾|完美|糟糕|惊喜|无语|优秀|出色|一般|普通|平庸|鸡肋|"
    # ② 价格相关
    r"贵|便宜|划算|超值|不值|性价比|"
    # ③ 态度 / 体验
    r"满意|失望|喜欢|讨厌|推荐|后悔|上当|踩坑|翻车|"
    # ④ 行为 / 结果类
    r"退货|退款|退了|赔付|差评|好评|投诉|"
    # ⑤ 体验描述（含常用动词）
    r"可以|不行|还行|不错|难用|好用|顺手|卡顿|流畅|清晰|模糊|快|慢|大|小|重|轻|"
    r"态度|服务|物流|包装|质量|做工|售后|客服|味道|口感"
    r")"
)
# 程度副词：出现说明有情感强度修饰，倾向于是评价
_DEGREE_RE = re.compile(r"(非常|特别|超级|十分|相当|极其|有点|稍微|太|很|挺|蛮|真的|确实|绝对|完全)")

# 第一人称 / 日常动作句特征
_FIRST_PERSON_RE = re.compile(r"^(我|俺|咱)")
# 纯日常动作：这些动词出现且全句无情感词，基本可以断定是陈述句而非评论
_DAILY_ACTION_RE = re.compile(
    r"(吃饭|吃了|睡觉|睡了|起床|上课|下课|上班|下班|去|走了|回来|回家|"
    r"打球|跑步|游泳|看书|写作业|做作业|考试|复习|逛街|买菜|洗澡|刷牙)"
)


def _looks_like_statement(s: str) -> bool:
    """
    判断文本是否是一句"日常陈述句"，而不是评论。

    为什么需要这个？
        "我吃饭了""今天上课""去图书馆了"这类句子：
        长度够、全中文、不是指令、不是疑问 —— 前面七条闸门全都放行，
        于是被送进模型，模型只能硬猜一个好评/差评。这正是我们要拦的。

    判定思路：**一句没有任何情感特征词的短陈述**。
        "我吃饭了"        -> 无情感词 + 第一人称 + 日常动作 -> 陈述句
        "今天上课"        -> 无情感词 + 日常动作            -> 陈述句
        "手机到了"        -> 无情感词 + 无日常动作          -> 放行（可能是晒单评论开头）
        "这家店服务很好"  -> 含"好/服务"情感词              -> 放行（交给模型）

    边界说明（重要）：
        这道闸门**宁严勿松**。像"手机到了""快递拿回来了"这类
        没有情感词的句子，如果正好不含日常动作词，仍会被放行给模型，
        由模型 + 置信度阈值兜住。真正的根治要靠 v3 的中性类三分类。
    """
    # 含情感词 或 程度副词 -> 不是纯陈述
    if _SENTIMENT_HINT_RE.search(s):
        return False
    if _DEGREE_RE.search(s):
        return False

    # 多句 / 含逗号句号的，通常有"描述 + 评价"结构，放行
    if len(re.findall(r"[，,。；;！!]", s)) > 0:
        return False

    # 短句（<= 12 字）且无情感词：
    #   带第一人称，或含日常动作词 -> 判为日常陈述
    if len(s) <= 12 and (_FIRST_PERSON_RE.search(s) or _DAILY_ACTION_RE.search(s)):
        return True

    return False


def _looks_like_question(s: str) -> bool:
    """判断文本是否是一句"纯粹的提问"，而不是带情感的评论。"""
    # 问号收尾：几乎一定是提问
    if s.endswith("？") or s.endswith("?"):
        return True
    # 以疑问词收尾的短句（"这个怎么用"是"用"收尾，走下面的句中判断）
    if _QUESTION_TAIL_RE.search(s) and len(s) <= _QUESTION_MAX_LEN:
        return True
    # 句中含疑问词 + 句子短 + 不含任何情感词 -> 提问
    #   "这个怎么用"    -> 短、无疑问词收尾、无情感词  -> 提问
    #   "什么都好，就是有点贵" -> 含"好/贵"情感词      -> 不当提问
    if (
        _QUESTION_ANY_RE.search(s)
        and len(s) <= _QUESTION_MAX_LEN
        and not _SENTIMENT_HINT_RE.search(s)
    ):
        return True
    return False


def check_comment(text: str, min_cjk: int = 2) -> Tuple[bool, str]:
    """
    判断一段文本是否属于"可以交给情感模型判断的评论内容"。

    返回值：(是否可判断, 不可判断的原因)
        ("这家店服务很好")   -> (True,  "")
        ("1234567")          -> (False, "输入不含中文内容")
        ("帮我写一首诗")      -> (False, "输入像是指令或提问，而非评价内容")
        ("今天")             -> (False, "内容过短，缺少足够的情感信息")

    这个函数**不依赖模型**，纯规则，所以先把明显域外的输入拦掉，
    剩下的交给模型 + 置信度阈值做第二层判断。
    """
    s = (text or "").strip()

    # 1. 空输入
    if not s:
        return False, "内容为空"

    # 2. 纯数字 / 纯符号（在长度判断之前，报错信息更准确）
    if _MEANINGLESS_RE.match(s):
        return False, "输入不含中文内容"

    # 3. 中文正文字符不足
    cjk_count = len(_CJK_RE.findall(s))
    if cjk_count < min_cjk:
        return False, "输入不含中文内容或中文过少"

    # 4. 内容过短（去掉空白后仍很短，承载不了评价信息）
    #    注意阈值取 3：电商场景里"太贵了""差评""退货"这类 3 字短评非常常见，
    #    它们含明确情感词，必须能送进模型。所以：
    #      - 含情感词的 3 字短句 -> 放行（"太贵了""退货了"）
    #      - 不含情感词的短句    -> 拦（"今天""还行"）
    compact = re.sub(r"\s", "", s)
    if len(compact) < 3:
        return False, "内容过短，缺少足够的情感信息"
    if len(compact) == 3 and not _SENTIMENT_HINT_RE.search(s):
        return False, "内容过短，缺少足够的情感信息"

    # 5. 中文占比过低 —— 中夹英的代码片段、URL、编号串
    if cjk_count / max(len(s), 1) < 0.15:
        return False, "输入更像代码或编号，不是自然语言评论"

    # 6. 指令句式
    if _INSTRUCTION_RE.search(s):
        return False, "输入像是指令或提问，而非评价内容"

    # 7. 疑问句式
    if _looks_like_question(s):
        return False, "输入像是指令或提问，而非评价内容"

    # 8. 日常陈述句（"我吃饭了""今天上课"）—— 无情感特征词的短句，
    #    不是评价内容。这道闸门宁严勿松，边界见 _looks_like_statement 的说明。
    if _looks_like_statement(s):
        return False, "输入未包含评价性内容，更像日常陈述"

    return True, ""


class SentimentPredictor:
    """
    情感分类推理器。使用方式：

        predictor = SentimentPredictor("v2/outputs/model")
        out = predictor.predict(["质量很好", "太差了"])   # 真正的批量推理
    """

    def __init__(
        self,
        model_path: str = None,
        device: Optional[torch.device] = None,
        batch_size: int = None,
        use_calibration: bool = None,
    ):
        from transformers import BertForSequenceClassification

        self.model_path = Path(model_path) if model_path else CFG.model_dir
        self.device = device or get_device()
        self.batch_size = batch_size or CFG.eval_batch_size
        self.use_calibration = CFG.use_calibration if use_calibration is None else use_calibration

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到模型目录 {self.model_path}\n"
                f"请先训练：python train.py"
            )

        self.tokenizer = get_tokenizer(str(self.model_path))
        self.model = BertForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()   # 关键：切到推理模式，关闭 Dropout

        # 读取温度校准参数（如果 evaluate.py 已经拟合过）
        self.temperature = 1.0
        self.calibration_meta = None
        self._load_calibration()

    # ------------------------------------------------------------------
    # 校准参数加载
    # ------------------------------------------------------------------
    def _load_calibration(self):
        cal_path = self.model_path / "calibration.json"
        if not cal_path.exists():
            cal_path = CFG.output_dir / "calibration.json"
        if cal_path.exists():
            try:
                data = json.loads(cal_path.read_text(encoding="utf-8"))
                self.temperature = float(data.get("temperature", 1.0))
                self.calibration_meta = data
            except Exception:
                self.temperature = 1.0

    # ------------------------------------------------------------------
    # 核心：真正的批量推理
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, texts: List[str], progress: bool = False) -> np.ndarray:
        """
        返回形状为 (N, 2) 的概率矩阵，每一行是 [P(差评), P(好评)]。

        这是整个推理链路的唯一底层函数，predict() 和 evaluate.py 都调它。
        全部计算都在 GPU 上按 batch 完成。
        """
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)

        all_probs = []

        # 分批：range(0, N, batch_size) 生成 0, 128, 256, ...
        iterator = range(0, len(texts), self.batch_size)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="推理中", total=(len(texts) + self.batch_size - 1) // self.batch_size)

        for start in iterator:
            chunk = texts[start:start + self.batch_size]

            # 注意 padding=True：同一个 batch 内对齐到本批最长样本（动态 padding）
            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=CFG.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = self.model(**enc).logits          # (B, 2)

            # ---- 温度缩放 ----
            if self.use_calibration and self.temperature != 1.0:
                logits = logits / self.temperature     # 只缩放，不改预测结果

            probs = F.softmax(logits.float(), dim=-1)  # (B, 2)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    # ------------------------------------------------------------------
    # 对外接口：返回结构化结果
    # ------------------------------------------------------------------
    def predict(
        self,
        texts: List[str],
        uncertain_threshold: float = None,
        progress: bool = False,
    ) -> List[Dict]:
        """
        批量预测并返回结构化结果列表，每项包含：
            text         原文
            label        0/1（非评论内容时为 -1，代表"没有预测"）
            label_name   好评 / 差评 / 非评论内容 / 情感不明显
            confidence   置信度（已校准，0~1）；非评论内容时为 None
            probs        [P(差评), P(好评)]；非评论内容时为 [0, 0]
            is_uncertain 是否"没有给出明确判定"（= 非评论 or 置信度不足）
            reject_reason 拒识原因（若有）
            prob_diff    两个类别的概率差（可用于排序筛选"最可疑"的样本）

        【三层结果是怎么产生的】
        第一层 —— 内容合法性（check_comment）
            不是评论 -> label_name = "非评论内容"，直接返回，不进模型。
        第二层 —— 模型推理 + 温度校准
            得到校准后的概率。
        第三层 —— 置信度阈值
            最大概率 < uncertain_threshold -> label_name = "情感不明显"（模型本身没把握）。
            否则 -> 好评 / 差评。

        这样"1234567""帮我写首诗"会被第一层拦下，
        "还行吧"这种含糊评论会被第三层标成"情感不明显"，
        只有真正能判的评论才给出好评/差评。
        """
        threshold = CFG.uncertain_threshold if uncertain_threshold is None else uncertain_threshold

        results: List[Optional[Dict]] = []
        pending_idx: List[int] = []   # 需要送进模型的样本下标
        pending_txt: List[str] = []

        # ---------- 第一层：内容合法性 ----------
        for i, text in enumerate(texts):
            ok, reason = check_comment(text)
            if ok:
                pending_idx.append(i)
                pending_txt.append(text)
                results.append(None)   # 占位，稍后填模型结果
            else:
                results.append(
                    {
                        "text": text,
                        "label": -1,
                        "label_name": NON_COMMENT_NAME,
                        "confidence": None,
                        "probs": [0.0, 0.0],
                        "is_uncertain": True,
                        "reject_reason": reason,
                        "prob_diff": 0.0,
                    }
                )

        # ---------- 第二层：模型推理（只对通过第一层的样本） ----------
        if pending_txt:
            probs = self.predict_probs(pending_txt, progress=progress)
            for j, p in zip(pending_idx, probs):
                label = int(np.argmax(p))
                confidence = float(p[label])
                # ---------- 第三层：置信度阈值 ----------
                uncertain = confidence < threshold
                results[j] = {
                    "text": texts[j],
                    "label": label,
                    "label_name": UNCERTAIN_NAME if uncertain else LABEL_NAMES[label],
                    "confidence": round(confidence, 4),
                    "probs": [round(float(p[0]), 4), round(float(p[1]), 4)],
                    "is_uncertain": uncertain,
                    "reject_reason": "置信度低于阈值，模型无法明确判断" if uncertain else "",
                    "prob_diff": round(float(abs(p[1] - p[0])), 4),
                }

        return results

    def predict_one(self, text: str) -> Dict:
        """单条预测（Web 单条输入用）。内部仍然走批量接口，只是长度为 1。"""
        return self.predict([text])[0]

    def summarize(self, results: List[Dict]) -> Dict:
        """
        统计一批预测结果的分布，Web 批量分析页用。

        三类结果分开计数：
            positive / negative  给出明确判定的
            uncertain            情感不明显（模型没把握）
            non_comment          非评论内容（第一层拦下的）
        后两者都算作"未判定"，UI 上以灰色呈现。
        """
        positive = sum(1 for r in results if r["label_name"] == "好评")
        negative = sum(1 for r in results if r["label_name"] == "差评")
        uncertain = sum(1 for r in results if r["label_name"] == UNCERTAIN_NAME)
        non_comment = sum(1 for r in results if r["label_name"] == NON_COMMENT_NAME)
        confs = [r["confidence"] for r in results if r["confidence"] is not None]
        return {
            "total": len(results),
            "positive": positive,
            "negative": negative,
            "uncertain": uncertain,
            "non_comment": non_comment,
            "judged": positive + negative,
            "avg_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
        }


# ===========================================================================
# 温度缩放的拟合（评估阶段调用）
# ===========================================================================
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    """
    在验证集上搜索使 NLL（负对数似然）最小的温度 T。

    实现方式：用 LBFGS 优化器在 [0.5, 5.0] 范围内搜一个标量。
    这是 Guo et al. 2017 论文 "On Calibration of Modern Neural Networks"
    提出的方法，也是业界标准做法。

    返回值：(最优温度 T, 校准前后对比指标)

    关键点：必须用**验证集**拟合，不能用手上的测试集。
    否则就是把测试集信息泄漏进模型，前面辛苦做的隔离就白费了。
    """
    device = get_device()
    # 先留一份 numpy 副本，后面统计计算要用（转换后就没法拿回来了）
    labels_np = np.asarray(labels).astype(int)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)

    # 可学习参数：log(T)，用 log 参数化保证 T 恒为正
    log_T = torch.zeros(1, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / torch.exp(log_T), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).item())
    # 防止极端值把概率压得太平
    T = float(np.clip(T, 0.5, 5.0))

    # 保留一份 numpy 版的标签用于统计计算。
    # 注意：上面为了构建 tensor 已经把 labels 转过了，这里必须用独立的 numpy 副本，
    # 否则 acc[mask] 会因为 labels 是 0 维标量而报 IndexError（这个坑已经踩过一次）。
    y = np.asarray(labels_np).astype(int)

    # 计算校准前后对比
    def nll_and_ece(temp):
        p = F.softmax(logits_t / temp, dim=-1).detach().cpu().numpy()
        conf = p.max(axis=1)
        pred = p.argmax(axis=1)
        # correct 是每个样本是否预测正确的布尔数组，形状 (N,)
        correct = (pred == y).astype(float)
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        # ECE：期望校准误差，把置信度分成 15 个桶，算"置信度-准确率"的平均偏差
        ece, n = 0.0, len(y)
        for i in range(15):
            lo, hi = i / 15, (i + 1) / 15
            mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if mask.sum() > 0:
                ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
        return nll, float(ece), float(correct.mean())

    nll_b, ece_b, acc_b = nll_and_ece(1.0)
    nll_a, ece_a, acc_a = nll_and_ece(T)

    meta = {
        "temperature": round(T, 4),
        "before": {"nll": round(nll_b, 4), "ece": round(ece_b, 4), "accuracy": round(acc_b, 4)},
        "after": {"nll": round(nll_a, 4), "ece": round(ece_a, 4), "accuracy": round(acc_a, 4)},
        # 说明：accuracy 校准前后应完全一致，这是温度缩放"不改变预测"的证明
        "accuracy_unchanged": abs(acc_a - acc_b) < 1e-9,
    }
    return T, meta
