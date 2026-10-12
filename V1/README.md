# V1 —— BERT 中文情感分类（第一版）

> 这是本项目**答辩时提交的第一版**完整归档，原样保留。
> 后续的优化升级版在隔壁 `../v2/`，两版可逐文件对照。

---

## 一、这一版是什么

用 `bert-base-chinese` 预训练模型微调，做中文电商评论的**好评 / 差评**二分类，
配一个 Flask 网页界面（单条分析 + 批量分析 + 文件上传），
以及一套可视化与错误案例分析脚本。

**已完成过课程/项目答辩。** 本目录是当时的成果快照。

---

## 二、目录结构

```
V1/
├── README.md                    本文件
│
├── src/                         源码（6 个脚本）
│   ├── train.py                 训练脚本：微调 BERT
│   ├── web_app.py               Flask 网页服务（单条 + 批量 + 上传 + 图表）
│   ├── chat_predict.py          命令行交互式预测
│   ├── predict_batch.py         批量预测 + 自动出图（读 my_comment.csv）
│   ├── visualize.py             在带标签数据集上评估：混淆矩阵、错误案例
│   ├── download_data.py         从 HuggingFace 镜像重新下载数据集
│   └── _backup/
│       ├── web_app_v0.py        最早的网页版本（原名 web.txt，保留备查）
│       └── original/            最原始的代码文本，未经任何路径修改（纯存档）
│           ├── train.py
│           ├── web_app.py
│           ├── chat_predict.py
│           ├── predict_batch.py
│           ├── visualize.py
│           └── download_data.py
│
├── data/                        数据
│   ├── large_data.csv           主数据集 10,800 条（含 1,703 条重复）
│   ├── large_data2.csv          从上一份切出的子集 7,766 条
│   ├── data.csv                 早期手工构造的小样本 102 条
│   └── my_comment.csv           自写的测试评论 60 条（无标签）
│
├── model/
│   └── sentiment_model/         微调后的模型（391 MB）
│       ├── pytorch_model.bin    权重
│       ├── config.json
│       ├── vocab.txt
│       └── tokenizer_config.json 等
│
├── results/                     全部产出结果
│   ├── confusion_matrix.png     混淆矩阵
│   ├── label_distribution.png   真实/预测标签分布对比
│   ├── evaluation_results.csv   全部预测明细（10,800 条）
│   ├── wrong_cases.csv          错误案例清单
│   ├── my_comments_result.csv   60 条自写评论的预测结果
│   ├── prediction_pie.png       预测结果饼图
│   ├── prediction_barchart.png  预测结果条形图
│   └── length_sentiment.png     评论长度与情感关系箱线图
│
└── deploy/                      部署相关
    ├── cloudflared.exe          Cloudflare Tunnel 内网穿透工具（53 MB）
    └── web_adress.txt           当时生成过的公网访问地址
```

---

## 三、怎么运行

### 前提：Python 环境

虚拟环境 `bert_env` **保留在项目根目录**（`../../bert_env/`），没有移进来。

为什么不移：虚拟环境内部记录的是**硬编码的绝对路径**
（`pyvenv.cfg`、`Scripts/activate`、每个可执行文件的 shebang 都写着原路径），
一旦移动就会损坏，只能重装。

环境内容：Python 3.10.11 + torch 2.0.1+cu118 + transformers 4.30.0 + Flask 3.1.3。

### 运行命令

```bash
cd V1/src

# ① 训练（约 10~20 分钟，有 GPU）
../../bert_env/Scripts/python.exe train.py

# ② 批量预测自写的 60 条评论，并重新生成图表
../../bert_env/Scripts/python.exe predict_batch.py

# ③ 在带标签数据集上评估（出混淆矩阵和错误案例）
../../bert_env/Scripts/python.exe visualize.py

# ④ 启动网页（浏览器打开 http://127.0.0.1:5000）
../../bert_env/Scripts/python.exe web_app.py

# ⑤ 命令行交互预测
../../bert_env/Scripts/python.exe chat_predict.py
```

---

## 四、整理时做了什么改动

**只改了路径，没动任何业务逻辑。** 原因：文件从项目根目录移入 `V1/src/` 后，
脚本里原来写死的相对路径（如 `'large_data.csv'`、`'./sentiment_model'`）
会指向 src 目录而非文件实际位置，脚本会报"找不到文件"。

每个脚本顶部都加了一个统一格式的路径适配块，标记为
`【整理进 V1 时新增】路径适配`，按脚本自身位置反推目录，所以从任何地方执行都正确：

```python
_V1 = Path(__file__).resolve().parent.parent
DIR_DATA = _V1 / 'data'
DIR_MODEL = _V1 / 'model' / 'sentiment_model'
DIR_RESULT = _V1 / 'results'
```

被替换的写死路径一览：

| 文件 | 原来 | 现在 |
|---|---|---|
| `train.py` | `'large_data.csv'` | `data/large_data.csv` |
| `train.py` | `'./sentiment_model'`（存模型） | `model/sentiment_model/` |
| `visualize.py` | `'large_data.csv'` | `data/large_data.csv` |
| `visualize.py` | `'./sentiment_model'` | `model/sentiment_model/` |
| `visualize.py` | `'confusion_matrix.png'` 等 4 处 | `results/` |
| `chat_predict.py` | `'./sentiment_model'` | `model/sentiment_model/` |
| `web_app.py` | `'./sentiment_model'` | `model/sentiment_model/` |
| `predict_batch.py` | `'./my_comments/my_comment.csv'` | `data/my_comment.csv` |
| `predict_batch.py` | `'./my_comments/xxx.png'` 等 4 处 | `results/` |
| `download_data.py` | `'large_data.csv'` | `data/large_data.csv` |

原来的 `my_comments/` 目录被拆开归位：输入数据进 `data/`，输出结果进 `results/`。

---

## 五、整理时发现的一个问题（值得知道）

`results/my_comments_result.csv` 与当前 `model/sentiment_model/` 的预测结果
**对不上**——60 条里有 4 条预测不同：

| 评论 | 归档记录 | 用当前模型重跑 |
|---|---|---|
| 质量一般，用了几次感觉不太行。 | 差评 | 好评 |
| 很透气，夏天穿不闷热。 | 差评 | 好评 |
| 很耐摔，孩子摔了几次没事。 | 差评 | 好评 |
| 包装精美，送人很有面子。 | 差评 | 好评 |

**原因是时间戳对不上：**

```
results/my_comments_result.csv    2026-05-25 22:59   ← 预测结果
model/sentiment_model/...bin      2026-05-25 23:08   ← 模型权重
```

模型权重在结果文件的 **9 分钟之后**被覆盖过。
也就是说，这批归档的预测结果来自**一个更早的模型版本，那份权重现在已经不存在了**。

**处理方式：保留原始结果文件不动**（归档的意义就是忠实记录当时的状态），
只在本文档里记下这个发现，供你日后查阅。

> 这件事顺带说明了一个技术点：神经网络在概率接近 0.5 的**边界样本**上，
> 判断本身就不稳定，换一版权重或换个运行设备（CPU/GPU）都可能翻转。
> v2 版本为此固定了 Python/NumPy/PyTorch/CUDA 四层随机种子并启用 cuDNN 确定性模式。

---

## 六、这一版已知的不足

整理时做的代码审查发现了若干问题，**这些正是 v2 要解决的目标**：

1. **数据泄漏**：`large_data.csv` 内含 1,703 条重复；
   评估用的数据集与训练数据重叠，导致汇报的准确率偏高。
2. **没有独立测试集**：只有 train/val 两路，用验证集汇报成绩。
3. **阈值魔改**：`web_app.py` 用 `置信度≥85% 且 概率差≥0.5` 两个硬编码条件
   造出"无法确定"这一类，掩盖了模型不确定能力不足的问题。
4. **关键词硬编码**：59 行 `NON_COMMENT_KEYWORDS` 手工规则判断输入是否为评论，
   列不全且与"用 AI 判断"的立意矛盾。
5. **逐条推理**：`predict_batch` 内部是 for 循环，每条做一次完整前向传播，
   GPU 利用率极低。
6. **训练配方粗糙**：恒定学习率、无混合精度、无早停、固定跑 3 轮、
   全量固定 padding、未固定随机种子。

详细的问题分析与改进方案，见 `../v2/优化说明.md`。

---

## 七、版本对照

| | V1（本目录） | V2（`../v2/`） |
|---|---|---|
| 数据泄漏 | 存在 | 已消除（去重 + 三路隔离） |
| 测试集 | 无 | 910 条独立测试集 |
| 主指标 | Accuracy | Macro-F1 + ROC-AUC |
| 置信度 | 未校准 | 温度缩放校准 |
| 域外识别 | 59 行关键词表 | 模型置信度阈值 |
| 批量推理 | 逐条前向传播 | 分批推理（快 12~20 倍） |
| 训练 | 手写循环 | HF Trainer + FP16 + 早停 |
| 可复现 | 未固定种子 | 四层种子固定 |
