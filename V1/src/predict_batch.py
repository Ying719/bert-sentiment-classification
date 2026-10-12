import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# 【整理进 V1 时新增】路径适配（详细说明见 train.py 顶部）
# 原脚本把输入和输出都堆在 ./my_comments/ 里；整理后已拆分为：
#   输入数据 -> data/my_comment.csv
#   输出结果 -> results/（图与 CSV）
# ---------------------------------------------------------------------------
from pathlib import Path

_V1 = Path(__file__).resolve().parent.parent
DIR_DATA = _V1 / 'data'
DIR_MODEL = _V1 / 'model' / 'sentiment_model'
DIR_RESULT = _V1 / 'results'

# 创建保存目录（如果不存在）
os.makedirs(DIR_RESULT, exist_ok=True)

# 1. 加载模型
print("正在加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(str(DIR_MODEL))
tokenizer = BertTokenizer.from_pretrained(str(DIR_MODEL))
model.to(device)
model.eval()
print(f"模型加载完成，使用设备: {device}")

# 2. 读取无标签评论（输入数据在 data/my_comment.csv）
print("\n正在读取评论...")
df = pd.read_csv(str(DIR_DATA / 'my_comment.csv'), encoding='utf-8')
comments = df['review'].tolist()
print(f"读取到 {len(comments)} 条评论")

# 3. 预测函数
def predict_batch(texts, batch_size=16):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
    return all_preds

# 4. 批量预测
print("\n正在预测...")
predictions = predict_batch(comments)
df['predicted_label'] = predictions
df['sentiment'] = df['predicted_label'].apply(lambda x: '好评' if x == 1 else '差评')

# 5. 显示预测结果
print("\n" + "="*50)
print("预测结果")
print("="*50)
for i, row in df.iterrows():
    print(f"{i+1}. {row['review']}")
    print(f"   → {row['sentiment']}\n")

# ========== 可视化1：预测结果饼图 ==========
plt.figure(figsize=(6, 5))
counts = df['sentiment'].value_counts()
colors = ['#4ECDC4', '#FF6B6B']
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('预测结果分布', fontsize=14)
plt.tight_layout()
plt.savefig(str(DIR_RESULT / 'prediction_pie.png'), dpi=150)
plt.show()
print("饼图已保存到 results/prediction_pie.png")

# ========== 可视化2：预测结果条形图 ==========
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='sentiment', palette=['#4ECDC4', '#FF6B6B'], order=['好评', '差评'])
plt.title('好评/差评数量分布', fontsize=14)
plt.xlabel('情感类别')
plt.ylabel('数量')
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(str(DIR_RESULT / 'prediction_barchart.png'), dpi=150)
plt.show()
print("条形图已保存到 results/prediction_barchart.png")

# ========== 可视化3：评论长度与情感关系 ==========
df['review_length'] = df['review'].str.len()
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='sentiment', y='review_length', palette=['#4ECDC4', '#FF6B6B'], order=['好评', '差评'])
plt.title('评论长度与情感关系', fontsize=14)
plt.xlabel('情感类别')
plt.ylabel('评论长度（字符数）')
plt.tight_layout()
plt.savefig(str(DIR_RESULT / 'length_sentiment.png'), dpi=150)
plt.show()
print("评论长度分析图已保存到 results/length_sentiment.png")

# 6. 保存结果到 results/my_comments_result.csv
df.to_csv(str(DIR_RESULT / 'my_comments_result.csv'), index=False, encoding='utf-8-sig')
print("\n预测结果已保存到 results/my_comments_result.csv")

# 7. 统计信息
print("\n" + "="*50)
print("统计信息")
print("="*50)
print(f"总评论数: {len(df)}")
print(f"好评数量: {len(df[df['predicted_label']==1])}")
print(f"差评数量: {len(df[df['predicted_label']==0])}")
if len(df) > 0:
    print(f"好评比例: {len(df[df['predicted_label']==1])/len(df)*100:.1f}%")

print("\n✅ 预测和可视化完成！")
print("生成的文件（位于 V1/results/ 文件夹）：")
print("  1. my_comments_result.csv - 详细预测结果")
print("  2. prediction_pie.png - 预测结果饼图")
print("  3. prediction_barchart.png - 预测结果条形图")
print("  4. length_sentiment.png - 评论长度与情感关系图")