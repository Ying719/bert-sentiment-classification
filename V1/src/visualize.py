import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------------------------------------------
# 【整理进 V1 时新增】路径适配（详细说明见 train.py 顶部）
# ---------------------------------------------------------------------------
from pathlib import Path

_V1 = Path(__file__).resolve().parent.parent
DIR_DATA = _V1 / 'data'
DIR_MODEL = _V1 / 'model' / 'sentiment_model'
DIR_RESULT = _V1 / 'results'

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
print("正在加载数据...")
df = pd.read_csv(str(DIR_DATA / 'large_data.csv'), encoding='utf-8')
print(f"读取到 {len(df)} 条评论")

# 2. 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model = BertForSequenceClassification.from_pretrained(str(DIR_MODEL))
tokenizer = BertTokenizer.from_pretrained(str(DIR_MODEL))
model.to(device)
model.eval()

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

# 4. 预测
texts = df['review'].tolist()
true_labels = df['label'].tolist()
print("正在预测...")
predictions = predict_batch(texts)
print("预测完成！")

# 5. 混淆矩阵
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['差评', '好评'],
            yticklabels=['差评', '好评'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵 - BERT情感分类')
plt.tight_layout()
plt.savefig(str(DIR_RESULT / 'confusion_matrix.png'), dpi=150)
plt.show()
print("混淆矩阵已保存到 results/confusion_matrix.png")

# 6. 分类报告
print("\n分类报告:")
print(classification_report(true_labels, predictions, target_names=['差评', '好评']))

# 7. 准确率（修复后的版本）
correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
accuracy = correct / len(true_labels)
print(f"\n总体准确率: {accuracy:.2%}")

# 8. 标签分布
plt.figure(figsize=(10, 4))

# 子图1：真实标签分布
plt.subplot(1, 2, 1)
true_counts = df['label'].value_counts().sort_index()
bars1 = plt.bar(['差评', '好评'], true_counts.values, color=['#FF6B6B', '#4ECDC4'])
plt.title('真实标签分布', fontsize=12)
plt.ylabel('数量')
# 在柱子上显示数值
for bar, val in zip(bars1, true_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(val), ha='center', va='bottom', fontsize=11)

# 子图2：预测标签分布
plt.subplot(1, 2, 2)
pred_counts = pd.Series(predictions).value_counts().sort_index()
bars2 = plt.bar(['差评', '好评'], pred_counts.values, color=['#FF6B6B', '#4ECDC4'])
plt.title('预测标签分布', fontsize=12)
plt.ylabel('数量')
# 在柱子上显示数值
for bar, val in zip(bars2, pred_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(val), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(str(DIR_RESULT / 'label_distribution.png'), dpi=150)
plt.show()
print("标签分布图已保存到 results/label_distribution.png")

# 9. 【重点】错误案例标注与保存
df_results = pd.DataFrame({
    'index': df.index,  # 加上原始行号，方便你定位
    'review': texts,
    'true_label': true_labels,
    'predicted_label': predictions
})
wrong_cases = df_results[df_results['true_label'] != df_results['predicted_label']].copy()
print(f"\n错误案例数量: {len(wrong_cases)}/{len(df_results)}")

if len(wrong_cases) > 0:
    print("\n===== 标注的错误案例（全部）=====")
    for idx, row in wrong_cases.iterrows():
        true_sent = "好评" if row['true_label'] == 1 else "差评"
        pred_sent = "好评" if row['predicted_label'] == 1 else "差评"
        print(f"【行号 {row['index']}】评论: {row['review']}")
        print(f"真实: {true_sent} → 预测: {pred_sent}\n")

    # 额外保存一份错误案例文件，方便你后续分析
    wrong_cases.to_csv(str(DIR_RESULT / 'wrong_cases.csv'), index=False, encoding='utf-8-sig')
    print("所有错误案例已保存到 results/wrong_cases.csv")

# 10. 保存全部结果
df_results.to_csv(str(DIR_RESULT / 'evaluation_results.csv'), index=False, encoding='utf-8-sig')
print("详细结果已保存到 results/evaluation_results.csv")
print("\n✅ 可视化与标注完成！")
