import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ========== 配置参数（可修改） ==========
DATA_PATH = 'large_data.csv'   # 训练数据文件路径（大数据集）
BATCH_SIZE = 8                  # 批次大小
EPOCHS = 3                      # 训练轮数（BERT微调推荐3-5轮）
LEARNING_RATE = 2e-5            # 学习率
MAX_LEN = 128                   # 输入文本最大长度
TEST_SIZE = 0.2                 # 验证集比例
RANDOM_SEED = 42                # 随机种子
# ======================================

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. 读取数据
print("正在读取数据...")
df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"读取到 {len(df)} 条评论")
print(f"好评数量: {len(df[df['label']==1])}")
print(f"差评数量: {len(df[df['label']==0])}")

# 2. 提取文本和标签
texts = df['review'].tolist()
labels = df['label'].tolist()

# 3. 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
)
print(f"训练集: {len(train_texts)} 条")
print(f"验证集: {len(val_texts)} 条")

# 4. 加载BERT分词器
print("加载BERT分词器...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 5. 编码函数
def encode(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    return encodings, torch.tensor(labels)

# 6. 编码数据
print("编码训练集...")
train_enc, train_labels_tensor = encode(train_texts, train_labels)
print("编码验证集...")
val_enc, val_labels_tensor = encode(val_texts, val_labels)

# 7. 创建DataLoader
train_dataset = TensorDataset(train_enc['input_ids'], train_enc['attention_mask'], train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 8. 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

# 9. 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 10. 训练
print("\n开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch in progress_bar:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

    # 验证集评估
    model.eval()
    correct = 0
    with torch.no_grad():
        input_ids = val_enc['input_ids'].to(device)
        attention_mask = val_enc['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct = (predictions.cpu() == val_labels_tensor).sum().item()
    accuracy = correct / len(val_labels_tensor)
    print(f"验证集准确率: {accuracy:.2%}")

# 11. 保存模型
print("\n保存模型...")
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
print("模型已保存到 ./sentiment_model")

# 12. 简单测试
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "好评" if pred == 1 else "差评"

print("\n测试示例：")
test_comments = [
    "这个商品质量很好",
    "物流太慢了，差评",
    "客服态度不错",
    "收到货是坏的"
]
for comment in test_comments:
    result = predict(comment)
    print(f"  '{comment}' -> {result}")

print("\n✅ 训练完成！")