import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型
print("正在加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
model.to(device)
model.eval()
print("模型加载完成！\n")


# 预测函数
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "好评 😊" if pred == 1 else "差评 😞"


# 交互界面
print("=" * 50)
print("  情感分析助手 - BERT 评论判断")
print("=" * 50)
print("输入评论，我会告诉你这是好评还是差评")
print("输入 'exit' 或 'quit' 退出\n")

while True:
    text = input("请输入评论: ")
    if text.lower() in ['exit', 'quit', '退出']:
        print("\n再见！👋")
        break
    if text.strip() == "":
        print("请输入内容！\n")
        continue

    result = predict(text)
    print(f"判断结果: {result}\n")