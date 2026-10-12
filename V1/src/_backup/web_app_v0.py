from flask import Flask, request, render_template_string
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

app = Flask(__name__)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
model.to(device)
model.eval()


def is_valid_comment(text):
    """判断是否为有效评论"""
    if '？' in text or '?' in text:
        return False
    question_words = ['怎么', '如何', '为什么', '哪里', '几', '多少', '是否', '吗']
    if any(w in text for w in question_words):
        return False
    if len(text.strip()) < 2:
        return False
    return True


def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
        confidence = probs[pred] * 100

    if confidence < 90:
        return "🤔 无法确定（置信度偏低）"
    return "😊 好评" if pred == 1 else "😞 差评"


HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 评论侦探 🕵️‍♂️</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', 'Poppins', 'Microsoft YaHei', sans-serif;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 40px;
            padding: 30px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 25px 45px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .glass-card:hover {
            transform: scale(1.01);
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 10px;
        }
        .sub {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }
        textarea {
            width: 100%;
            padding: 18px;
            font-size: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 28px;
            resize: vertical;
            font-family: inherit;
            transition: 0.3s;
            outline: none;
        }
        textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
        }
        .char-count {
            text-align: right;
            font-size: 0.8rem;
            color: #888;
            margin-top: 8px;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 40px;
            font-size: 1.2rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: 0.2s;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            filter: brightness(1.05);
        }
        .result-box {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 28px;
            text-align: center;
            transition: 0.3s;
            border-left: 5px solid #667eea;
        }
        .result-emoji {
            font-size: 4rem;
            margin-bottom: 10px;
        }
        .result-text {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 10px 0;
        }
        .confidence {
            font-size: 0.9rem;
            color: #666;
        }
        .examples {
            margin-top: 25px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }
        .example-btn {
            background: #e9ecef;
            padding: 8px 16px;
            border-radius: 40px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: 0.2s;
        }
        .example-btn:hover {
            background: #d0d5db;
            transform: translateY(-1px);
        }
        .history {
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .history-title {
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 10px;
        }
        .history-item {
            background: #f1f3f5;
            border-radius: 20px;
            padding: 10px 15px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.7rem;
            color: #aaa;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px);}
            to { opacity: 1; transform: translateY(0);}
        }
        .animate {
            animation: fadeIn 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="glass-card">
        <h1>🕵️ 评论侦探</h1>
        <div class="sub">输入一句评论，我将帮你判断是好评还是差评</div>

        <form method="POST" id="sentimentForm">
            <textarea name="text" id="commentInput" rows="3" placeholder="例如：这个产品质量真的很好，物流也快！">{{ text }}</textarea>
            <div class="char-count"><span id="charCount">0</span> 字符</div>
            <button type="submit">🔍 开始侦探</button>
        </form>

        {% if result %}
        <div class="result-box animate">
            <div class="result-emoji">{{ '😊' if '好评' in result else '😞' if '差评' in result else '❓' }}</div>
            <div class="result-text">{{ result }}</div>
        </div>
        {% endif %}
        
        <div style="background:#fef3c7; padding:10px; border-radius:20px; margin-top:15px; font-size:0.8rem; color:#92400e;">
        💡 提示：本工具专为电商评论设计，请输入如“这个产品质量很好”等评价。非购物类文本结果可能不准确。
        </div>

        <div class="examples">
            <div class="example-btn" data-text="这个手机屏幕清晰，续航很长">📱 正面例子</div>
            <div class="example-btn" data-text="物流太慢，客服态度差">📦 负面例子</div>
            <div class="example-btn" data-text="一般般吧，还行">🤔 中性例子</div>
            <div class="example-btn" data-text="你怎么发货这么慢？">❓ 问句测试</div>
        </div>

        <div class="history" id="historyDiv" style="display: none;">
            <div class="history-title">📜 最近判断记录</div>
            <div id="historyList"></div>
        </div>

        <div class="footer">
            ⚡ 基于BERT微调 | 训练数据10,800条 | 置信度&lt;85%显示无法确定
        </div>
    </div>

    <script>
        // 字符计数
        const textarea = document.getElementById('commentInput');
        const charSpan = document.getElementById('charCount');
        textarea.addEventListener('input', function() {
            charSpan.innerText = this.value.length;
        });
        charSpan.innerText = textarea.value.length;

        // 示例按钮填充
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                textarea.value = this.getAttribute('data-text');
                charSpan.innerText = textarea.value.length;
                // 自动提交表单（可选）
                // document.getElementById('sentimentForm').submit();
            });
        });

        // 历史记录功能 (使用localStorage)
        function saveToHistory(text, result) {
            let history = JSON.parse(localStorage.getItem('commentHistory') || '[]');
            history.unshift({ text: text.substring(0, 40), result: result, time: new Date().toLocaleTimeString() });
            if (history.length > 5) history.pop();
            localStorage.setItem('commentHistory', JSON.stringify(history));
            displayHistory();
        }

        function displayHistory() {
            let history = JSON.parse(localStorage.getItem('commentHistory') || '[]');
            const container = document.getElementById('historyList');
            const historyDiv = document.getElementById('historyDiv');
            if (history.length === 0) {
                historyDiv.style.display = 'none';
                return;
            }
            historyDiv.style.display = 'block';
            container.innerHTML = history.map(item => `
                <div class="history-item">
                    <span>${item.text}...</span>
                    <span style="font-weight:bold;">${item.result}</span>
                </div>
            `).join('');
        }

        // 当有结果时保存
        {% if result %}
        saveToHistory("{{ text }}", "{{ result }}");
        {% endif %}
        displayHistory();
    </script>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    text = ""
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text.strip():
            if not is_valid_comment(text):
                result = "❓ 无法判断（这不是评论）"
            else:
                result = predict(text)
    return render_template_string(HTML, result=result, text=text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)