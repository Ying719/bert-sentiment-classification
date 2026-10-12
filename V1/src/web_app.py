from flask import Flask, request, render_template_string, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import io
import re

# ---------------------------------------------------------------------------
# 【整理进 V1 时新增】路径适配（详细说明见 train.py 顶部）
# ---------------------------------------------------------------------------
from pathlib import Path

_V1 = Path(__file__).resolve().parent.parent
DIR_MODEL = _V1 / 'model' / 'sentiment_model'

app = Flask(__name__)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(str(DIR_MODEL))
tokenizer = BertTokenizer.from_pretrained(str(DIR_MODEL))
model.to(device)
model.eval()

# 非评论关键词（电商评论通常不会出现的词）
NON_COMMENT_KEYWORDS = [
    '吃饭', '睡觉', '打游戏', '洗澡', '上厕所', '看电视', '出去', '回家',
    '天气', '你好', '你是谁', '讲个笑话', '帮我', '请问', '谢谢',
    '哈哈', '嘻嘻', '呵呵', '嗯嗯', '哦哦', '好的', '知道了'
]

# 非评论模式
NON_COMMENT_PATTERNS = [
    r'^我[吃饭睡觉去]',  # 我去/我吃饭/我睡觉
    r'^你[是叫什么]',  # 你是谁/你叫什么
    r'^今天天气',  # 今天天气...
    r'^讲个',  # 讲个笑话/讲个故事
    r'^帮我想',  # 帮我想...
]


def is_likely_comment(text):
    """快速判断是否可能是电商评论"""
    text = text.strip()



    # 问句
    if '？' in text or '?' in text:
        return False
    if text.endswith(('吗', '嘛', '呢', '啊', '哦')):
        return False

    # 包含非评论关键词
    for kw in NON_COMMENT_KEYWORDS:
        if kw in text:
            return False

    # 匹配非评论模式
    import re
    for pattern in NON_COMMENT_PATTERNS:
        if re.search(pattern, text):
            return False

    # 必须包含电商相关词
    ecommerce_words = ['质量', '物流', '客服', '价格', '包装', '发货', '商品', '东西',
                           '好用', '不错', '满意', '快递', '售后', '划算', '收到', '坏了', '一般', '还行']
    for word in ecommerce_words:
            if word in text:
                return True

    # 如果没有电商词但长度大于10，保守起见交给模型
    if len(text) > 2:
        return True

    return False



def predict_single(text):
    """单条预测，返回 (结果, 置信度)"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pos_prob = probs[1]
        neg_prob = probs[0]
        prob_diff = abs(pos_prob - neg_prob)
        confidence = max(pos_prob, neg_prob) * 100

    if confidence >= 85 and prob_diff >= 0.5:
        sentiment = "好评" if pos_prob > neg_prob else "差评"
        emoji = "😊" if sentiment == "好评" else "😞"
        return f"{emoji} {sentiment}", confidence
    else:
        return "🤔 无法确定", confidence


def predict_batch(texts):
    """批量预测，返回结果列表和统计信息"""
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            results.append(("空行", 0))
            continue
        result, conf = predict_single(text)
        results.append((text, result, conf))

    # 统计
    stats = {
        'total': len([r for r in results if r[0] != "空行"]),
        'positive': len([r for r in results if '好评' in r[1]]),
        'negative': len([r for r in results if '差评' in r[1]]),
        'uncertain': len([r for r in results if '无法确定' in r[1]]),
    }
    return results, stats


# 判断是否为问句
def is_question(text):
    if '？' in text or '?' in text:
        return True
    if text.strip().endswith(('吗', '嘛', '呢')):
        return True
    question_words = ['怎么', '如何', '为什么', '哪里', '几', '多少', '是否']
    if any(w in text for w in question_words):
        return True
    return False


def is_valid_comment(text):
    if is_question(text):
        return False
    return True


# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>评论侦探 · AI情感分析</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .glass-card {
            background: rgba(255,255,255,0.95);
            border-radius: 40px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 25px 45px rgba(0,0,0,0.2);
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
        }
        /* 两列布局 */
        .two-columns {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .column {
            flex: 1;
            min-width: 300px;
        }
        textarea {
            width: 100%;
            padding: 18px;
            font-size: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 28px;
            resize: vertical;
            outline: none;
            font-family: inherit;
        }
        textarea:focus {
            border-color: #667eea;
        }
        button {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 40px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
            margin-right: 10px;
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            filter: brightness(1.05);
        }
        .result-box {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 28px;
            text-align: center;
        }
        .result-emoji { font-size: 4rem; }
        .result-text { font-size: 1.8rem; font-weight: bold; }
        .stats-grid {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .stat-card {
            flex: 1;
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
            padding: 15px;
            border-radius: 20px;
            text-align: center;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        .batch-results {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .result-item {
            background: #f1f3f5;
            border-radius: 12px;
            padding: 10px 15px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .result-item .text {
            flex: 1;
            font-size: 0.85rem;
            margin-right: 10px;
        }
        .result-item .tag {
            font-size: 0.85rem;
            font-weight: bold;
            min-width: 80px;
            text-align: center;
        }
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }
        .example-btn {
            background: #e9ecef;
            padding: 8px 16px;
            border-radius: 40px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .example-btn:hover {
            background: #d0d5db;
        }
        .file-upload {
            margin-top: 15px;
            padding: 15px;
            border: 2px dashed #ccc;
            border-radius: 20px;
            text-align: center;
        }
        .warning-box {
            background: #fef3c7;
            padding: 12px;
            border-radius: 20px;
            margin-top: 20px;
            font-size: 0.8rem;
            color: #92400e;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.7rem;
            color: #aaa;
        }
        canvas { max-height: 200px; }
        hr { margin: 20px 0; }
        .tab {
            display: inline-block;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 20px 20px 0 0;
            background: #e9ecef;
            margin-right: 5px;
        }
        .tab.active {
            background: white;
            font-weight: bold;
            border-bottom: 3px solid #667eea;
        }
        .tab-content {
            display: none;
            padding-top: 20px;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="glass-card">
        <h1>🕵️ 评论侦探</h1>
        <div class="sub">一句话识别好评/差评 · 基于BERT微调</div>

        <!-- 标签页切换 -->
        <div>
            <div class="tab active" onclick="switchTab('single')">📝 单条分析</div>
            <div class="tab" onclick="switchTab('batch')">📊 批量分析</div>
        </div>

        <!-- 单条分析 Tab -->
        <div id="singleTab" class="tab-content active">
            <form method="POST" id="singleForm">
                <textarea name="text" rows="3" placeholder="例如：这个产品质量真的很好，物流也快！">{{ text }}</textarea>
                <button type="submit">🔍 开始侦探</button>
            </form>
            {% if result %}
            <div class="result-box">
                <div class="result-emoji">
                    {% if '好评' in result %}😊
                    {% elif '差评' in result %}😞
                    {% else %}🤔
                    {% endif %}
                </div>
                <div class="result-text">{{ result }}</div>
                {% if confidence %}
                <div class="confidence">置信度: {{ "%.1f"|format(confidence) }}%</div>
                {% endif %}
            </div>
            {% endif %}
        </div>

        <!-- 批量分析 Tab -->
        <div id="batchTab" class="tab-content">
            <div class="two-columns">
                <div class="column">
                    <h3>✍️ 手动输入</h3>
                    <textarea id="batchText" rows="6" placeholder="每行一条评论，例如：&#10;这个产品质量很好&#10;物流太慢了&#10;一般般吧"></textarea>
                    <button onclick="analyzeBatch()">📊 开始批量分析</button>
                </div>
                <div class="column">
                    <h3>📁 上传文件</h3>
                    <div class="file-upload">
                        <input type="file" id="fileInput" accept=".csv,.txt" />
                        <button onclick="uploadFile()" style="margin-top:10px;">📂 上传并分析</button>
                        <p style="font-size:0.7rem; margin-top:8px;">支持 CSV/TXT，每行一条评论</p>
                    </div>
                </div>
            </div>

            <!-- 统计卡片 -->
            <div id="statsContainer" style="display:none;">
                <hr>
                <div class="stats-grid" id="statsGrid"></div>
                <div style="margin:20px 0;">
                    <canvas id="statsChart" width="400" height="200"></canvas>
                </div>
                <h3>📋 详细结果</h3>
                <div id="batchResults" class="batch-results"></div>
            </div>
        </div>

        <div class="examples">
            <div class="example-btn" data-text="这个手机屏幕清晰，续航很长">📱 正面</div>
            <div class="example-btn" data-text="物流太慢，客服态度差">📦 负面</div>
            <div class="example-btn" data-text="一般般吧，还行">🤔 中性</div>
            <div class="example-btn" data-text="你怎么发货这么慢？">❓ 问句</div>
            <div class="example-btn" data-text="我吃饭了">🍚 无关</div>
        </div>

        <div class="warning-box">
            💡 提示：只有模型非常确信的内容才会判断为好评/差评。<br>
            🤔 不确定的内容（包括问句、无关内容、中性评论）会显示“无法确定”。<br>
            📊 批量分析支持上传CSV/TXT文件，自动统计并可视化结果。
        </div>
        <div class="footer">
            ⚡ 基于BERT微调 | 高阈值模式（置信度≥85%且概率差≥0.5）
        </div>
    </div>
</div>

<script>
    let chartInstance = null;

    function switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        if (tab === 'single') {
            document.querySelector('.tab:first-child').classList.add('active');
            document.getElementById('singleTab').classList.add('active');
        } else {
            document.querySelectorAll('.tab')[1].classList.add('active');
            document.getElementById('batchTab').classList.add('active');
        }
    }

    async function analyzeBatch() {
        const textarea = document.getElementById('batchText');
        const texts = textarea.value.split('\\n').filter(t => t.trim().length > 0);
        if (texts.length === 0) {
            alert('请输入评论内容，每行一条');
            return;
        }
        await sendBatchRequest(texts);
    }

    async function uploadFile() {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        if (!file) {
            alert('请选择文件');
            return;
        }
        const content = await file.text();
        const texts = content.split(/\\r?\\n/).filter(t => t.trim().length > 0);
        if (texts.length === 0) {
            alert('文件中没有有效内容');
            return;
        }
        document.getElementById('batchText').value = texts.join('\\n');
        await sendBatchRequest(texts);
    }

    async function sendBatchRequest(texts) {
        const response = await fetch('/batch_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: texts })
        });
        const data = await response.json();

        // 显示统计
        const stats = data.stats;
        const statsDiv = document.getElementById('statsContainer');
        statsDiv.style.display = 'block';

        // 统计卡片
        const statsGrid = document.getElementById('statsGrid');
        statsGrid.innerHTML = `
            <div class="stat-card"><div class="stat-number">${stats.total}</div><div class="stat-label">总数</div></div>
            <div class="stat-card"><div class="stat-number" style="color:#4ECDC4">${stats.positive}</div><div class="stat-label">😊 好评</div></div>
            <div class="stat-card"><div class="stat-number" style="color:#FF6B6B">${stats.negative}</div><div class="stat-label">😞 差评</div></div>
            <div class="stat-card"><div class="stat-number" style="color:#FFB347">${stats.uncertain}</div><div class="stat-label">🤔 无法确定</div></div>
        `;

        // 更新图表
        if (chartInstance) chartInstance.destroy();
        const ctx = document.getElementById('statsChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['好评', '差评', '无法确定'],
                datasets: [{
                    data: [stats.positive, stats.negative, stats.uncertain],
                    backgroundColor: ['#4ECDC4', '#FF6B6B', '#FFB347'],
                    borderWidth: 0
                }]
            },
            options: { responsive: true, maintainAspectRatio: true }
        });

        // 详细结果
        const resultsDiv = document.getElementById('batchResults');
        resultsDiv.innerHTML = data.results.map(r => `
            <div class="result-item">
                <span class="text">${escapeHtml(r.text.substring(0, 100))}${r.text.length > 100 ? '...' : ''}</span>
                <span class="tag">${r.result}</span>
            </div>
        `).join('');
    }

    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    // 示例按钮
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            let txt = this.getAttribute('data-text');
            document.querySelector('#singleForm textarea').value = txt;
            document.querySelector('#singleForm').submit();
        });
    });
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
            # 先快速判断是否为电商评论
            if not is_likely_comment(text):
                result = "❌ 这不是电商评论"
            else:
                result, confidence = predict_single(text)
    return render_template_string(HTML_TEMPLATE, result=result, text=text)


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    texts = data.get('texts', [])
    results, stats = predict_batch(texts)

    # 格式化结果
    formatted_results = []
    for text, result, conf in results:
        if text == "空行":
            continue
        formatted_results.append({
            'text': text,
            'result': result,
            'confidence': round(conf, 1)
        })

    return jsonify({
        'results': formatted_results,
        'stats': stats
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)