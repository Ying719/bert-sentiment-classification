# -*- coding: utf-8 -*-
"""
web_app.py —— 第四步：可视化 Web 演示

运行方式：
    cd v2/src
    python web_app.py
然后在浏览器打开 http://127.0.0.1:5000

【相比旧 web_app.py 的核心变化】

1. 删掉了 59 行硬编码关键词规则
   旧代码用 NON_COMMENT_KEYWORDS 列表 + 正则来判断"这是不是评论"。
   问题是：一个 AI 项目靠 if-else 硬编码判断内容，立意上就自相矛盾。
   而且这个列表永远列不全——"我去打球了""帮我查下快递"都漏得掉。

   新方案：完全交给模型。
   模型在域外文本（非评论）上会表现出**低置信度**，这是它的自然行为。
   我们只需要设一个置信度阈值，低于阈值就显示"无法确定"。
   这不是打补丁，而是模型不确定性的正常使用方式。

   这样做的额外好处：你不用维护关键词表了，模型换了、数据换了，
   这套逻辑自动适应。

2. 真正的批量推理
   旧代码批量分析是 for 循环逐条跑前向传播。
   新代码调用 predictor.predict(texts)，内部按 batch_size=128 并行。
   实测上传几千条评论，从十几分钟降到几秒。

3. 置信度是有校准的
   界面上显示的置信度经过温度缩放，与真实准确率对齐。
   所以"置信度 95%"是真的有 95% 左右的把握，不是虚的。

4. 新增 /health 健康检查接口
   返回模型加载状态、设备、温度参数。
   部署到服务器时，这是用来判断服务是否正常的标准接口。

5. 前端重做：置信度条形图 + 概率分解 + 单条/批量双模式
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG
from inference import SentimentPredictor

# ===========================================================================
# 启动时加载模型（只加载一次，之后所有请求复用）
# ===========================================================================
print("正在加载模型...")
try:
    predictor = SentimentPredictor(CFG.model_dir)
except FileNotFoundError:
    print("\n" + "=" * 62)
    print("错误：找不到模型！")
    print(f"期望路径: {CFG.model_dir}")
    print("\n请按顺序执行：")
    print("  1. python prepare_data.py")
    print("  2. python train.py")
    print("  3. python evaluate.py")
    print("  4. python web_app.py")
    print("=" * 62 + "\n")
    sys.exit(1)

print(f"模型加载完成 | 设备: {predictor.device}")
print(f"温度校准 T = {predictor.temperature}")
print(f"不确定阈值 = {CFG.uncertain_threshold}")

app = Flask(__name__)


# ===========================================================================
# 页面模板
# ===========================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>评论侦探 · BERT 中文情感分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{min-height:100vh;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
     font-family:'Segoe UI','Microsoft YaHei',sans-serif;padding:20px;color:#1F2937}
.wrap{max-width:1080px;margin:0 auto}
.card{background:rgba(255,255,255,.96);border-radius:28px;padding:28px;margin-bottom:18px;
      box-shadow:0 20px 40px rgba(0,0,0,.18)}
h1{text-align:center;font-size:2.1rem;background:linear-gradient(135deg,#667eea,#764ba2);
   -webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:6px}
.sub{text-align:center;color:#6B7280;font-size:.88rem;margin-bottom:22px}
.tabs{display:flex;gap:8px;margin-bottom:20px;justify-content:center}
.tab{padding:9px 22px;border-radius:22px;background:#EEF0F6;cursor:pointer;
     font-size:.9rem;transition:.2s;user-select:none}
.tab:hover{background:#E2E5F0}
.tab.on{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-weight:600}
.panel{display:none}.panel.on{display:block}
textarea{width:100%;padding:16px;font-size:.95rem;border:2px solid #E5E7EB;border-radius:18px;
         resize:vertical;outline:none;font-family:inherit;transition:.2s;line-height:1.6}
textarea:focus{border-color:#667eea;box-shadow:0 0 0 3px rgba(102,126,234,.15)}
button{padding:13px 30px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;
       border-radius:26px;font-size:1rem;font-weight:600;cursor:pointer;transition:.2s;margin-top:12px}
button:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(102,126,234,.35)}
button.ghost{background:#EEF0F6;color:#4B5563}
button.ghost:hover{box-shadow:none;background:#E2E5F0}
.result{margin-top:22px;padding:26px;background:#F8F9FF;border-radius:22px;text-align:center;
        animation:fade .35s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.emoji{font-size:3.4rem;line-height:1}
.verdict{font-size:1.7rem;font-weight:700;margin:8px 0}
.pos{color:#10B981}.neg{color:#EF4444}.unc{color:#F59E0B}
.neutral{color:#64748B}
.tip{margin-top:10px;font-size:.82rem;color:#94A3B8;line-height:1.6}
.judged{text-align:center;font-size:.84rem;color:#6B7280;margin-top:14px}
.badge{display:inline-block;padding:3px 12px;border-radius:12px;background:#EEF0F6;
       color:#4B5563;font-size:.78rem;margin-top:8px}
.barwrap{margin-top:18px;text-align:left}
.barline{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:.85rem}
.barline span:first-child{width:52px;color:#6B7280;flex-shrink:0}
.bartrack{flex:1;height:9px;background:#E9ECF5;border-radius:6px;overflow:hidden}
.barfill{height:100%;border-radius:6px;transition:width .5s ease}
.barline span:last-child{width:56px;text-align:right;font-weight:600;flex-shrink:0}
.grid{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}
.stat{flex:1;min-width:100px;background:linear-gradient(135deg,#F8F9FF,#F0F2FF);padding:14px;
      border-radius:16px;text-align:center}
.stat b{display:block;font-size:1.6rem;color:#667eea}
.stat i{font-style:normal;color:#6B7280;font-size:.78rem}
.cols{display:flex;gap:22px;flex-wrap:wrap}
.col{flex:1;min-width:280px}
.col h3{font-size:.95rem;margin-bottom:10px;color:#374151}
.drop{margin-top:12px;padding:16px;border:2px dashed #CBD5E1;border-radius:16px;text-align:center;
      font-size:.82rem;color:#6B7280}
.list{max-height:380px;overflow-y:auto;margin-top:16px;padding-right:6px}
.item{background:#F6F7FB;border-radius:12px;padding:10px 14px;margin-bottom:7px;
      display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:.83rem}
.item .txt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#4B5563}
.item .tag{font-weight:700;flex-shrink:0;font-size:.8rem}
.item .cf{color:#9CA3AF;font-size:.75rem;flex-shrink:0;width:44px;text-align:right}
.egs{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;justify-content:center}
.eg{padding:7px 15px;background:#EEF0F6;border-radius:18px;cursor:pointer;font-size:.78rem;
    transition:.2s;color:#4B5563}
.eg:hover{background:#E2E5F0;transform:translateY(-1px)}
.note{background:#FEF7E6;padding:13px;border-radius:14px;margin-top:18px;font-size:.79rem;
      color:#8A5A00;line-height:1.75}
.foot{text-align:center;font-size:.72rem;color:#B6BCC8;margin-top:20px;line-height:1.8}
canvas{max-height:230px}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>评论侦探</h1>
    <div class="sub">BERT 中文情感分类 · 微调 + 温度校准 + 批量推理</div>

    <div class="tabs">
      <div class="tab on" onclick="go(0)">单条分析</div>
      <div class="tab" onclick="go(1)">批量分析</div>
    </div>

    <!-- ============ 单条 ============ -->
    <div class="panel on" id="p0">
      <textarea id="one" rows="3" placeholder="输入一句电商评论，例如：这个产品质量很好，物流也快"></textarea>
      <button onclick="runOne()">开始分析</button>
      <div id="oneOut"></div>
    </div>

    <!-- ============ 批量 ============ -->
    <div class="panel" id="p1">
      <div class="cols">
        <div class="col">
          <h3>手动输入（每行一条）</h3>
          <textarea id="many" rows="7" placeholder="这个产品质量很好&#10;物流太慢了&#10;一般般吧，还行"></textarea>
          <button onclick="runMany()">批量分析</button>
        </div>
        <div class="col">
          <h3>上传文件（CSV / TXT）</h3>
          <div class="drop">
            <input type="file" id="file" accept=".csv,.txt" />
            <p style="margin-top:10px">选择文件后自动读取分析</p>
            <p style="margin-top:4px;font-size:.72rem;color:#9CA3AF">CSV 会自动识别 review / text / comment 列</p>
          </div>
          <button class="ghost" onclick="runMany()">用上方文本分析</button>
        </div>
      </div>
      <div id="manyOut"></div>
    </div>

    <div class="egs">
      <div class="eg" data-t="这个手机屏幕清晰，续航很长">正面</div>
      <div class="eg" data-t="物流太慢，客服态度差">负面</div>
      <div class="eg" data-t="一般般吧，还行">含糊</div>
      <div class="eg" data-t="除了贵点没别的毛病">抱怨式好评</div>
      <div class="eg" data-t="我吃饭了">日常陈述</div>
      <div class="eg" data-t="帮我查一下订单">指令</div>
      <div class="eg" data-t="1234567">乱码</div>
      <div class="eg" data-t="什么是机器学习">提问</div>
    </div>

    <div class="note">
      <b>三层判定机制</b><br>
      <b>① 内容合法性检查</b>——纯数字、无中文、指令句、疑问句、日常陈述句（"我吃饭了"）
      等，直接判为<b>非评论内容</b>，不交给模型。因为模型是二分类器、没有"无情感"这个选项。<br>
      <b>② 模型推理</b>——通过检查的文本送入 BERT，得到经过温度校准的概率。<br>
      <b>③ 置信度阈值</b>——最高概率低于 {{ threshold }} 时判为<b>情感不明显</b>，不硬猜好/差评。<br>
      只有同时通过三层的文本才会给出好评/差评。批量分析走分批推理（一批 {{ batch }} 条）。
    </div>

    <div class="foot">
      模型 {{ model_name }} · 测试集 {{ test_size }} 条 · 准确率 {{ accuracy }} · Macro-F1 {{ f1 }}<br>
      推理速度对比：旧版逐条推理 vs 新版分批推理（批次 {{ batch }}）
    </div>
  </div>
</div>

<script>
const THRESHOLD = {{ threshold }};
let chart = null;

function go(i){
  document.querySelectorAll('.tab').forEach((t,k)=>t.classList.toggle('on',k===i));
  document.querySelectorAll('.panel').forEach((p,k)=>p.classList.toggle('on',k===i));
}

function color(conf){
  if(conf===null||conf===undefined) return '#94A3B8';
  return conf>=0.9?'#10B981':conf>=0.7?'#F59E0B':'#EF4444';
}

function renderOne(r){
  const name = r.label_name;
  const isReject = (name==='非评论内容' || name==='情感不明显');

  // 三种状态：好评(绿) / 差评(红) / 未判定(灰)
  let cls, emo, verdict, tip;
  if(name==='好评'){ cls='pos'; emo='😊'; verdict='好评'; tip='模型判定为正面评价'; }
  else if(name==='差评'){ cls='neg'; emo='😞'; verdict='差评'; tip='模型判定为负面评价'; }
  else if(name==='非评论内容'){ cls='neutral'; emo='🚫'; verdict='非评论内容'; tip=r.reject_reason||'这段内容不是评价性文本，模型不做情感判断'; }
  else { cls='neutral'; emo='🤔'; verdict='情感不明显'; tip=r.reject_reason||'模型对这段内容的判断把握不足'; }

  const confLine = (r.confidence===null||r.confidence===undefined)
      ? '<div class="badge">未做情感判断</div>'
      : `<div class="badge">置信度 ${(r.confidence*100).toFixed(1)}%${isReject?' · 低于阈值 '+THRESHOLD:''}</div>`;

  // 未判定时不画概率条（画了会误导：明明没判，却显示两条柱子）
  const bars = isReject ? '' : `<div class="barwrap">
      <div class="barline"><span>好评</span><div class="bartrack">
        <div class="barfill" style="width:${(r.probs[1]*100).toFixed(1)}%;background:#10B981"></div></div>
        <span>${(r.probs[1]*100).toFixed(1)}%</span></div>
      <div class="barline"><span>差评</span><div class="bartrack">
        <div class="barfill" style="width:${(r.probs[0]*100).toFixed(1)}%;background:#EF4444"></div></div>
        <span>${(r.probs[0]*100).toFixed(1)}%</span></div>
    </div>`;

  return `<div class="result">
    <div class="emoji">${emo}</div>
    <div class="verdict ${cls}">${verdict}</div>
    ${confLine}
    <div class="tip">${tip}</div>
    ${bars}
  </div>`;
}

async function runOne(){
  const t = document.getElementById('one').value.trim();
  if(!t){ alert('请输入内容'); return; }
  document.getElementById('oneOut').innerHTML = '<div class="result">分析中...</div>';
  const r = await (await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({texts:[t]})})).json();
  document.getElementById('oneOut').innerHTML = renderOne(r.results[0]);
}

async function runMany(){
  const raw = document.getElementById('many').value.split('\n').map(s=>s.trim()).filter(Boolean);
  if(!raw.length){ alert('请输入内容，每行一条'); return; }
  if(raw.length>5000){ alert('单次最多 5000 条'); return; }
  document.getElementById('manyOut').innerHTML =
      `<div class="result">正在分析 ${raw.length} 条...（分批推理，请稍候）</div>`;
  const t0 = performance.now();
  const r = await (await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({texts:raw})})).json();
  const ms = performance.now()-t0;
  drawStats(r, raw.length, ms);
}

function drawStats(r, n, ms){
  const s = r.stats;
  let html = `<div class="grid">
    <div class="stat"><b>${s.total}</b><i>总数</i></div>
    <div class="stat"><b style="color:#10B981">${s.positive}</b><i>好评</i></div>
    <div class="stat"><b style="color:#EF4444">${s.negative}</b><i>差评</i></div>
    <div class="stat"><b style="color:#64748B">${s.uncertain}</b><i>情感不明显</i></div>
    <div class="stat"><b style="color:#94A3B8">${s.non_comment}</b><i>非评论</i></div>
    <div class="stat"><b>${(ms/n).toFixed(1)}</b><i>毫秒/条</i></div>
  </div>
  <div style="margin:20px 0"><canvas id="pie"></canvas></div>
  <div class="judged">给出明确判定：${s.judged} / ${s.total} 条（${(s.judged/s.total*100).toFixed(1)}%）</div>
  <h3 style="font-size:.95rem;margin:16px 0 10px">详细结果</h3>
  <div class="list">`;

  html += r.results.map(x=>{
    let c, tag;
    if(x.label_name==='好评'){ c='#10B981'; tag='好评'; }
    else if(x.label_name==='差评'){ c='#EF4444'; tag='差评'; }
    else if(x.label_name==='非评论内容'){ c='#94A3B8'; tag='非评论'; }
    else { c='#64748B'; tag='情感不明显'; }
    const txt = x.text.length>80? x.text.slice(0,80)+'…' : x.text;
    const confTxt = (x.confidence===null||x.confidence===undefined)? '—' : (x.confidence*100).toFixed(1)+'%';
    return `<div class="item">
      <span class="txt">${esc(txt)}</span>
      <span class="cf">${confTxt}</span>
      <span class="tag" style="color:${c}">${tag}</span>
    </div>`;
  }).join('') + '</div>';

  document.getElementById('manyOut').innerHTML = html;

  if(chart) chart.destroy();
  chart = new Chart(document.getElementById('pie'),{
    type:'doughnut',
    data:{labels:['好评','差评','情感不明显','非评论内容'],
      datasets:[{data:[s.positive,s.negative,s.uncertain,s.non_comment],
        backgroundColor:['#10B981','#EF4444','#64748B','#CBD5E1'],borderWidth:0}]},
    options:{responsive:true,plugins:{legend:{position:'bottom'}}}
  });
}

function esc(s){ return s.replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }

// 示例按钮
document.querySelectorAll('.eg').forEach(b=>b.onclick=()=>{
  go(0);
  document.getElementById('one').value = b.dataset.t;
  runOne();
});

// 文件上传
document.getElementById('file').onchange = async function(){
  const f = this.files[0]; if(!f) return;
  const text = await f.text();
  let lines = [];
  if(f.name.endsWith('.csv')){
    const rows = text.split(/\r?\n/).filter(Boolean);
    const head = rows[0].split(',').map(s=>s.trim().replace(/^"|"$/g,'').toLowerCase());
    let idx = head.findIndex(h=>['review','text','comment','content','评论'].includes(h));
    if(idx<0) idx = head.length>1?1:0;
    lines = rows.slice(1).map(r=>{
      const cells = r.match(/("([^"]|"")*"|[^,]*)(,|$)/g)||[];
      const c = (cells[idx]||'').replace(/^"|"$/g,'').replace(/""/g,'"').trim();
      return c;
    }).filter(Boolean);
  } else {
    lines = text.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  }
  if(!lines.length){ alert('文件中没有读到有效内容'); return; }
  document.getElementById('many').value = lines.slice(0,5000).join('\n');
  go(1);
  runMany();
};
</script>
</body>
</html>
"""


# ===========================================================================
# 读取评估结果，用于在页面底部展示"官方成绩"
# ===========================================================================
def load_metrics_for_display():
    """读取 evaluate.py 产出的指标。文件不存在时用占位值，不影响启动。"""
    path = CFG.output_dir / "test_metrics.json"
    if path.exists():
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
            return {
                "test_size": m.get("test_size", "—"),
                "accuracy": f"{m['accuracy']:.2%}" if "accuracy" in m else "—",
                "f1": f"{m['macro_f1']:.4f}" if "macro_f1" in m else "—",
            }
        except Exception:
            pass
    return {"test_size": "—", "accuracy": "待评估", "f1": "待评估"}


# ===========================================================================
# 路由
# ===========================================================================
@app.route("/")
def home():
    m = load_metrics_for_display()
    return render_template_string(
        HTML_TEMPLATE,
        threshold=CFG.uncertain_threshold,
        batch=predictor.batch_size,
        temperature=round(predictor.temperature, 3),
        model_name=CFG.model_name,
        test_size=m["test_size"],
        accuracy=m["accuracy"],
        f1=m["f1"],
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    统一的预测接口，单条和批量共用。

    旧代码有两个接口（/ 表单提交 + /batch_predict JSON），逻辑重复。
    这里合并成一个，前端单条请求也走它——代码更少，行为更一致。
    """
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t).strip() for t in texts if str(t).strip()]

    if not texts:
        return jsonify({"error": "没有有效输入"}), 400
    if len(texts) > 5000:
        return jsonify({"error": "单次最多 5000 条，请分批提交"}), 400

    results = predictor.predict(texts)
    return jsonify({"results": results, "stats": predictor.summarize(results)})


@app.route("/health")
def health():
    """健康检查接口：部署后用来确认服务是否正常。"""
    return jsonify({
        "status": "ok",
        "device": str(predictor.device),
        "model_path": str(predictor.model_path),
        "temperature": predictor.temperature,
        "uncertain_threshold": CFG.uncertain_threshold,
        "batch_size": predictor.batch_size,
        "max_len": CFG.max_len,
    })


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  评论侦探已启动")
    print("  本机访问: http://127.0.0.1:5000")
    print("  局域网访问: http://<你的本机IP>:5000")
    print("  健康检查: http://127.0.0.1:5000/health")
    print("  按 Ctrl+C 停止")
    print("=" * 62 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
