"""
Dreamer Training Dashboard
- /: 메인 대시보드 (crash rate, V values, reward, entropy 그래프)
- 자동 새로고침 (10초)
"""
import sqlite3
import os
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'models', 'dreamer', 'train_log.db')

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AAA Dreamer Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
  h1 { color: #4fc3f7; margin-bottom: 5px; }
  .subtitle { color: #888; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
  .card { background: #16213e; border-radius: 10px; padding: 15px; }
  .card h3 { color: #4fc3f7; margin: 0 0 10px 0; font-size: 14px; }
  canvas { width: 100% !important; }
  .stats { display: flex; gap: 30px; margin-bottom: 15px; flex-wrap: wrap; }
  .stat { text-align: center; }
  .stat .val { font-size: 28px; font-weight: bold; }
  .stat .label { font-size: 11px; color: #888; }
  .val.green { color: #4caf50; }
  .val.red { color: #f44336; }
  .val.yellow { color: #ffeb3b; }
  .val.blue { color: #4fc3f7; }
</style>
</head>
<body>
<h1>AAA Dreamer Training Dashboard</h1>
<div class="subtitle" id="status">Loading...</div>
<div class="stats" id="stats"></div>
<div class="grid">
  <div class="card"><h3>Crash Rate (%)</h3><canvas id="crashChart"></canvas></div>
  <div class="card"><h3>V Values (Safety / Efficiency / Mission)</h3><canvas id="valueChart"></canvas></div>
  <div class="card"><h3>Axis Rewards (Safety / Efficiency / Mission)</h3><canvas id="axisRewardChart"></canvas></div>
  <div class="card"><h3>Entropy</h3><canvas id="entropyChart"></canvas></div>
</div>

<script>
const chartOpts = {
  responsive: true,
  animation: false,
  plugins: { legend: { labels: { color: '#ccc', font: {size: 11} } } },
  scales: {
    x: { ticks: { color: '#888', maxTicksLimit: 10 }, grid: { color: '#333' } },
    y: { ticks: { color: '#888' }, grid: { color: '#333' } }
  }
};

let crashChart, valueChart, rewardChart, entropyChart;

function initCharts() {
  crashChart = new Chart(document.getElementById('crashChart'), {
    type: 'line', data: { labels: [], datasets: [{ label: 'Crash Rate %', data: [], borderColor: '#f44336', borderWidth: 2, pointRadius: 0 }] }, options: chartOpts
  });
  valueChart = new Chart(document.getElementById('valueChart'), {
    type: 'line', data: { labels: [], datasets: [
      { label: 'V_safety', data: [], borderColor: '#4caf50', borderWidth: 2, pointRadius: 0 },
      { label: 'V_efficiency', data: [], borderColor: '#ffeb3b', borderWidth: 2, pointRadius: 0 },
      { label: 'V_mission', data: [], borderColor: '#9c27b0', borderWidth: 2, pointRadius: 0 },
    ] }, options: chartOpts
  });
  axisRewardChart = new Chart(document.getElementById('axisRewardChart'), {
    type: 'line', data: { labels: [], datasets: [
      { label: 'R_safety', data: [], borderColor: '#4caf50', borderWidth: 2, pointRadius: 0 },
      { label: 'R_efficiency', data: [], borderColor: '#ffeb3b', borderWidth: 2, pointRadius: 0 },
      { label: 'R_mission', data: [], borderColor: '#9c27b0', borderWidth: 2, pointRadius: 0 },
    ] }, options: chartOpts
  });
  entropyChart = new Chart(document.getElementById('entropyChart'), {
    type: 'line', data: { labels: [], datasets: [{ label: 'Entropy', data: [], borderColor: '#ff9800', borderWidth: 2, pointRadius: 0 }] }, options: chartOpts
  });
}

async function update() {
  try {
    const res = await fetch('/api/data');
    const d = await res.json();
    if (!d.steps.length) return;

    const labels = d.steps;
    crashChart.data.labels = labels;
    crashChart.data.datasets[0].data = d.crash_rate;
    crashChart.update();

    valueChart.data.labels = labels;
    valueChart.data.datasets[0].data = d.v_safety;
    valueChart.data.datasets[1].data = d.v_efficiency;
    valueChart.data.datasets[2].data = d.v_mission;
    valueChart.update();

    axisRewardChart.data.labels = labels;
    axisRewardChart.data.datasets[0].data = d.r_safety;
    axisRewardChart.data.datasets[1].data = d.r_efficiency;
    axisRewardChart.data.datasets[2].data = d.r_mission;
    axisRewardChart.update();

    entropyChart.data.labels = labels;
    entropyChart.data.datasets[0].data = d.entropy;
    entropyChart.update();

    const last = d.steps.length - 1;
    const cr = d.crash_rate[last].toFixed(1);
    const ep = d.episodes[last];
    document.getElementById('status').textContent =
      `Step ${d.steps[last]} | ${(ep/1e6).toFixed(1)}M episodes | Updated ${new Date().toLocaleTimeString()}`;

    document.getElementById('stats').innerHTML = `
      <div class="stat"><div class="val ${cr < 5 ? 'green' : cr < 10 ? 'yellow' : 'red'}">${cr}%</div><div class="label">Crash Rate</div></div>
      <div class="stat"><div class="val blue">${d.v_safety[last].toFixed(1)}</div><div class="label">V Safety</div></div>
      <div class="stat"><div class="val yellow">${d.v_efficiency[last].toFixed(1)}</div><div class="label">V Efficiency</div></div>
      <div class="stat"><div class="val" style="color:#9c27b0">${d.v_mission[last].toFixed(1)}</div><div class="label">V Mission</div></div>
      <div class="stat"><div class="val" style="color:#ff9800">${d.entropy[last].toFixed(2)}</div><div class="label">Entropy</div></div>
      <div class="stat"><div class="val blue">${d.reward[last].toFixed(1)}</div><div class="label">Reward</div></div>
    `;
  } catch(e) { console.error(e); }
}

initCharts();
update();
setInterval(update, 10000);
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/data')
def api_data():
    if not os.path.exists(DB_PATH):
        return jsonify({'steps': [], 'crash_rate': [], 'v_safety': [], 'v_efficiency': [], 'v_mission': [], 'reward': [], 'entropy': [], 'episodes': []})

    db = sqlite3.connect(DB_PATH)
    # 최대 2000개 포인트 (간격 조절)
    total = db.execute('SELECT COUNT(*) FROM dreamer_steps').fetchone()[0]
    skip = max(1, total // 2000)

    rows = db.execute(f'''
        SELECT step, mean_reward, mean_value, crashes, entropy,
               total_episodes, total_crashes, total_safe,
               COALESCE(v_safety, mean_value) as vs,
               COALESCE(v_efficiency, -10) as ve,
               COALESCE(v_mission, -10) as vm,
               COALESCE(r_safety, 0) as rs,
               COALESCE(r_efficiency, 0) as re,
               COALESCE(r_mission, 0) as rm
        FROM dreamer_steps
        WHERE step % {skip * 10} = 0
        ORDER BY step
    ''').fetchall()
    db.close()

    steps, crash_rate, v_safety, v_efficiency, v_mission = [], [], [], [], []
    r_safety, r_efficiency, r_mission = [], [], []
    entropy, episodes = [], []

    for r in rows:
        step, mean_r, mean_v, crashes, ent, tot_ep, tot_cr, tot_safe, vs, ve, vm, rs, re, rm = r
        steps.append(step)
        cr = (tot_cr / max(tot_ep, 1)) * 100
        crash_rate.append(round(cr, 2))
        v_safety.append(round(vs, 2))
        v_efficiency.append(round(ve, 2))
        v_mission.append(round(vm, 2))
        r_safety.append(round(rs, 2))
        r_efficiency.append(round(re, 2))
        r_mission.append(round(rm, 2))
        entropy.append(round(ent, 2))
        episodes.append(tot_ep)

    return jsonify({
        'steps': steps, 'crash_rate': crash_rate,
        'v_safety': v_safety, 'v_efficiency': v_efficiency, 'v_mission': v_mission,
        'r_safety': r_safety, 'r_efficiency': r_efficiency, 'r_mission': r_mission,
        'entropy': entropy, 'episodes': episodes,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5556, debug=False)
