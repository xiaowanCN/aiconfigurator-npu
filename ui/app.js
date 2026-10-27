'use strict';

/* ============================================================
 * 数据源抽象
 *   - API 模式：调用 infer-strategy-npu-server 的 /api/v1（支持新建任务）
 *   - 静态模式：直接读 results/infer-strategy-npu/index.json（只读）
 * 模式自动探测（探测 /api/v1/health）；也可用 ?source=api / ?source=static 强制。
 * 两者返回的“单任务结构”都是 run.json，因此渲染逻辑无需改动。
 * ============================================================ */
const API_BASE = '/api/v1';

// 结果根目录：默认 ../results/infer-strategy-npu；
// 用 --out-dir _verify_results 跑出来时，可在 URL 上加 ?dir=_verify_results 覆盖；
// 若由 infer-strategy-npu-server 托管（out_dir 挂在 /results），直接用服务端给的路径。
const _dirParam = new URLSearchParams(location.search).get('dir');
const _resultsBase = _dirParam ? `../${String(_dirParam).replace(/^\/+|\/+$/g, '')}` : '../results/infer-strategy-npu';
let _apiResultsUrl = null;
const resultsBase = () => _apiResultsUrl || _resultsBase;
const RESULTS_INDEX = () => `${resultsBase()}/index.json`;
const RESULTS_RUN = (id) => `${resultsBase()}/${id}/run.json`;

const _forcedSource = new URLSearchParams(location.search).get('source'); // api | static | null
let _apiMode = _forcedSource ? _forcedSource === 'api' : null; // null = 未探测

async function isApiMode() {
  if (_apiMode !== null) return _apiMode;
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
    const j = r.ok ? await r.json() : null;
    _apiMode = !!(j && j.code === 200);
    if (_apiMode && j.data && j.data.results_url) _apiResultsUrl = j.data.results_url;
  } catch (e) {
    _apiMode = false;
  }
  return _apiMode;
}

const DataSource = {
  async listTasks() {
    if (await isApiMode()) {
      const r = await fetch(`${API_BASE}/tasks?page_size=200`);
      const j = await r.json();
      return (j.data && j.data.items || []).map(apiTaskToSummary);
    }
    const r = await fetch(RESULTS_INDEX());
    if (!r.ok) throw new Error(
      `无法读取 ${RESULTS_INDEX()}。两种解法：① 启动服务后从服务端打开本页 —— ` +
      '`infer-strategy-npu-server --port 8000`，浏览器访问 http://<host>:8000/ ；' +
      `② 或在仓库根目录执行 \`python -m http.server\` 后再访问（浏览器 file:// 直接打开会因 CORS 拒绝读取本地文件）；` +
      '结果目录不是默认位置时，在 URL 上加 ?dir=<结果目录名>，如 ?dir=_verify_results。'
    );
    const j = await r.json();
    return j.tasks || [];
  },
  async getTask(id) {
    if (await isApiMode()) {
      const r = await fetch(`${API_BASE}/tasks/${encodeURIComponent(id)}`);
      const j = await r.json();
      if (j.code !== 200) throw new Error(j.msg || '任务不存在');
      return j.data.result;
    }
    const r = await fetch(RESULTS_RUN(id));
    if (!r.ok) throw new Error('找不到该任务结果: ' + id);
    return await r.json();
  }
};

function apiTaskToSummary(t) {
  // API 列表项是精简视图，这里做最小映射（详情仍走 getTask）
  return {
    task_id: t.task_id, status: t.status, created_at: t.created_at,
    model: (t.request && t.request.model) || '', device: (t.request && t.request.device) || '',
    num_devices: (t.request && t.request.num_devices), quantize: (t.request && t.request.quantize),
    framework: (t.status === 'RUNNING' ? '运行中' : ''),
  };
}

/* ============================================================
 * 工具
 * ============================================================ */
const state = { tasks: [], filtered: [], page: 1, pageSize: 8, currentFw: null, currentId: null };

/** 状态归一：run.json 用 ok/failed，HTTP 任务用 SUCCESS/FAILED/… */
function normStatus(s) {
  const m = { SUCCESS: 'ok', FAILED: 'failed', RUNNING: 'running', PENDING: 'running', CANCELLED: 'cancelled', TIMEOUT: 'timeout' };
  return m[s] || String(s || '').toLowerCase();
}

const fmt = (n, d = 2) => (n == null || isNaN(n) ? '—' : Number(n).toLocaleString('zh-CN', { maximumFractionDigits: d }));
const fmtInt = (n) => (n == null || isNaN(n) ? '—' : Number(n).toLocaleString('zh-CN'));
const len = (x) => (x && x.length ? x.length : 0);

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function shortId(id) { return id.length > 42 ? id.slice(0, 22) + '…' + id.slice(-16) : id; }

function statusTag(s) {
  const n = normStatus(s);
  if (n === 'ok') return '<span class="tag ok">成功</span>';
  if (n === 'failed') return '<span class="tag critical">失败</span>';
  if (n === 'running') return '<span class="tag info">运行中</span>';
  if (n === 'cancelled') return '<span class="tag neutral">已取消</span>';
  if (n === 'timeout') return '<span class="tag warn">超时</span>';
  return `<span class="tag neutral">${escapeHtml(s || '未知')}</span>`;
}

function metric(k, v, unit) {
  return `<div class="metric"><div class="k">${k}</div><div class="v">${v}${unit ? `<small>${unit}</small>` : ''}</div></div>`;
}

/* ============================================================
 * 列表视图
 * ============================================================ */
async function loadList() {
  showView('list');
  setCrumb('推理策略寻优', '任务记录');
  const body = document.getElementById('list-body');
  body.innerHTML = '<tr><td colspan="8" class="sub">加载中…</td></tr>';
  try {
    state.tasks = await DataSource.listTasks();
    fillModelFilter();
    applyFilters();
  } catch (e) {
    body.innerHTML = `<tr><td colspan="8" class="err-note" style="padding:14px">${escapeHtml(e.message)}</td></tr>`;
  }
}

function fillModelFilter() {
  const sel = document.getElementById('f-model');
  const models = [...new Set(state.tasks.map(t => t.model).filter(Boolean))].sort();
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m; sel.appendChild(o);
  });
}

function applyFilters() {
  const m = document.getElementById('f-model').value;
  const st = document.getElementById('f-status').value;
  const kw = document.getElementById('f-kw').value.trim().toLowerCase();
  state.filtered = state.tasks.filter(t =>
    (!m || t.model === m) &&
    (!st || normStatus(t.status) === st) &&
    (!kw || (t.task_id || '').toLowerCase().includes(kw) || (t.model || '').toLowerCase().includes(kw))
  );
  state.page = 1;
  renderList();
}

function renderList() {
  const body = document.getElementById('list-body');
  const items = state.filtered;
  if (!items.length) { body.innerHTML = '<tr><td colspan="8" class="sub">无匹配任务</td></tr>'; document.getElementById('pager').innerHTML = ''; return; }
  const totalPages = Math.max(1, Math.ceil(items.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * state.pageSize;
  const chunk = items.slice(start, start + state.pageSize);
  body.innerHTML = chunk.map(t => `
    <tr class="clickable" onclick="openDetail('${encodeURIComponent(t.task_id)}')">
      <td class="strong" title="${escapeHtml(t.task_id)}">${shortId(t.task_id)}</td>
      <td>${escapeHtml(t.model || '—')}</td>
      <td class="num">${escapeHtml(t.device || '—')} × ${fmtInt(t.num_devices)}</td>
      <td>${escapeHtml(t.quantize || '—')}</td>
      <td>${escapeHtml(t.framework || '—')}</td>
      <td>${statusTag(t.status)}</td>
      <td class="sub">${escapeHtml(t.created_at || '—')}</td>
      <td><button class="btn-text" onclick="event.stopPropagation();openDetail('${encodeURIComponent(t.task_id)}')">查看</button></td>
    </tr>`).join('');
  const pager = document.getElementById('pager');
  pager.innerHTML = totalPages > 1
    ? ['‹', ...Array.from({ length: totalPages }, (_, i) => i + 1), '›']
        .map((p, i) => `<button class="${p === state.page ? 'cur' : ''}" onclick="gotoPage(${i === 0 ? Math.max(1, state.page - 1) : i === totalPages ? Math.min(totalPages, state.page + 1) : p})">${p}</button>`).join('')
    : '';
}
function gotoPage(p) { state.page = p; renderList(); }

/* ============================================================
 * 详情视图
 * ============================================================ */
/** 打开单文件静态报告：优先走服务端接口，静态模式下直接读 report.html */
async function openReport() {
  const id = state.currentId;
  if (!id) return;
  if (await isApiMode()) { window.open(`${API_BASE}/tasks/${encodeURIComponent(id)}/report`, '_blank'); return; }
  window.open(`${resultsBase()}/${id}/report.html`, '_blank');
}

async function openDetail(encId) {
  const id = decodeURIComponent(encId);
  state.currentId = id;
  stopPendingTimer();
  showView('detail');
  setCrumb('推理策略寻优', '任务详情');
  const body = document.getElementById('detail-body');
  body.innerHTML = '<div class="notice">加载中…</div>';
  try {
    const data = await DataSource.getTask(id);
    // 任务尚在执行时 run.json 还没落盘，result 为 null —— 此时展示进度而不是崩
    if (!data || !data.frameworks) {
      renderPending(id, data);
      return;
    }
    renderDetail(data);
  } catch (e) {
    body.innerHTML = `<div class="err-box"><pre>${escapeHtml(e.message)}</pre></div>`;
  }
}

/** 任务未完成（无 run.json）时的占位视图：展示状态与进度。 */
async function renderPending(id, data) {
  const body = document.getElementById('detail-body');
  document.getElementById('detail-title').textContent = `任务 ${id}`;
  document.getElementById('detail-sub').textContent = '结果尚未生成';

  let status = (data && data.status) || 'RUNNING';
  let pg = (data && data.progress) || {};
  if (await isApiMode()) {
    try {
      const r = await fetch(`${API_BASE}/tasks/${encodeURIComponent(id)}`, { cache: 'no-store' });
      const j = await r.json();
      if (j && j.code === 200) {
        status = j.data.status || status;
        pg = j.data.progress || pg;
      }
    } catch (e) { /* 探测失败就用已有信息 */ }
  }

  const done = TERMINAL.includes(status);
  const stage = pg.label ? `${pg.label}（${pg.kind || ''}）` : (pg.stage || '—');
  const bar = pg.total
    ? `<div class="nf-running">进度：${pg.done || 0} / ${pg.total}（${pg.percent || 0}%）</div>`
    : '';
  const tip = done
    ? '任务已结束，但没有生成结果文件（run.json）。可能是失败、被取消或超时，可回到列表查看状态。'
    : 'PD 分离每组 PD-ratio 实测约 177s；<code>--pd-ratio-sizes 8 16 32</code>（9 组）约 27 分钟。页面每 10 秒自动刷新，结果出来后会自动展示。';
  body.innerHTML =
    `<div class="card"><div class="card-title"><span class="bar"></span>${done ? '任务已结束（无结果）' : '任务执行中'}</div>
      <div class="stat-grid">
        ${statItem('状态', statusTag(status), '')}
        ${statItem('当前阶段', escapeHtml(stage), '')}
      </div>
      ${bar}
      <div class="sub" style="margin-top:12px">${tip}</div>
      <div class="form-foot">
        <button class="btn-cancel" onclick="backToList()">← 返回任务记录</button>
        <button class="btn-cancel" onclick="openDetail('${encodeURIComponent(id)}')">刷新</button>
      </div>
    </div>`;

  // 未结束的任务自动轮询，结果落盘后自动切换到详情
  if (!done && await isApiMode()) {
    stopPendingTimer();
    _pendingTimer = setInterval(() => openDetail(encodeURIComponent(id)), 10000);
  }
}

function pickFramework(data) {
  const fws = (data && data.frameworks) || [];
  return fws.find(f => (f.agg && f.agg.available) || (f.disagg && f.disagg.available)) || fws[0] || {};
}
function bestOf(m) { return m ? (m.best || (m.candidates && m.candidates[0]) || null) : null; }
/** 该部署形态是否真的产出了可用配置 */
function modeReady(m) { return !!(m && m.available && bestOf(m)); }
function modeName(m) { return m === 'disagg' ? 'PD 分离（disagg）' : 'PD 混部（agg）'; }

/* ------------------------------------------------------------
 * 并行策略解析
 *   parallel_str 有两种写法，统一拆成 [{key, value, hot}]：
 *     msmodeling : "TP=1 | PP=1 | DP=8 | EP=8 | MOE-TP=1 | MOE-DP=1 | DFlash=8/acc=5"
 *     aic-npu    : "tp4pp1dp8etp1ep32"
 *   hot=true 的项（TP/PP/DP/EP/MOE-*）在结论卡里重点标记。
 * ---------------------------------------------------------- */
const PP_ALIAS = {
  TP: 'TP', PP: 'PP', DP: 'DP', EP: 'EP',
  'MOE-TP': 'MOE-TP', 'MOE-DP': 'MOE-DP', 'MOE-EP': 'MOE-EP',
  ETP: 'MOE-TP', EDP: 'MOE-DP', MOE_TP: 'MOE-TP', MOE_DP: 'MOE-DP', MOE_EP: 'MOE-EP',
  DFLASH: 'DFlash',
};
const KEY_PARALLEL = { TP: 1, PP: 1, DP: 1, EP: 1, 'MOE-TP': 1, 'MOE-DP': 1, 'MOE-EP': 1 };

function parseParallel(s) {
  if (!s) return [];
  const out = [];
  for (const it of String(s).split('|').map(x => x.trim()).filter(Boolean)) {
    const kv = /^([A-Za-z][A-Za-z0-9_ -]*?)\s*[:=]\s*(.+)$/.exec(it);
    if (kv) { out.push([kv[1].trim().toUpperCase(), kv[2].trim()]); continue; }
    // 紧凑写法（aic-npu）："tp4pp1dp8etp1ep32" —— key 只吃字母，避免贪婪回溯把 "ep32" 切成 "ep3"+"2"
    const tight = [...it.matchAll(/([A-Za-z_][A-Za-z_]*)\s*(\d+(?:\.\d+)?)/g)];
    if (tight.length) { tight.forEach(t => out.push([t[1].toUpperCase(), t[2]])); continue; }
    out.push([null, it]);
  }
  return out.map(([k, v]) => {
    const key = PP_ALIAS[k] || k || '';
    return { key: key || '配置', value: v, hot: !!KEY_PARALLEL[key] };
  });
}

function chipsHtml(list, large) {
  if (!list || !list.length) return '<span class="sub">—</span>';
  return list.map(c => `<span class="chip${c.hot ? ' hot' : ''}${large ? ' lg' : ''}">${escapeHtml(c.key)}<b>${escapeHtml(c.value)}</b></span>`).join('');
}

/** 兼容 parallel 为字典的情况（aic-npu / 部分 msmodeling 结构） */
function parallelStr(b) {
  if (!b) return '';
  if (b.parallel_str) return b.parallel_str;
  const p = b.parallel;
  if (!p) return '';
  const order = [['tp', 'TP'], ['pp', 'PP'], ['dp', 'DP'], ['ep', 'EP'], ['moe_tp', 'MOE-TP'], ['moe_dp', 'MOE-DP'], ['moe_ep', 'MOE-EP']];
  return order.filter(([k]) => p[k] != null).map(([k, label]) => `${label}=${p[k]}`).join(' | ');
}

/** disagg：拆出 P/D 两侧的并行串与实例信息（优先取 extra.prefill / extra.decode） */
function pdSides(b) {
  const ex = (b && b.extra) || {};
  const ps = ex.prefill || {}, ds = ex.decode || {};
  let pStr = ps.parallel || null, dStr = ds.parallel || null;
  if (!pStr || !dStr) {
    const m = /prefill\[([^\]]*)\][\s/]*decode\[([^\]]*)\]/i.exec((b && b.parallel_str) || '');
    if (m) { pStr = pStr || m[1]; dStr = dStr || m[2]; }
  }
  const pack = (o, str) => ({
    str, instances: o.instances, per: o.devices_per_instance,
    batch_size: o.batch_size, concurrency: o.concurrency, qps: o.qps_per_instance,
  });
  return { p: pack(ps, pStr), d: pack(ds, dStr) };
}

/* ------------------------------------------------------------
 * 一、推理策略推荐结论：agg / disagg 各出一张结论卡
 * ---------------------------------------------------------- */
function statItem(k, v, unit, cls) {
  return `<div class="stat${cls ? ' ' + cls : ''}"><div class="v">${v}${unit ? `<small>${unit}</small>` : ''}</div><div class="l">${k}</div></div>`;
}

/** 关键指标：建模估算值，刻意做成弱化排版（结论以并行策略为准）。 */
function metricsMini(b) {
  const items = [
    ['集群吞吐', fmt(b.throughput_tokens_per_s), 'tokens/s'],
    ['单卡吞吐', fmt(b.throughput_tokens_per_s_per_device), 'tokens/s'],
    ['TTFT', b.ttft_ms == null ? '—' : fmt(b.ttft_ms, 1), 'ms'],
    ['TPOT', b.tpot_ms == null ? '—' : fmt(b.tpot_ms, 1), 'ms'],
    ['并发', fmtInt(b.concurrency), 'req'],
    ['批大小', fmtInt(b.batch_size), ''],
    ['req/s', fmt(b.throughput_req_per_s, 3), ''],
    ['用卡', b.num_devices == null ? '—' : fmtInt(b.num_devices), '卡'],
  ];
  return `<div class="metrics-mini">${items.map(([k, v, u]) =>
    `<span class="mi"><i>${k}</i>${v}${u ? `<em>${u}</em>` : ''}</span>`).join('')}</div>
    <div class="sub">指标为建模估算值，仅作辅助参考；策略取舍以并行配置为准。</div>`;
}

function conclCard(mode, m, isBest) {
  const b = bestOf(m);
  if (!b) {
    return `<div class="concl-card empty"><div class="cc-head"><div class="idx">${modeName(mode)}</div></div>
      <div class="sub">本次未产出可用配置</div></div>`;
  }
  const ex = b.extra || {};
  const q = b.quantize || {};
  let paraBlock, extraPairs;

  if (mode === 'disagg') {
    const s = pdSides(b);
    paraBlock = `<div class="para-block">
      <div class="para-head">并行策略（推荐配置）</div>
      <div class="para-line lg"><span class="k">Prefill 并行</span><span class="v">${chipsHtml(parseParallel(s.p.str), true)}</span></div>
      <div class="para-line lg"><span class="k">Decode 并行</span><span class="v">${chipsHtml(parseParallel(s.d.str), true)}</span></div>
    </div>`;
    extraPairs = [
      ['PD 组合', ex.label || '—'],
      ['实例 × 卡数', `${fmtInt(s.p.instances)} × ${fmtInt(s.p.per)}卡 / ${fmtInt(s.d.instances)} × ${fmtInt(s.d.per)}卡`],
      ['P/D batch', `${fmtInt(s.p.batch_size)} / ${fmtInt(s.d.batch_size)}`],
      ['PD 比值', fmt(ex.pd_ratio, 3)],
      ['单用户吞吐', fmt(ex.tokens_per_s_per_user, 2) + ' tokens/s'],
      ['请求时延', fmt(ex.request_latency_ms, 0) + ' ms'],
    ];
  } else {
    paraBlock = `<div class="para-block">
      <div class="para-head">并行策略（推荐配置）</div>
      <div class="para-line lg"><span class="k">TP / PP / DP</span><span class="v">${chipsHtml(parseParallel(parallelStr(b)), true)}</span></div>
    </div>`;
    extraPairs = [
      ['实例 × 卡数', ex.instances && ex.devices_per_instance ? `${fmtInt(ex.instances)} × ${fmtInt(ex.devices_per_instance)}卡` : `${fmtInt(b.num_devices)} 卡`],
      ['GEMM / KV 量化', (ex.gemm_quant || ex.kvcache_quant) ? `${ex.gemm_quant || '—'} / ${ex.kvcache_quant || '—'}` : null],
      ['单卡集群吞吐', ex.tokens_per_s_gpu_cluster == null ? null : fmt(ex.tokens_per_s_gpu_cluster)],
      ['整机功耗', ex.power_w ? fmt(ex.power_w, 0) + ' W' : null],
    ].filter(x => x[1]);
  }

  const qPairs = [
    ['线性层量化', q.linear], ['非专家线性层', q.non_expert_linear], ['KV cache', q.attention],
  ].filter(x => x[1]);

  const chipList = (pairs) => pairs.map(([k, v]) => `<span class="chip">${escapeHtml(k)}<b>${escapeHtml(String(v))}</b></span>`).join('');

  return `<div class="concl-card${isBest ? ' best' : ''}">
    <div class="cc-head">
      <div class="idx">${modeName(mode)}</div>
      <div class="cc-tags">
        ${isBest ? '<span class="tag info">推荐</span>' : ''}
        ${b.meets_slo ? '<span class="tag ok">SLO 达标</span>' : '<span class="tag warn">SLO 未达标</span>'}
      </div>
    </div>
    ${paraBlock}
    <h4 class="blk">附加部署配置</h4>
    <div class="chips">${chipList(extraPairs.concat(qPairs))}</div>
    <h4 class="blk">关键指标（估算）</h4>
    ${metricsMini(b)}
  </div>`;
}

function renderConclusion(data, fw) {
  const rec = data.recommendation || {};
  const modes = [];
  if (modeReady(fw.agg)) modes.push('agg');
  if (modeReady(fw.disagg)) modes.push('disagg');
  if (!modes.length) {
    return `<div class="card"><div class="card-title"><span class="bar"></span>推理策略推荐结论</div>
      <div class="sub">本次任务没有产出可用配置，请查看“分析过程 / 错误信息”。</div></div>`;
  }
  const cards = modes.map(m => conclCard(m, m === 'agg' ? fw.agg : fw.disagg, rec.serving_mode === m)).join('');
  return `<div class="card">
    <div class="card-title"><span class="bar"></span>推理策略推荐结论 <span class="card-desc">按部署形态分别给出最优配置</span></div>
    <div class="concl-grid">${cards}</div>
    ${rec.reason ? `<div class="stat-caption">择优口径：${escapeHtml(rec.reason)}</div>` : ''}
  </div>`;
}

/* ------------------------------------------------------------
 * 候选表（排名列统一在最左，避免列错位）
 * ---------------------------------------------------------- */
function candidatesTable(cands, title) {
  if (!cands || !cands.length) return `<div class="sub" style="margin-top:12px">${title || '候选'}：无</div>`;
  const rows = cands.map((c, i) => {
    const sla = c.meets_slo ? '<span class="tag ok">达标</span>' : '<span class="tag warn">未达标</span>';
    const top = i === 0 ? ' class="top1"' : '';
    return `<tr${top}>
      <td class="strong">#${i + 1}</td>
      <td>${chipsHtml(parseParallel(parallelStr(c)))}</td>
      <td class="num">${fmt(c.throughput_tokens_per_s)}</td>
      <td class="num">${fmt(c.throughput_tokens_per_s_per_device)}</td>
      <td class="num">${c.ttft_ms == null ? '—' : fmt(c.ttft_ms, 1)}</td>
      <td class="num">${c.tpot_ms == null ? '—' : fmt(c.tpot_ms, 1)}</td>
      <td class="num">${fmtInt(c.concurrency)}</td>
      <td class="num">${fmtInt(c.batch_size)}</td>
      <td>${sla}</td>
    </tr>`;
  }).join('');
  return `<table style="margin-top:12px"><thead><tr>
    <th>排名</th><th>并行策略</th><th>集群吞吐</th><th>单卡吞吐</th><th>TTFT</th><th>TPOT</th><th>并发</th><th>批大小</th><th>SLO</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

/* 二、分析过程的折叠块：整行 summary 都能点开/收起，交互统一 */
function foldCard(title, inner, open) {
  return `<details class="card fold"${open ? ' open' : ''}>
    <summary class="card-title"><span class="bar"></span><span class="fold-text">${title}</span><span class="fold-ico"></span></summary>
    <div class="fold-body">${inner}</div>
  </details>`;
}

/** 组合列表：优先 disagg_variants；缺失时按 framework 级候选的 label 兜底，保证每个 P×D 都出现。 */
function pdVariants(fw) {
  const vs = (fw.disagg_variants || []).filter(v => v && (v.label || v.pd_ratio));
  if (vs.length) return vs;
  const byLabel = new Map();
  for (const c of fw.pd_ratio_candidates || []) {
    const key = c.label || '—';
    if (!byLabel.has(key)) byLabel.set(key, { label: key, pd_ratio: { prefill_devices: c.prefill && c.prefill.devices_per_instance, decode_devices: c.decode && c.decode.devices_per_instance }, candidates: [] });
    byLabel.get(key).candidates.push(c);
  }
  return [...byLabel.values()];
}

function renderVariantsInner(fw) {
  const vs = pdVariants(fw);
  if (!vs.length) return '<div class="sub">无 PD 分离组合</div>';
  const tabs = vs.map((v, i) => `<button class="tab ${i === 0 ? 'active' : ''}" data-vi="${i}" onclick="selectVariant(${i})">${escapeHtml(v.label || v.mode || ('#' + (i + 1)))}</button>`).join('');
  return `<div class="tabs">${tabs}</div><div id="variant-body"></div>`;
}

/* ------------------------------------------------------------
 * PD 分离候选汇总：frameworks[].pd_ratio_candidates
 * 跨全部 P×D 组合合并，已按「单 D 实例 decode 吞吐」降序
 * ------------------------------------------------------------ */
function pdSideSummary(s) {
  const ex = (s && s.extra) || {};
  const inst = ex.instances;
  const per = ex.devices_per_instance;
  if (inst && per) return `${fmtInt(inst)} 实例 × ${fmtInt(per)} 卡（共 ${fmtInt(inst * per)} 卡）`;
  if (per) return `${fmtInt(per)} 卡/实例`;
  return `${fmtInt(s && s.num_devices)} 卡`;
}

function fmtParallel(s) {
  if (!s) return '—';
  // TP=4 | PP=1 | DP=2 | EP=8 | MOE-TP=1 | MOE-DP=1  ->  TP4 PP1 DP2 EP8 MOE-TP1 MOE-DP1
  return s.split('|').map(part => {
    const kv = part.split('=');
    return kv.length === 2 ? `${kv[0].trim()}${kv[1].trim()}` : part.trim();
  }).filter(Boolean).join(' ');
}

function pdSideCells(c, i) {
  const p = c.prefill || {};
  const d = c.decode || {};
  const sideCell = s => `${fmtInt(s.instances)} × ${fmtInt(s.devices_per_instance)} 卡`;
  return `
    <td class="strong">#${i + 1}</td>
    <td class="strong">${escapeHtml(c.label || '—')}</td>
    <td class="num">${sideCell(p)}</td>
    <td class="num">${sideCell(d)}</td>
    <td class="sub nowrap">${escapeHtml(fmtParallel(p.parallel_str))}</td>
    <td class="sub nowrap">${escapeHtml(fmtParallel(d.parallel_str))}</td>
    <td class="num">${fmt(c.decode_tps_per_instance)}</td>
    <td class="num">${fmt(c.throughput_tokens_per_s)}</td>
    <td class="num">${fmt(c.throughput_tokens_per_s_per_device)}</td>
    <td class="num">${fmt(c.throughput_req_per_s, 3)}</td>
    <td class="num">${c.ttft_ms == null ? '—' : fmt(c.ttft_ms, 1)}</td>
    <td class="num">${c.tpot_ms == null ? '—' : fmt(c.tpot_ms, 1)}</td>
    <td class="num">${fmtInt(c.devices_used)}${c.num_devices ? ` / ${fmtInt(c.num_devices)}` : ''}</td>`;
}

function pdCandidatesTable(cands) {
  if (!cands || !cands.length) return '';
  const rows = cands.map((c, i) => {
    const top = i === 0 ? ' class="top1"' : '';
    return `<tr${top}>${pdSideCells(c, i)}</tr>`;
  }).join('');
  return `<div class="tbl-scroll"><table class="cand" style="margin-top:12px"><thead><tr>
    <th>排名</th><th>P×D</th><th>P 实例×卡</th><th>D 实例×卡</th><th>P 并行</th><th>D 并行</th>
    <th>decode 吞吐(单实例)</th><th>集群吞吐</th><th>单卡吞吐</th><th>req/s</th><th>TTFT</th><th>TPOT</th><th>用卡</th>
  </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function variantBody(v) {
  const b = bestOf(v) || {};
  const pf = v.prefill || {};
  const dc = v.decode || {};
  const pfStr = pf.parallel_str || parallelStr(pf);
  const dcStr = dc.parallel_str || parallelStr(dc);
  // 该组合的全部候选（条数由求解结果决定），按集群吞吐降序
  let own = v.pd_ratio_candidates || [];
  if (!own.length && v.candidates && v.candidates.length) own = v.candidates;
  if (!own.length && v.label) {
    own = ((state.currentFw && state.currentFw.pd_ratio_candidates) || []).filter(c => c.label === v.label);
  }
  const shown = own.slice().sort((a, b) => (b.throughput_tokens_per_s || 0) - (a.throughput_tokens_per_s || 0));
  return `
   <h4 class="blk">该组合最优 · 关键指标（估算）</h4>
   ${metricsMini(b)}
   <div class="pd-grid">
     <div class="pd-side">
       <div class="pd-role">Prefill 侧</div>
       <div class="chips">${chipsHtml(parseParallel(pfStr))}</div>
       <div class="sub">${fmt(pf.throughput_tokens_per_s)} tokens/s · ${pdSideSummary(pf)} · batch ${fmtInt(pf.batch_size)} · 并发 ${fmtInt(pf.concurrency)}</div>
     </div>
     <div class="pd-side">
       <div class="pd-role">Decode 侧</div>
       <div class="chips">${chipsHtml(parseParallel(dcStr))}</div>
       <div class="sub">${fmt(dc.throughput_tokens_per_s)} tokens/s · ${pdSideSummary(dc)} · batch ${fmtInt(dc.batch_size)} · 并发 ${fmtInt(dc.concurrency)}</div>
     </div>
   </div>
   ${shown.length
     ? `<h4 class="blk">该组合候选（${shown.length} 条 · 按集群吞吐降序）</h4>${pdCandidatesTable(shown)}`
     : '<div class="sub" style="margin-top:10px">该组合无候选明细</div>'}`;
}

function selectVariant(i) {
  const v = pdVariants(state.currentFw || {})[i];
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', +t.dataset.vi === i));
  if (v) document.getElementById('variant-body').innerHTML = variantBody(v);
}

/* ------------------------------------------------------------
 * 三、任务信息（沉底）：量化参数详细化，不再展示路由原因
 * ---------------------------------------------------------- */
function kvTable(pairs) {
  return `<table class="kv">${pairs.map(([k, v]) =>
    `<tr><th>${escapeHtml(k)}</th><td class="mono">${escapeHtml(String(v == null || v === '' ? '—' : v))}</td></tr>`).join('')}</table>`;
}

function renderMeta(data) {
  const req = data.request || {};
  const base = [
    ['run_id', data.run_id],
    ['模型', req.model],
    ['设备', req.device],
    ['卡数', req.num_devices],
    ['输入 / 输出长度', `${fmtInt(req.input_length)} / ${fmtInt(req.output_length)}`],
    ['TTFT / TPOT 上限', `${fmtInt(req.ttft_limit)} / ${fmtInt(req.tpot_limit)} ms`],
    ['部署形态', (req.serving_modes || []).map(modeName).join(' + ') || '—'],
    ['PD 每实例卡数', (req.pd_ratio_sizes || []).join(' / ') || '—'],
    ['TP 候选', (req.tp_sizes || []).join(' / ') || '自动'],
    ['并发度 jobs', req.jobs || '默认'],
    ['top-n', req.top_n],
    ['投机解码', req.speculative_method
      ? `${req.speculative_method} · ${req.num_speculative_tokens} tokens · acc ${req.acceptance_length}`
      : '关闭'],
    ['图编译', req.compile_model ? '开启' : '关闭'],
    ['生成时间', data.generated_at],
  ];
  const quant = [
    ['量化开关', req.quantize],
    ['专家线性层（linear）', req.quantize_linear_action],
    ['非专家线性层（non_expert_linear）', req.quantize_non_expert_linear_action],
    ['KV cache / attention', req.quantize_attention_action],
    ['实际生效（best）', quantSummary(data)],
  ];
  return `<div class="card">
    <div class="card-title"><span class="bar"></span>任务信息</div>
    ${kvTable(base)}
    <h4 class="blk">量化参数</h4>
    ${kvTable(quant)}
  </div>`;
}

/** 取推荐配置里真实生效的量化组合，便于核对“请求 vs 生效” */
function quantSummary(data) {
  const fw = pickFramework(data);
  const b = bestOf(fw.disagg) || bestOf(fw.agg);
  const q = (b && b.quantize) || {};
  if (!q || (!q.linear && !q.attention)) return '—';
  return [q.linear && `linear=${q.linear}`, q.non_expert_linear && `non_expert=${q.non_expert_linear}`, q.attention && `attention=${q.attention}`]
    .filter(Boolean).join(' · ');
}

/* ------------------------------------------------------------
 * 四、执行命令（沉底最后）
 * ---------------------------------------------------------- */
function renderCommands(data) {
  const cmds = (data.routing && data.routing.commands) || [];
  if (!cmds.length) return '';
  const items = cmds.map(c => `<li>${escapeHtml(c)}</li>`).join('');
  return foldCard(`执行命令（${cmds.length} 条）`, `<ul class="cmd-list">${items}</ul>`, false);
}

/* ------------------------------------------------------------
 * 二、分析过程
 * ---------------------------------------------------------- */
function renderAnalysis(data, fw) {
  const blocks = [];
  if (modeReady(fw.agg)) {
    blocks.push(foldCard(`PD 混部候选排名（${len(fw.agg.candidates)} 条）`, candidatesTable(fw.agg.candidates), false));
  }
  const variants = modeReady(fw.disagg) ? pdVariants(fw) : [];
  if (variants.length) {
    const labels = variants.map(v => v.label).filter(Boolean);
    const sizes = ((data.request || {}).pd_ratio_sizes || []);
    const only = variants.length === 1
      ? `<div class="notice"><b>本次只跑了 ${labels[0]} 一组</b>：任务的 PD 每实例卡数为 <code>${sizes.join(' / ') || '默认 8'}</code>，` +
        `组合数 = 卡数候选的笛卡尔积，想要 P8D8 / P8D16 / P16D8 / P16D16 请填 <code>--pd-ratio-sizes 8 16</code>（新建任务表单的“PD 每实例卡数”）。</div>`
      : '';
    blocks.push(foldCard(`PD 分离 · 各组合最优（${variants.length} 组：${labels.join(' / ')}）`, only + renderVariantsInner(fw), true));
  }
  if (!blocks.length) return '<div class="card"><div class="sub">本次没有产出候选方案。</div></div>';
  return blocks.join('');
}

function renderErrors(data) {
  let errs = data.errors || [];
  if (typeof errs === 'string') errs = [errs];
  if (!errs.length) return '';
  const items = errs.map(e => {
    if (typeof e === 'string') return `<div class="err-box"><pre>${escapeHtml(e)}</pre></div>`;
    return `<div class="err-box"><b>${escapeHtml(e.stage || '')}</b><pre>${escapeHtml(e.message || '')}</pre>${e.tip ? `<div class="err-note">${e.tip}</div>` : ''}</div>`;
  }).join('');
  return `<div class="card"><div class="card-title"><span class="bar"></span>错误信息</div>${items}</div>`;
}

function sectionTitle(t, sub) {
  return `<div class="section-title"><span class="bar"></span>${t}${sub ? ` <span class="sub">${sub}</span>` : ''}</div>`;
}

function renderDetail(data) {
  const fw = pickFramework(data);
  state.currentFw = fw;
  const req = data.request || {};
  document.getElementById('detail-title').textContent = `${req.model || '任务'} · ${fmtInt(req.num_devices)} 卡 · ${req.quantize || ''}`;
  document.getElementById('detail-sub').textContent =
    `${req.device || '—'} · 输入/输出 ${fmtInt(req.input_length)}/${fmtInt(req.output_length)} · TTFT ≤ ${fmtInt(req.ttft_limit)}ms · TPOT ≤ ${fmtInt(req.tpot_limit)}ms`;

  const failed = data.status === 'failed' || len(data.errors);
  const parts = [sectionTitle('一、推理策略推荐结论', 'by 部署形态')];
  parts.push(failed ? renderErrors(data) : renderConclusion(data, fw));
  if (!failed) {
    parts.push(sectionTitle('二、分析过程', '各场景候选方案与排序'));
    parts.push(renderAnalysis(data, fw));
  }
  parts.push(sectionTitle('三、任务信息'));
  parts.push(renderMeta(data));
  parts.push(sectionTitle('四、执行命令'));
  parts.push(renderCommands(data) || '<div class="card"><div class="sub">无</div></div>');

  document.getElementById('detail-body').innerHTML = parts.join('');
  if (!failed && len(fw.disagg_variants)) selectVariant(0);
}

/* ============================================================
 * 视图切换 / 面包屑 / 弹窗
 * ============================================================ */
let _pendingTimer = null;
function stopPendingTimer() { if (_pendingTimer) { clearInterval(_pendingTimer); _pendingTimer = null; } }
function showView(v) {
  if (v !== 'detail') stopPendingTimer();
  document.getElementById('view-list').classList.toggle('hide', v !== 'list');
  document.getElementById('view-detail').classList.toggle('hide', v !== 'detail');
}
function setCrumb(a, b) { document.getElementById('crumb').innerHTML = `<b>${escapeHtml(a)}</b> / ${escapeHtml(b)}`; }
function backToList() { loadList(); }
function openModal() {
  document.getElementById('mask').classList.add('show');
  hideNfMsg('nf-error');
  hideNfMsg('nf-preview');
  fillFormOptions();
}
function closeModal() { document.getElementById('mask').classList.remove('show'); }

/* ============================================================
 * 新建任务：表单 -> POST /api/v1/tasks -> 轮询 -> 打开详情
 * ============================================================ */
const nf = (id) => document.getElementById(id);
function hideNfMsg(id) { const el = nf(id); if (el) el.classList.add('hide'); }
function showNfError(msg) {
  const el = nf('nf-error');
  el.innerHTML = escapeHtml(msg);
  el.classList.remove('hide');
}

async function fillFormOptions() {
  if (!await isApiMode()) {
    nf('nf-hint').innerHTML =
      '当前为<b>静态只读模式</b>（未检测到 infer-strategy-npu-server），无法提交任务。' +
      '请先启动服务：<code>infer-strategy-npu-server --port 8000</code>，再从 <code>http://&lt;host&gt;:8000/</code> 打开本页。';
    const btn = nf('nf-submit');
    btn.disabled = true;
    return;
  }
  try {
    const [ms, ds] = await Promise.all([
      fetch(`${API_BASE}/models`).then(r => r.json()),
      fetch(`${API_BASE}/devices`).then(r => r.json()),
    ]);
    fillSelect(nf('nf-model'), (ms.data || []), '模型');
    fillSelect(nf('nf-device'), (ds.data || []), '设备');
  } catch (e) {
    showNfError('拉取模型/设备清单失败: ' + e.message);
  }
}

function fillSelect(sel, values, label) {
  if (sel.options.length > 1) return; // 已填充过
  sel.innerHTML = '';
  if (!values.length) sel.innerHTML = `<option value="">（无${label}）</option>`;
  values.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  });
}

function buildPayload() {
  const modes = [];
  if (nf('nf-mode-agg').checked) modes.push('agg');
  if (nf('nf-mode-disagg').checked) modes.push('disagg');
  const num = (v) => { const s = String(v == null ? '' : v).trim(); return s === '' ? null : Number(s); };
  const list = (v) => {
    const arr = String(v || '').trim().split(/[\s,]+/).filter(Boolean).map(Number);
    return arr.length ? arr : null;
  };
  const payload = {
    model: nf('nf-model').value,
    device: nf('nf-device').value,
    num_devices: num(nf('nf-num-devices').value),
    quantize: nf('nf-quantize').value,
    input_length: num(nf('nf-isl').value),
    output_length: num(nf('nf-osl').value),
    ttft_limit: num(nf('nf-ttft').value),
    tpot_limit: num(nf('nf-tpot').value),
    serving_modes: modes,
    pd_ratio_sizes: list(nf('nf-pd-sizes').value),
    tp_sizes: list(nf('nf-tp-sizes').value),
    jobs: num(nf('nf-jobs').value),
    top_n: num(nf('nf-top-n').value),
    quantize_linear_action: nf('nf-q-linear').value || null,
    quantize_attention_action: nf('nf-q-attn').value || null,
    compile_model: !!nf('nf-compile').checked,
  };
  // 清掉 null，让服务端走默认值
  Object.keys(payload).forEach(k => { if (payload[k] === null) delete payload[k]; });
  return payload;
}

function validatePayload(p) {
  if (!p.model) return '请选择模型';
  if (!p.device) return '请选择设备';
  if (!p.num_devices || p.num_devices < 1) return '卡数必须为 >= 1 的整数';
  if (!p.serving_modes || !p.serving_modes.length) return '至少选择一种部署形态（agg / disagg）';
  if (p.pd_ratio_sizes && p.pd_ratio_sizes.some(n => !(n > 0))) return 'PD 每实例卡数必须为正整数';
  if (p.tp_sizes && p.tp_sizes.some(n => !(n > 0))) return 'TP 候选必须为正整数';
  return null;
}

async function previewNewTask() {
  hideNfMsg('nf-error');
  const box = nf('nf-preview');
  const p = buildPayload();
  const err = validatePayload(p);
  if (err) { showNfError(err); box.classList.add('hide'); return; }

  box.classList.remove('hide');
  box.innerHTML = '<div class="pv-title">预览中…</div>';
  try {
    const r = await fetch(`${API_BASE}/tasks/preview`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p),
    });
    const j = await r.json();
    if (!r.ok || j.code !== 200) {
      showNfError((j.detail && j.detail.reason) || j.msg || ('预览失败 HTTP ' + r.status));
      box.classList.add('hide');
      return;
    }
    const d = j.data;
    const grid = (d.pd_ratio_grid || []).map(c => c.label).join(' / ') || '（无可用组合）';
    const warns = (d.warnings || []).map(w => `<li class="pv-warn">⚠ ${escapeHtml(w)}</li>`).join('');
    const cmds = (d.commands || []).map(c => `<li>${escapeHtml(c)}</li>`).join('');
    box.innerHTML =
      `<div class="pv-title">框架：${escapeHtml(d.framework || '—')} · ${escapeHtml(d.reason || '')}</div>` +
      `<div>PD 组合（${(d.pd_ratio_grid || []).length} 组）：${escapeHtml(grid)}</div>` +
      `<div>待下发命令：${(d.commands || []).length} 条</div>` +
      (warns ? `<ul>${warns}</ul>` : '') +
      (cmds ? `<ul>${cmds}</ul>` : '');
  } catch (e) {
    showNfError('预览请求失败: ' + e.message);
    box.classList.add('hide');
  }
}

async function submitNewTask() {
  hideNfMsg('nf-error');
  const p = buildPayload();
  const err = validatePayload(p);
  if (err) { showNfError(err); return; }

  const btn = nf('nf-submit');
  btn.disabled = true;
  const box = nf('nf-preview');
  box.classList.remove('hide');
  box.innerHTML = '<div class="pv-title"><span class="nf-running">提交中…</span></div>';

  try {
    const r = await fetch(`${API_BASE}/tasks`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p),
    });
    const j = await r.json();
    if (!r.ok || j.code !== 202) {
      const detail = j.detail && j.detail.errors
        ? (j.detail.errors || []).map(e => `${e.field}: ${e.reason}`).join('；')
        : (j.msg || '');
      throw new Error(detail || ('提交失败 HTTP ' + r.status));
    }
    const taskId = j.data.task_id;
    const grid = (j.data.pd_ratio_grid || []).map(c => c.label).join(' / ');
    const warns = (j.data.warnings || []).map(w => `<li class="pv-warn">⚠ ${escapeHtml(w)}</li>`).join('');
    box.innerHTML =
      `<div class="pv-title">已创建：${escapeHtml(taskId)}</div>` +
      (grid ? `<div>PD 组合：${escapeHtml(grid)}</div>` : '') +
      (warns ? `<ul>${warns}</ul>` : '') +
      `<div class="nf-running" id="nf-poll">等待执行…</div>`;

    await pollAndOpen(taskId);
  } catch (e) {
    showNfError(e.message);
    box.classList.add('hide');
    btn.disabled = false;
  }
}

const TERMINAL = ['SUCCESS', 'FAILED', 'CANCELLED', 'TIMEOUT'];
async function pollAndOpen(taskId) {
  const pollEl = nf('nf-poll');
  for (let i = 0; i < 720; i++) { // 最多约 1 小时（5s 一次）
    await new Promise(r => setTimeout(r, 5000));
    let j;
    try {
      const r = await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}`, { cache: 'no-store' });
      j = await r.json();
    } catch (e) {
      continue; // 网络抖动，下一轮再试
    }
    if (!j || j.code !== 200) continue;
    const st = j.data && j.data.status;
    const pg = (j.data && j.data.progress) || {};
    if (pollEl) {
      const stage = pg.label ? `${pg.label} (${pg.kind || ''})` : (pg.stage || '—');
      pollEl.textContent = `状态 ${st} · ${stage} · ${pg.done || 0}/${pg.total || 0}（${pg.percent || 0}%）`;
    }
    if (TERMINAL.includes(st)) {
      nf('nf-submit').disabled = false;
      closeModal();
      if (st === 'SUCCESS') {
        await loadList();
        await openDetail(encodeURIComponent(taskId));
      } else {
        await loadList();
        const msg = (j.data && j.data.error) || ('任务' + st);
        showNfError(String(msg));
        document.getElementById('mask').classList.add('show');
      }
      return;
    }
  }
  nf('nf-submit').disabled = false;
  if (pollEl) pollEl.textContent = '轮询超时，请刷新任务记录查看';
}

document.getElementById('f-model').addEventListener('change', applyFilters);
document.getElementById('f-status').addEventListener('change', applyFilters);
document.getElementById('f-kw').addEventListener('input', applyFilters);

/** 静态报告模式：HTML 里已内嵌 window.__REPORT_DATA__，不发任何请求。 */
function bootReport(data) {
  _apiMode = false;
  state.currentId = (data && data.run_id) || '';
  document.body.classList.add('report-mode');
  const badge = document.getElementById('src-badge');
  if (badge) badge.textContent = '静态报告（离线）';
  showView('detail');
  setCrumb('推理策略寻优', '任务详情');
  try {
    if (!data || !data.frameworks) { renderPending(data && data.run_id || '', data); return; }
    renderDetail(data);
  } catch (e) {
    document.getElementById('detail-body').innerHTML = `<div class="err-box"><pre>${escapeHtml(e.message)}</pre></div>`;
  }
}

(async () => {
  if (window.__REPORT_DATA__) { bootReport(window.__REPORT_DATA__); return; }
  const api = await isApiMode();
  document.getElementById('src-badge').textContent = api ? '数据源: API' : `数据源: ${resultsBase()}（只读）`;
  await loadList();
})();
