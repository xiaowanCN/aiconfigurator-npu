'use strict';

/* ============================================================
 * 数据源抽象：ResultsSource（当前） / ApiSource（?source=api）
 * 两者返回的“单任务结构”都是 run.json，因此渲染逻辑无需改动。
 * ============================================================ */
const useApi = new URLSearchParams(location.search).get('source') === 'api';
const API_BASE = '/api/v1';
const RESULTS_INDEX = '../results/infer-strategy-npu/index.json';
const RESULTS_RUN = (id) => `../results/infer-strategy-npu/${id}/run.json`;

const DataSource = {
  async listTasks() {
    if (useApi) {
      const r = await fetch(`${API_BASE}/tasks?page_size=200`);
      const j = await r.json();
      return (j.data && j.data.items || []).map(apiTaskToSummary);
    }
    const r = await fetch(RESULTS_INDEX);
    if (!r.ok) throw new Error('无法读取 results/index.json：请用 `python -m http.server` 在仓库根目录启动后再访问本页面（浏览器 file:// 直接打开会因 CORS 拒绝读取本地文件）。');
    const j = await r.json();
    return j.tasks || [];
  },
  async getTask(id) {
    if (useApi) {
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
const state = { tasks: [], filtered: [], page: 1, pageSize: 8, currentFw: null };

const fmt = (n, d = 2) => (n == null || isNaN(n) ? '—' : Number(n).toLocaleString('zh-CN', { maximumFractionDigits: d }));
const fmtInt = (n) => (n == null || isNaN(n) ? '—' : Number(n).toLocaleString('zh-CN'));
const len = (x) => (x && x.length ? x.length : 0);

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function shortId(id) { return id.length > 42 ? id.slice(0, 22) + '…' + id.slice(-16) : id; }

function statusTag(s) {
  if (s === 'ok' || s === 'SUCCESS') return '<span class="tag tag-green">成功</span>';
  if (s === 'failed' || s === 'FAILED') return '<span class="tag tag-red">失败</span>';
  if (s === 'RUNNING') return '<span class="tag tag-blue">运行中</span>';
  return `<span class="tag tag-gray">${escapeHtml(s || '未知')}</span>`;
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
    (!st || t.status === st) &&
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
async function openDetail(encId) {
  const id = decodeURIComponent(encId);
  showView('detail');
  setCrumb('推理策略寻优', '任务详情');
  const body = document.getElementById('detail-body');
  body.innerHTML = '<div class="notice">加载中…</div>';
  try {
    const data = await DataSource.getTask(id);
    renderDetail(data);
  } catch (e) {
    body.innerHTML = `<div class="err-box"><pre>${escapeHtml(e.message)}</pre></div>`;
  }
}

function pickFramework(data) {
  const fws = data.frameworks || [];
  return fws.find(f => (f.agg && f.agg.available) || (f.disagg && f.disagg.available)) || fws[0] || {};
}
function bestOf(m) { return m ? (m.best || (m.candidates && m.candidates[0]) || null) : null; }
function parallelStr(b) {
  if (!b) return '—';
  if (b.parallel_str) return b.parallel_str;
  if (b.parallel) return `TP=${b.parallel.tp} | PP=${b.parallel.pp} | DP=${b.parallel.dp}`;
  return '—';
}

function renderMeta(data) {
  const req = data.request || {};
  const rt = data.routing || {};
  const items = [
    ['run_id', data.run_id], ['模型', req.model], ['设备', req.device], ['卡数', req.num_devices],
    ['量化', req.quantize], ['输入/输出长度', `${req.input_length} / ${req.output_length}`],
    ['TTFT/TPOT 上限', `${req.ttft_limit} / ${req.tpot_limit} ms`], ['路由原因', rt.reason || '—'],
  ];
  return `<div class="card"><div class="card-title"><span class="bar"></span>任务信息</div>
    <div class="meta-grid">${items.map(([k, v]) => `<div class="meta"><div class="k">${k}</div><div class="v">${escapeHtml(String(v == null ? '—' : v))}</div></div>`).join('')}</div></div>`;
}

function renderRecommendation(data) {
  const rec = data.recommendation || {};
  const b = rec.best || {};
  const modeLabel = rec.serving_mode === 'disagg' ? 'PD 分离（disagg）' : '聚合模式（agg）';
  return `<div class="reco">
    <div class="reco-left">
      <div class="mode">推荐部署模式</div>
      <div class="big">${modeLabel}</div>
      <div class="reco-badge">${escapeHtml(rec.framework || '')}</div>
    </div>
    <div class="reco-right">
      <div class="reco-metrics">
        ${metric('集群吞吐', fmt(b.throughput_tokens_per_s), 'tokens/s')}
        ${metric('单卡吞吐', fmt(b.throughput_tokens_per_s_per_device), 'tokens/s')}
        ${metric('TTFT', b.ttft_ms == null ? '—' : fmt(b.ttft_ms, 1), 'ms')}
        ${metric('TPOT', b.tpot_ms == null ? '—' : fmt(b.tpot_ms, 1), 'ms')}
        ${metric('并发', fmtInt(b.concurrency), 'req')}
        ${metric('批大小', fmtInt(b.batch_size), '')}
      </div>
      <div class="reco-why"><span class="tag tag-blue">${escapeHtml(parallelStr(b))}</span><span class="sub">${escapeHtml(rec.reason || '')}</span></div>
    </div>
  </div>`;
}

function cfgCard(title, m) {
  const b = bestOf(m);
  if (!b) return `<div class="card"><div class="card-title"><span class="bar"></span>${title}</div><div class="sub">无可用配置</div></div>`;
  return `<div class="card">
    <div class="card-title"><span class="bar"></span>${title}</div>
    <div class="reco-metrics">
      ${metric('集群吞吐', fmt(b.throughput_tokens_per_s), 'tokens/s')}
      ${metric('单卡吞吐', fmt(b.throughput_tokens_per_s_per_device), 'tokens/s')}
      ${metric('TTFT', b.ttft_ms == null ? '—' : fmt(b.ttft_ms, 1), 'ms')}
      ${metric('TPOT', b.tpot_ms == null ? '—' : fmt(b.tpot_ms, 1), 'ms')}
      ${metric('并发', fmtInt(b.concurrency), 'req')}
      ${metric('批大小', fmtInt(b.batch_size), '')}
    </div>
    <div class="reco-why"><span class="tag tag-blue">${escapeHtml(parallelStr(b))}</span></div>
  </div>`;
}

function candidatesTable(cands, title) {
  if (!cands || !cands.length) return `<div class="sub" style="margin-top:12px">${title || '候选'}：无</div>`;
  const rows = cands.map((c, i) => {
    const sla = c.meets_slo ? '<span class="tag tag-green">达标</span>' : '<span class="tag tag-red">未达标</span>';
    const top = i === 0 ? ' class="top1"' : '';
    return `<tr${top}>
      <td class="strong">#${i + 1}</td>
      <td>${escapeHtml(parallelStr(c))}</td>
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

function renderVariants(fw) {
  const vs = fw.disagg_variants || [];
  if (!vs.length) return '';
  const tabs = vs.map((v, i) => `<button class="tab ${i === 0 ? 'active' : ''}" data-vi="${i}" onclick="selectVariant(${i})">${escapeHtml(v.label || v.mode || ('#' + (i + 1)))}</button>`).join('');
  return `<div class="card">
    <div class="card-title"><span class="bar"></span>PD 分离均衡搜索（xPyD）</div>
    <div class="tabs">${tabs}</div>
    <div id="variant-body"></div>
  </div>`;
}

function variantBody(v) {
  const b = bestOf(v) || {};
  const pf = v.prefill || {};
  const dc = v.decode || {};
  const pfP = pf.parallel_str || (pf.parallel ? `TP=${pf.parallel.tp} | PP=${pf.parallel.pp} | DP=${pf.parallel.dp}` : '—');
  const dcP = dc.parallel_str || (dc.parallel ? `TP=${dc.parallel.tp} | PP=${dc.parallel.pp} | DP=${dc.parallel.dp}` : '—');
  return `
   <div class="reco-metrics">
     ${metric('组合吞吐', fmt(b.throughput_tokens_per_s), 'tokens/s')}
     ${metric('单卡吞吐', fmt(b.throughput_tokens_per_s_per_device), 'tokens/s')}
     ${metric('TPOT', b.tpot_ms == null ? '—' : fmt(b.tpot_ms, 1), 'ms')}
     ${metric('并发', fmtInt(b.concurrency), 'req')}
   </div>
   <div class="pd-grid">
     <div class="pd-side"><div class="pd-role">Prefill 侧</div><div class="pd-p">${escapeHtml(pfP)}</div><div class="sub">${fmt(pf.throughput_tokens_per_s)} tokens/s · ${fmtInt(pf.num_devices)} 卡</div></div>
     <div class="pd-side"><div class="pd-role">Decode 侧</div><div class="pd-p">${escapeHtml(dcP)}</div><div class="sub">${fmt(dc.throughput_tokens_per_s)} tokens/s · ${fmtInt(dc.num_devices)} 卡</div></div>
   </div>
   ${candidatesTable(v.candidates, '该组合候选')}`;
}

function selectVariant(i) {
  const v = (state.currentFw.disagg_variants || [])[i];
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', +t.dataset.vi === i));
  if (v) document.getElementById('variant-body').innerHTML = variantBody(v);
}

function renderCommands(data) {
  const cmds = (data.routing && data.routing.commands) || [];
  if (!cmds.length) return '';
  const items = cmds.map(c => `<li>${escapeHtml(c)}</li>`).join('');
  return `<details class="card"><summary class="card-title"><span class="bar"></span>执行命令（${cmds.length}）</summary><ul class="cmd-list">${items}</ul></details>`;
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

function renderDetail(data) {
  const fw = pickFramework(data);
  state.currentFw = fw;
  const req = data.request || {};
  document.getElementById('detail-title').textContent = `${req.model || '任务'} · ${req.num_devices || ''} 卡 · ${req.quantize || ''}`;
  document.getElementById('detail-sub').textContent = '点击各卡片查看并行策略（TP/PP/DP/EP）与候选配置';

  const parts = [renderMeta(data)];
  if (data.status === 'failed' || len(data.errors)) {
    parts.push(renderErrors(data));
  } else {
    parts.push(renderRecommendation(data));
    parts.push(cfgCard('聚合模式（agg）', fw.agg));
    parts.push(`<details class="card"><summary class="card-title"><span class="bar"></span>聚合候选（${len(fw.agg && fw.agg.candidates)}）</summary>${candidatesTable(fw.agg && fw.agg.candidates)}</details>`);
    parts.push(renderVariants(fw));
  }
  parts.push(renderCommands(data));
  document.getElementById('detail-body').innerHTML = parts.join('');
  if (len(fw.disagg_variants)) selectVariant(0);
}

/* ============================================================
 * 视图切换 / 面包屑 / 弹窗
 * ============================================================ */
function showView(v) {
  document.getElementById('view-list').classList.toggle('hide', v !== 'list');
  document.getElementById('view-detail').classList.toggle('hide', v !== 'detail');
}
function setCrumb(a, b) { document.getElementById('crumb').innerHTML = `<b>${escapeHtml(a)}</b> / ${escapeHtml(b)}`; }
function backToList() { loadList(); }
function openModal() { document.getElementById('mask').classList.add('show'); }
function closeModal() { document.getElementById('mask').classList.remove('show'); }

document.getElementById('f-model').addEventListener('change', applyFilters);
document.getElementById('f-status').addEventListener('change', applyFilters);
document.getElementById('f-kw').addEventListener('input', applyFilters);

document.getElementById('src-badge').textContent = useApi ? '数据源: API' : '数据源: results';

loadList();
