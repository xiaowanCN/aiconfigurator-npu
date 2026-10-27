'use strict';

/* ============================================================
 * 数据源抽象
 *   - API 模式：调用 infer-strategy-npu-server 的 /api/v1（支持新建任务）
 *   - 静态模式：直接读 results/infer-strategy-npu/index.json（只读）
 * 模式自动探测（探测 /api/v1/health）；也可用 ?source=api / ?source=static 强制。
 * 两者返回的“单任务结构”都是 run.json，因此渲染逻辑无需改动。
 * ============================================================ */
const API_BASE = '/api/v1';
const RESULTS_INDEX = '../results/infer-strategy-npu/index.json';
const RESULTS_RUN = (id) => `../results/infer-strategy-npu/${id}/run.json`;

const _forcedSource = new URLSearchParams(location.search).get('source'); // api | static | null
let _apiMode = _forcedSource ? _forcedSource === 'api' : null; // null = 未探测

async function isApiMode() {
  if (_apiMode !== null) return _apiMode;
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
    const j = r.ok ? await r.json() : null;
    _apiMode = !!(j && j.code === 200);
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
    const r = await fetch(RESULTS_INDEX);
    if (!r.ok) throw new Error(
      '无法读取 results/index.json。两种解法：① 启动服务后从服务端打开本页 —— ' +
      '`infer-strategy-npu-server --port 8000`，浏览器访问 http://<host>:8000/ ；' +
      '② 或在仓库根目录执行 `python -m http.server` 后再访问（浏览器 file:// 直接打开会因 CORS 拒绝读取本地文件）。'
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
      <div class="reco-metrics">
        ${metric('状态', statusTag(status), '')}
        ${metric('当前阶段', escapeHtml(stage), '')}
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
    <div class="card-title"><span class="bar"></span>PD 分离（PD-ratio 每实例卡数组合）</div>
    <div class="tabs">${tabs}</div>
    <div id="variant-body"></div>
  </div>`;
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

function pdSideCells(c) {
  const p = c.prefill || {};
  const d = c.decode || {};
  const sideCell = s => `${fmtInt(s.instances)} × ${fmtInt(s.devices_per_instance)} 卡`;
  return `
    <td class="strong">${escapeHtml(c.label || '—')}</td>
    <td class="num">${sideCell(p)}</td>
    <td class="num">${sideCell(d)}</td>
    <td class="sub">${escapeHtml(p.parallel_str || '—')}</td>
    <td class="sub">${escapeHtml(d.parallel_str || '—')}</td>
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
    return `<tr${top}>${pdSideCells(c)}</tr>`;
  }).join('');
  return `<table style="margin-top:12px"><thead><tr>
    <th>排名</th><th>P×D</th><th>P 实例×卡</th><th>D 实例×卡</th><th>P 并行</th><th>D 并行</th>
    <th>decode 吞吐(单实例)</th><th>集群吞吐</th><th>单卡吞吐</th><th>req/s</th><th>TTFT</th><th>TPOT</th><th>用卡</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderPdCandidates(fw) {
  const cands = fw.pd_ratio_candidates || [];
  if (!cands.length) return '';
  const labels = [...new Set(cands.map(c => c.label).filter(Boolean))];
  return `<details class="card" open><summary class="card-title"><span class="bar"></span>PD 分离候选汇总（${cands.length} 条 · 组合 ${labels.join(' / ')} · 按单 D 实例 decode 吞吐降序）</summary>
    ${pdCandidatesTable(cands)}</details>`;
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
     ${metric('用卡', b.num_devices == null ? '—' : fmtInt(b.num_devices), '卡')}
   </div>
   <div class="pd-grid">
     <div class="pd-side"><div class="pd-role">Prefill 侧</div><div class="pd-p">${escapeHtml(pfP)}</div>
       <div class="sub">${fmt(pf.throughput_tokens_per_s)} tokens/s · ${pdSideSummary(pf)}</div></div>
     <div class="pd-side"><div class="pd-role">Decode 侧</div><div class="pd-p">${escapeHtml(dcP)}</div>
       <div class="sub">${fmt(dc.throughput_tokens_per_s)} tokens/s · ${pdSideSummary(dc)}</div></div>
   </div>
   ${len(v.pd_ratio_candidates)
     ? pdCandidatesTable(v.pd_ratio_candidates)
     : candidatesTable(v.candidates, '该组合候选')}`;
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
    parts.push(renderPdCandidates(fw));
  }
  parts.push(renderCommands(data));
  document.getElementById('detail-body').innerHTML = parts.join('');
  if (len(fw.disagg_variants)) selectVariant(0);
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

(async () => {
  const api = await isApiMode();
  document.getElementById('src-badge').textContent = api ? '数据源: API' : '数据源: results（只读）';
  await loadList();
})();
