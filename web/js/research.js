/**
 * EquiMind Research Execution Engine — research.js
 * Manages DAG visualization, streaming timeline, WebSocket + mock fallback
 */

/* ── Parse query params ── */
const params   = new URLSearchParams(window.location.search);
const TICKER   = params.get('ticker')  || 'NVDA';
const QUERY    = params.get('query')   || `Analyze ${TICKER} for long-term investment`;
const PROVIDER = params.get('provider')|| 'mock';

/* ── Execution State ── */
const Nodes = {
  planner:     { id: 'planner',     label: 'Planner',      icon: '🎯', state: 'waiting', progress: 0, output: '' },
  market:      { id: 'market',      label: 'Market Data',  icon: '📈', state: 'waiting', progress: 0, output: '' },
  fundamental: { id: 'fundamental', label: 'Fundamental',  icon: '📊', state: 'waiting', progress: 0, output: '' },
  macro:       { id: 'macro',       label: 'Macro',        icon: '🌍', state: 'waiting', progress: 0, output: '' },
  web:         { id: 'web',         label: 'Web Intel',    icon: '🕸️', state: 'waiting', progress: 0, output: '' },
  featureStore:{ id: 'featureStore',label: 'FeatureStore', icon: '⚗️', state: 'waiting', progress: 0, output: '' },
  quant:       { id: 'quant',       label: 'Quant Engines',icon: '🧮', state: 'waiting', progress: 0, output: '' },
  memory:      { id: 'memory',      label: 'Memory',       icon: '🧠', state: 'waiting', progress: 0, output: '' },
  committee:   { id: 'committee',   label: 'Committee',    icon: '⚔️', state: 'waiting', progress: 0, output: '' },
};

let timelineCount = 0;
let elapsedTimer  = null;
let elapsedSec    = 0;
let researchDone  = false;
let activeTab     = 'dag';

/* ── Timeline Event Script (mock) ── */
const MOCK_EVENTS = [
  { delay: 600,   node: 'planner', type: 'planner', state: 'running', progress: 10,
    agent: 'Reasoning Planner', color: '#00D4FF',
    msg: `<strong>Query received:</strong> "${QUERY}"`,
    metrics: [] },

  { delay: 1800,  node: 'planner', type: 'planner', state: 'running', progress: 60,
    agent: 'Reasoning Planner', color: '#00D4FF',
    msg: `Sector detected: <strong>Semiconductor / AI Infrastructure</strong>. Activating: Market Data, Fundamental, Macro, Web Intelligence, GitHub commit tracking.`,
    metrics: [
      { k: 'Horizon', v: 'Long-Term', cls: 'neutral' },
      { k: 'Sector', v: 'SEMIS', cls: 'neutral' },
      { k: 'Teams', v: '4 active', cls: 'neutral' },
    ]},

  { delay: 2800,  node: 'planner', type: 'planner', state: 'done', progress: 100,
    agent: 'Reasoning Planner', color: '#00D4FF',
    msg: `Research DAG finalized. Dispatching parallel research pipeline.`,
    metrics: [] },

  { delay: 3200,  node: 'memory', type: 'quant', state: 'running', progress: 50,
    agent: 'Memory Engine', color: '#10B981',
    msg: `Checking hierarchical memory for <strong>${TICKER}</strong>. Found 3 previous reports. Delta engine activated — only fetching signals newer than last analysis.`,
    metrics: [] },

  { delay: 3800,  node: 'market', type: 'market', state: 'running', progress: 20,
    agent: 'Market Data Team', color: '#7C3AED',
    msg: `Retrieving <strong>15 years</strong> of historical OHLCV price data for ${TICKER}. Fetching orderbook liquidity, volume profile, and benchmark index comparison.`,
    metrics: [] },

  { delay: 4200,  node: 'fundamental', type: 'fundamental', state: 'running', progress: 30,
    agent: 'Fundamental Team', color: '#F59E0B',
    msg: `Parsing 10-K annual report and 10-Q quarterly filings. Analyzing income statement, balance sheet, and cash flow statement for ${TICKER}.`,
    metrics: [] },

  { delay: 4600,  node: 'macro', type: 'macro', state: 'running', progress: 40,
    agent: 'Macro Research Team', color: '#10B981',
    msg: `Fetching macroeconomic signals: CPI 3.2%, Fed Funds Rate 5.25%, VIX 18.4, Brent Crude $82.40/bbl. Evaluating semiconductor supply chain pressure.`,
    metrics: [
      { k: 'CPI', v: '3.2%', cls: 'negative' },
      { k: 'Fed Rate', v: '5.25%', cls: 'negative' },
      { k: 'VIX', v: '18.4', cls: 'neutral' },
    ]},

  { delay: 5000,  node: 'web', type: 'web', state: 'running', progress: 35,
    agent: 'Web Intelligence Team', color: '#8B5CF6',
    msg: `Crawling evidence sources: SEC EDGAR filings, Bloomberg news (47 articles), Reddit r/stocks (23 threads), GitHub ${TICKER.toLowerCase()} repository (892 commits this month).`,
    metrics: [] },

  { delay: 6200,  node: 'market', type: 'quant', state: 'done', progress: 100,
    agent: 'Market Data Team', color: '#7C3AED',
    msg: `Price data retrieved. Running Technical Analysis Engine.`,
    metrics: [
      { k: 'RSI(14)', v: '68.4', cls: 'neutral' },
      { k: 'MACD', v: '+2.34', cls: 'positive' },
      { k: 'BB Width', v: '8.2%', cls: 'neutral' },
      { k: 'ATR(14)', v: '$12.80', cls: 'neutral' },
      { k: '200d SMA', v: '$412.50', cls: 'positive' },
    ]},

  { delay: 7000,  node: 'fundamental', type: 'quant', state: 'done', progress: 100,
    agent: 'Fundamental Team', color: '#F59E0B',
    msg: `Fundamental analysis complete. Piotroski F-Score computed across 9 criteria.`,
    metrics: [
      { k: 'PE Ratio', v: '42.3x', cls: 'neutral' },
      { k: 'P/B', v: '28.1x', cls: 'neutral' },
      { k: 'ROE', v: '68.4%', cls: 'positive' },
      { k: 'Piotroski', v: '8/9', cls: 'positive' },
      { k: 'Altman Z', v: '4.82 ✓ Safe', cls: 'positive' },
    ]},

  { delay: 7600,  node: 'macro', type: 'macro', state: 'done', progress: 100,
    agent: 'Macro Research Team', color: '#10B981',
    msg: `Macro analysis complete. AI infrastructure spending cycle remains intact. Taiwan Strait risk elevated: 34% supply chain exposure.`,
    metrics: [
      { k: 'AI Capex Growth', v: '+38% YoY', cls: 'positive' },
      { k: 'Supply Chain Risk', v: 'MEDIUM', cls: 'neutral' },
    ]},

  { delay: 8400,  node: 'web', type: 'web', state: 'done', progress: 100,
    agent: 'Web Intelligence Team', color: '#8B5CF6',
    msg: `Evidence collection complete. 63 verified evidence nodes. Context compressor applied: MD5 deduplication removed 12 duplicates, Jaccard clustering merged 8 redundant articles.`,
    metrics: [
      { k: 'Evidence Nodes', v: '63', cls: 'positive' },
      { k: 'Deduped', v: '-12', cls: 'neutral' },
      { k: 'Bullish', v: '68%', cls: 'positive' },
      { k: 'Credibility Avg', v: '0.81', cls: 'positive' },
    ]},

  { delay: 9000,  node: 'memory', type: 'quant', state: 'done', progress: 100,
    agent: 'Memory Engine', color: '#10B981',
    msg: `Delta-research complete. 41 new signals since last analysis (14 days ago). Hierarchical memory updated across Tier 1 and Tier 2.`,
    metrics: [] },

  { delay: 9400,  node: 'featureStore', type: 'quant', state: 'running', progress: 50,
    agent: 'Feature Engineering', color: '#00D4FF',
    msg: `FeatureStore extracting numerical feature vectors from 63 evidence nodes and 15-year price series. Z-score normalization applied.`,
    metrics: [] },

  { delay: 10200, node: 'quant', type: 'quant', state: 'running', progress: 20,
    agent: 'Quantitative Engine', color: '#10B981',
    msg: `Launching Advanced Quantitative Suite: Time Series Engine, Alpha Lab, Causal Reasoning Engine, Monte Carlo Simulator, Portfolio Optimizer.`,
    metrics: [] },

  { delay: 11000, node: 'featureStore', type: 'quant', state: 'done', progress: 100,
    agent: 'Feature Engineering', color: '#00D4FF',
    msg: `Feature vectors ready. 24-dimensional normalized feature matrix constructed.`,
    metrics: [
      { k: 'Features', v: '24 dims', cls: 'neutral' },
      { k: 'Sentiment', v: '+0.62', cls: 'positive' },
      { k: 'Credibility', v: '0.81', cls: 'positive' },
    ]},

  { delay: 11800, node: 'quant', type: 'quant', state: 'running', progress: 55,
    agent: 'Time Series Engine', color: '#10B981',
    msg: `1D Kalman Filter noise reduction applied. HMM Market Regime Classifier: <strong>BULL_TREND</strong> state detected with 84% confidence. GARCH(1,1) daily volatility: 1.68%.`,
    metrics: [
      { k: 'Regime', v: 'BULL_TREND', cls: 'positive' },
      { k: 'Regime Conf', v: '84%', cls: 'positive' },
      { k: 'GARCH Vol', v: '1.68%/day', cls: 'neutral' },
    ]},

  { delay: 12600, node: 'quant', type: 'quant', state: 'running', progress: 70,
    agent: 'Alpha Lab + Causal Engine', color: '#10B981',
    msg: `Alpha Research Lab: Momentum IC = 0.34, Rank IC = 0.29 (statistically significant). Causal Engine: raw correlation to SOX index 0.78, but direct causal effect after confound adjustment = 0.82 — <strong>genuine direct relationship confirmed</strong>.`,
    metrics: [
      { k: 'IC', v: '0.34', cls: 'positive' },
      { k: 'Rank IC', v: '0.29', cls: 'positive' },
      { k: 'Spurious?', v: 'NO ✓', cls: 'positive' },
    ]},

  { delay: 13500, node: 'quant', type: 'quant', state: 'running', progress: 88,
    agent: 'Monte Carlo Simulator', color: '#10B981',
    msg: `1,000 stochastic price paths simulated (GBM + Jump Diffusion, 30-day horizon). Probability of profit: 78%. P05 downside: $412.30 · Median: $519.40 · P95 upside: $648.20.`,
    metrics: [
      { k: 'P05 (Downside)', v: '$412.30', cls: 'negative' },
      { k: 'Median', v: '$519.40', cls: 'neutral' },
      { k: 'P95 (Upside)', v: '$648.20', cls: 'positive' },
      { k: 'P(profit)', v: '78%', cls: 'positive' },
    ]},

  { delay: 14400, node: 'quant', type: 'quant', state: 'done', progress: 100,
    agent: 'Portfolio Optimizer', color: '#10B981',
    msg: `Markowitz Mean-Variance + Black-Litterman optimization complete. Kelly fraction: 12.4% of portfolio. Diversification Score: 0.74.`,
    metrics: [
      { k: 'Sharpe Ratio', v: '2.14', cls: 'positive' },
      { k: 'Kelly Fraction', v: '12.4%', cls: 'positive' },
      { k: 'Max Drawdown', v: '-18.2%', cls: 'negative' },
      { k: 'Sortino', v: '3.28', cls: 'positive' },
    ]},

  { delay: 15200, node: 'committee', type: 'committee', state: 'running', progress: 25,
    agent: 'Bull Agent', color: '#F59E0B',
    msg: `<strong>BULL THESIS:</strong> NVDA Blackwell GPU architecture creates 18-24 month competitive moat. Data center AI revenue projected +$40B by FY2026. GitHub developer adoption velocity at all-time high (+892 commits/month in CUDA ecosystem). Piotroski F-Score 8/9 signals strong financial health.`,
    metrics: [] },

  { delay: 16400, node: 'committee', type: 'committee', state: 'running', progress: 55,
    agent: 'Bear Agent', color: '#F59E0B',
    msg: `<strong>BEAR THESIS:</strong> PE ratio 42.3x prices in perfection. Taiwan supply chain risk 34% exposure. AMD MI300X and Google TPUs emerging as credible alternatives. GOOG and MSFT internal chip development may reduce NVDA dependency by 2026.`,
    metrics: [] },

  { delay: 17500, node: 'committee', type: 'committee', state: 'running', progress: 80,
    agent: 'Judge Agent', color: '#F59E0B',
    msg: `Evidence weight analysis: Bull 7 supported claims, Bear 4 supported claims. 2 bear claims flagged as unverified speculation. Resolving contradiction on Taiwan supply chain — TSMC N3 node risk is real but limited to 2 quarters of disruption maximum.`,
    metrics: [
      { k: 'Bull Evidence', v: '7/7 ✓', cls: 'positive' },
      { k: 'Bear Evidence', v: '4/6 ✓', cls: 'neutral' },
      { k: 'Unverified Claims', v: '2 stripped', cls: 'negative' },
    ]},

  { delay: 18800, node: 'committee', type: 'final', state: 'done', progress: 100,
    agent: 'Judge Agent (Final)', color: '#10B981',
    msg: `<strong>RESEARCH COMPLETE.</strong> Final recommendation generated with full citation trail. 63 evidence nodes verified. All claims traceable.`,
    metrics: [] },
];

/* ── DOM Helpers ── */
function $(id) { return document.getElementById(id); }
function updateNodeState(nodeId, state, progress, output) {
  const card = $(`agent-${nodeId}`);
  if (!card) return;
  card.className = `agent-card ${state}`;
  const statusText = card.querySelector('.agent-status-text');
  const statusDot  = card.querySelector('.status-dot');
  const progBar    = card.querySelector('.agent-progress-bar');
  const outEl      = card.querySelector('.agent-output');

  const stateLabels = { waiting: 'Waiting', running: 'Running...', done: '✓ Complete', skipped: 'Skipped', failed: 'Error' };
  if (statusText) statusText.textContent = stateLabels[state] || state;
  if (statusDot)  statusDot.className = `status-dot ${state === 'done' ? 'done' : state === 'running' ? 'running' : 'waiting'}`;
  if (progBar)    progBar.style.width = progress + '%';
  if (outEl && output) outEl.textContent = output;

  // Update DAG node
  const dagNode = $(`dag-${nodeId}`);
  if (dagNode) {
    dagNode.className = `dag-node-box state-${state}`;
    const dur = dagNode.querySelector('.dag-node-duration');
    if (dur && state === 'done') dur.textContent = `${(Math.random() * 2 + 0.5).toFixed(1)}s`;
  }
}

function addTimelineItem(event, timestamp) {
  const container = $('timeline-container');
  if (!container) return;

  // Remove typing indicator
  const typing = container.querySelector('.typing-indicator');
  if (typing) typing.remove();

  const item = document.createElement('div');
  item.className = `timeline-item type-${event.type}`;
  item.style.animationDelay = '0ms';

  const metricsHtml = event.metrics?.length
    ? `<div class="timeline-metric-row">${event.metrics.map(m =>
        `<div class="timeline-metric">
          <span class="timeline-metric-key">${m.k}</span>
          <span class="timeline-metric-val ${m.cls}">${m.v}</span>
        </div>`
      ).join('')}</div>`
    : '';

  item.innerHTML = `
    <div style="flex:1;min-width:0">
      <div class="timeline-item-header">
        <span class="timeline-agent-badge" style="background:${event.color}20;color:${event.color};border:1px solid ${event.color}30">
          ${event.agent}
        </span>
        <span class="timeline-time">${timestamp}</span>
      </div>
      <div class="timeline-message">${event.msg}</div>
      ${metricsHtml}
    </div>
  `;

  container.appendChild(item);
  timelineCount++;
  $('timeline-count') && ($('timeline-count').textContent = timelineCount);

  // Add typing indicator back if not done
  if (!researchDone) {
    const typingEl = document.createElement('div');
    typingEl.className = 'typing-indicator';
    typingEl.innerHTML = `
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
      <span>Processing...</span>
    `;
    container.appendChild(typingEl);
  }

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

function getTimestamp() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

function showFinalCard() {
  const timeline = $('timeline-container');
  if (!timeline) return;

  // Remove typing indicator
  const typing = timeline.querySelector('.typing-indicator');
  if (typing) typing.remove();

  const card = document.createElement('div');
  card.className = 'final-rec-card';
  card.innerHTML = `
    <div class="t-label" style="margin-bottom:var(--space-3)">📄 Research Complete — ${TICKER}</div>
    <div class="final-rec-rating">⬆ STRONG BUY</div>
    <div class="final-rec-conviction">Conviction Score: 91% · 63 Evidence Nodes · All Claims Verified</div>
    <div class="final-rec-grid">
      <div class="final-rec-item">
        <span class="final-rec-label">Entry Range</span>
        <span class="final-rec-value">$480–$510</span>
      </div>
      <div class="final-rec-item">
        <span class="final-rec-label">Target (12M)</span>
        <span class="final-rec-value" style="color:var(--green)">$620–$650</span>
      </div>
      <div class="final-rec-item">
        <span class="final-rec-label">Stop Loss</span>
        <span class="final-rec-value" style="color:var(--red)">$430</span>
      </div>
      <div class="final-rec-item">
        <span class="final-rec-label">Risk/Reward</span>
        <span class="final-rec-value">1:2.8</span>
      </div>
      <div class="final-rec-item">
        <span class="final-rec-label">Kelly Fraction</span>
        <span class="final-rec-value">12.4%</span>
      </div>
      <div class="final-rec-item">
        <span class="final-rec-label">Time Horizon</span>
        <span class="final-rec-value">12–18M</span>
      </div>
    </div>
    <div style="margin-top:var(--space-5);display:flex;gap:var(--space-3)">
      <a href="report.html?ticker=${TICKER}" class="btn btn-primary btn-sm">View Full Report</a>
      <a href="evidence.html?ticker=${TICKER}" class="btn btn-ghost btn-sm">Browse Evidence</a>
      <a href="committee.html?ticker=${TICKER}" class="btn btn-ghost btn-sm">See Debate</a>
    </div>
  `;
  timeline.appendChild(card);
  timeline.scrollTop = timeline.scrollHeight;
}

/* ── WebSocket Client with Mock Fallback ── */
class ResearchStream {
  constructor() {
    this.ws = null;
    this.mockIndex = 0;
    this.mockTimer = null;
    this.usingMock = false;
  }

  start() {
    if (PROVIDER === 'mock') {
      this.startMock();
      return;
    }
    // Try WebSocket connection
    try {
      const wsUrl = `ws://localhost:8000/ws/research?ticker=${TICKER}&provider=${PROVIDER}`;
      this.ws = new WebSocket(wsUrl);
      this.ws.onopen = () => {
        this.ws.send(JSON.stringify({ ticker: TICKER, query: QUERY, provider: PROVIDER }));
      };
      this.ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          this.handleServerEvent(data);
        } catch {}
      };
      this.ws.onerror = () => { this.ws.close(); this.startMock(); };
      this.ws.onclose = () => { if (!this.usingMock) this.startMock(); };
      // Timeout to fallback
      setTimeout(() => {
        if (this.ws.readyState !== WebSocket.OPEN) { this.startMock(); }
      }, 3000);
    } catch {
      this.startMock();
    }
  }

  startMock() {
    this.usingMock = true;
    if (window.toast) toast.research(`Running research on ${TICKER} (simulation mode)`);
    this.scheduleNext();
  }

  scheduleNext() {
    if (this.mockIndex >= MOCK_EVENTS.length) {
      // All events played — finish
      setTimeout(() => this.finish(), 800);
      return;
    }
    const evt = MOCK_EVENTS[this.mockIndex];
    const delay = this.mockIndex === 0 ? evt.delay : (evt.delay - MOCK_EVENTS[this.mockIndex - 1].delay);
    this.mockTimer = setTimeout(() => {
      this.handleEvent(evt);
      this.mockIndex++;
      this.scheduleNext();
    }, delay);
  }

  handleEvent(evt) {
    updateNodeState(evt.node, evt.state, evt.progress, evt.output);
    addTimelineItem(evt, getTimestamp());
    updateElapsedDisplay();
  }

  handleServerEvent(data) {
    // Map server events to the same format
    if (data.node && data.state) updateNodeState(data.node, data.state, data.progress || 50, '');
    if (data.message) addTimelineItem({ ...data, agent: data.agent || 'System', msg: data.message, metrics: data.metrics || [], type: data.type || 'quant', color: '#00D4FF' }, getTimestamp());
  }

  finish() {
    researchDone = true;
    clearInterval(elapsedTimer);
    showFinalCard();
    updateNodeState('committee', 'done', 100, '');
    $('research-status-badge') && ($('research-status-badge').innerHTML = '<span class="badge badge-green" style="gap:5px"><span class="status-dot done"></span>Complete</span>');
    $('dag-tab-label') && ($('dag-tab-label').textContent = '✓ Research Complete');
  }

  stop() {
    if (this.ws) this.ws.close();
    clearTimeout(this.mockTimer);
  }
}

/* ── Tab Switching ── */
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.workspace-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));
}

/* ── Elapsed Timer ── */
function updateElapsedDisplay() {
  const el = $('elapsed-time');
  if (el) el.textContent = `${elapsedSec}s`;
}

/* ── Init ── */
document.addEventListener('DOMContentLoaded', () => {
  // Fill query info
  const qEl = $('query-display');
  const tEl = $('ticker-display');
  if (qEl) qEl.textContent = QUERY;
  if (tEl) tEl.textContent = TICKER;

  // Elapsed timer
  elapsedTimer = setInterval(() => {
    elapsedSec++;
    updateElapsedDisplay();
  }, 1000);

  // Tab listeners
  document.querySelectorAll('.workspace-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    tab.addEventListener('keydown', e => { if (e.key === 'Enter') switchTab(tab.dataset.tab); });
  });

  // DAG node click → scroll to timeline entry
  document.querySelectorAll('.dag-node-box').forEach(box => {
    box.addEventListener('click', () => switchTab('timeline'));
  });

  // Start research stream
  const stream = new ResearchStream();
  stream.start();
  window._researchStream = stream;
});
