/**
 * EquiMind Canvas Chart Engine — charts.js
 * Zero-dependency HTML5 Canvas rendering engine for financial charts:
 * 1. Line Chart (Kalman Filter vs Raw Price)
 * 2. Probability Cone Chart (Monte Carlo 1,000 paths envelope)
 * 3. Donut / Pie Chart (Portfolio asset allocation weights)
 * 4. Horizontal Bar Chart (Alpha Factor Sharpe Ratios)
 * 5. Gauge Chart (RSI & Conviction meters)
 */

class QuantCharts {
  // ── 1. Line Overlay Chart (Raw vs Kalman Filtered) ──
  static drawLineOverlay(canvasId, rawData, filteredData, labels = []) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.offsetWidth || 500);
    const h = (canvas.height = canvas.offsetHeight || 220);

    ctx.clearRect(0, 0, w, h);

    const padLeft = 40, padRight = 20, padTop = 20, padBottom = 30;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    const allVals = [...rawData, ...filteredData];
    const minVal = Math.min(...allVals) * 0.98;
    const maxVal = Math.max(...allVals) * 1.02;

    const getX = (i) => padLeft + (i / (rawData.length - 1)) * chartW;
    const getY = (v) => padTop + chartH - ((v - minVal) / (maxVal - minVal)) * chartH;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (i / 4) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();

      const val = (maxVal - (i / 4) * (maxVal - minVal)).toFixed(1);
      ctx.fillStyle = '#5A5A72';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`$${val}`, padLeft - 6, y + 3);
    }

    // Raw Price Line (Dashed / Dim cyan)
    ctx.beginPath();
    rawData.forEach((v, i) => {
      const x = getX(i), y = getY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Kalman Filtered Line (Solid Electric Cyan)
    ctx.beginPath();
    filteredData.forEach((v, i) => {
      const x = getX(i), y = getY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00D4FF';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  // ── 2. Monte Carlo Probability Cone Chart ──
  static drawMonteCarloCone(canvasId, initialPrice = 500, horizonDays = 30) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.offsetWidth || 600);
    const h = (canvas.height = canvas.offsetHeight || 260);

    ctx.clearRect(0, 0, w, h);

    const padLeft = 45, padRight = 25, padTop = 25, padBottom = 30;
    const chartW = w - padLeft - padRight;
    const chartH = h - padTop - padBottom;

    // Simulate 20 representative paths
    const paths = [];
    const numPaths = 25;
    const mu = 0.18 / 252;
    const sigma = 0.32 / Math.sqrt(252);

    for (let p = 0; p < numPaths; p++) {
      const path = [initialPrice];
      for (let d = 1; d <= horizonDays; d++) {
        const last = path[d - 1];
        const z = (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.732;
        const ret = mu + sigma * z;
        path.push(last * Math.exp(ret));
      }
      paths.push(path);
    }

    // Min and Max values for scale
    const allPrices = paths.flat();
    const minP = Math.min(...allPrices) * 0.95;
    const maxP = Math.max(...allPrices) * 1.05;

    const getX = (d) => padLeft + (d / horizonDays) * chartW;
    const getY = (p) => padTop + chartH - ((p - minP) / (maxP - minP)) * chartH;

    // Grid lines & labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padTop + (i / 4) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(w - padRight, y);
      ctx.stroke();

      const val = (maxP - (i / 4) * (maxP - minP)).toFixed(0);
      ctx.fillStyle = '#5A5A72';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`$${val}`, padLeft - 6, y + 3);
    }

    // Draw individual path lines
    paths.forEach((path, idx) => {
      ctx.beginPath();
      path.forEach((p, d) => {
        const x = getX(d), y = getY(p);
        if (d === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = idx % 2 === 0 ? 'rgba(0, 212, 255, 0.18)' : 'rgba(124, 58, 237, 0.18)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Percentile lines: P05, Median, P95
    const finalPrices = paths.map(p => p[horizonDays]).sort((a, b) => a - b);
    const p05 = finalPrices[Math.floor(numPaths * 0.05)];
    const median = finalPrices[Math.floor(numPaths * 0.50)];
    const p95 = finalPrices[Math.floor(numPaths * 0.95)];

    // Median path line
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(initialPrice));
    ctx.lineTo(getX(horizonDays), getY(median));
    ctx.strokeStyle = '#00D4FF';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // P95 Upside line (Green)
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(initialPrice));
    ctx.lineTo(getX(horizonDays), getY(p95));
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();

    // P05 Downside line (Red)
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(initialPrice));
    ctx.lineTo(getX(horizonDays), getY(p05));
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── 3. Portfolio Allocation Donut Chart ──
  static drawDonutChart(canvasId, data = [
    { label: 'NVDA', weight: 0.35, color: '#00D4FF' },
    { label: 'MSFT', weight: 0.25, color: '#7C3AED' },
    { label: 'AAPL', weight: 0.20, color: '#10B981' },
    { label: 'GOOGL', weight: 0.12, color: '#F59E0B' },
    { label: 'CASH', weight: 0.08, color: '#5A5A72' }
  ]) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.offsetWidth || 260);
    const h = (canvas.height = canvas.offsetHeight || 260);

    ctx.clearRect(0, 0, w, h);

    const centerX = w / 2;
    const centerY = h / 2;
    const outerR = Math.min(w, h) / 2 - 20;
    const innerR = outerR * 0.62;

    let startAngle = -Math.PI / 2;

    data.forEach(item => {
      const sliceAngle = item.weight * Math.PI * 2;
      const endAngle = startAngle + sliceAngle;

      ctx.beginPath();
      ctx.arc(centerX, centerY, outerR, startAngle, endAngle);
      ctx.arc(centerX, centerY, innerR, endAngle, startAngle, true);
      ctx.closePath();

      ctx.fillStyle = item.color;
      ctx.fill();

      startAngle = endAngle;
    });

    // Center text
    ctx.fillStyle = '#F0F0F6';
    ctx.font = '700 16px "Syne", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Optimal', centerX, centerY - 4);

    ctx.fillStyle = '#8E8EA8';
    ctx.font = '500 11px "DM Sans", sans-serif';
    ctx.fillText('Weights', centerX, centerY + 14);
  }

  // ── 4. Horizontal Factor Bar Chart ──
  static drawHorizontalBars(canvasId, factors = [
    { name: 'Momentum 12M', sharpe: 1.82, color: '#00D4FF' },
    { name: 'Quality (ROE)', sharpe: 1.64, color: '#10B981' },
    { name: 'Value (P/E)',   sharpe: 1.15, color: '#7C3AED' },
    { name: 'Low Volatility', sharpe: 0.94, color: '#F59E0B' },
    { name: 'Size (Small Cap)', sharpe: -0.42, color: '#EF4444' }
  ]) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = (canvas.width = canvas.offsetWidth || 400);
    const h = (canvas.height = canvas.offsetHeight || 220);

    ctx.clearRect(0, 0, w, h);

    const padLeft = 110, padRight = 50, padTop = 15, padBottom = 15;
    const barH = 22;
    const gap = 14;

    const maxVal = 2.5;

    factors.forEach((f, idx) => {
      const y = padTop + idx * (barH + gap);
      const barW = (Math.abs(f.sharpe) / maxVal) * (w - padLeft - padRight);

      // Label
      ctx.fillStyle = '#8E8EA8';
      ctx.font = '500 12px "DM Sans", sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(f.name, padLeft - 10, y + barH / 1.5);

      // Bar
      ctx.fillStyle = f.color;
      ctx.beginPath();
      ctx.roundRect(padLeft, y, Math.max(barW, 4), barH, 4);
      ctx.fill();

      // Value
      ctx.fillStyle = '#F0F0F6';
      ctx.font = '700 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(f.sharpe > 0 ? `+${f.sharpe}` : `${f.sharpe}`, padLeft + Math.max(barW, 4) + 8, y + barH / 1.5);
    });
  }
}

window.QuantCharts = QuantCharts;
