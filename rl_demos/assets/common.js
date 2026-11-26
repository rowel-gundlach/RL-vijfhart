// Small drawing helpers shared by every demo.
const INK = "#1f2d3d", MUTED = "#5d6c7c", GOOD = "#1f9d55", BAD = "#d64533", ACCENT = "#2f6fde", WARN = "#e39a12", LINE = "#d9e1ea";

function h(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "onclick" || k === "oninput" || k === "onkeydown") node[k] = v;
    else if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...kids);
  return node;
}

function speedSlider(start = 5, text = "Speed") {
  const input = h("input", { type: "range", min: 1, max: 10, value: start });
  const label = h("label", { class: "speed" }, text, input);
  return { label, delay: () => Math.round(600 / Math.pow(1.75, Number(input.value) - 1)) };
}

function valueColor(v, max) {
  if (!v) return "#eef2f6";
  const a = Math.min(Math.abs(v) / (max || 1), 1) * 0.8 + 0.12;
  return v > 0 ? `rgba(31,157,85,${a})` : `rgba(214,69,51,${a})`;
}

function popup(parent, text, good, x = 50, y = 45) {
  const p = h("div", { class: "pop", text, style: `left:${x}%;top:${y}%;color:${good ? GOOD : BAD}` });
  parent.append(p);
  setTimeout(() => p.remove(), 1000);
}

function sizeCanvas(canvas, height) {
  const width = canvas.parentElement.clientWidth - 24 || 600, ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

// series: [{name, color, data: [[x, y], ...]}]; the name is written at the end of each line.
function lineChart(canvas, series, { title = "", height = 280, xLabel = "", step = false } = {}) {
  const { ctx, width } = sizeCanvas(canvas, height);
  const left = 74, right = 150, top = 38, bottom = 44;
  ctx.clearRect(0, 0, width, height);
  ctx.font = "600 18px Segoe UI, sans-serif"; ctx.fillStyle = INK; ctx.fillText(title, 4, 22);
  const pts = series.flatMap(s => s.data);
  if (!pts.length) return;
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = Math.min(0, ...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) y1 = y0 + 1;
  const X = x => left + (x - x0) / (x1 - x0) * (width - left - right);
  const Y = y => height - bottom - (y - y0) / (y1 - y0) * (height - top - bottom);
  ctx.font = "15px Segoe UI, sans-serif"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    ctx.strokeStyle = LINE; ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(width - right, Y(v)); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.fillText(Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(Math.abs(y1 - y0) < 8 ? 1 : 0), left - 8, Y(v) + 5);
  }
  ctx.textAlign = "center";
  ctx.fillText(Math.round(x0), X(x0), height - 22); ctx.fillText(Math.round(x1), X(x1), height - 22);
  ctx.fillText(xLabel, (left + width - right) / 2, height - 6);
  ctx.textAlign = "left";
  for (const s of series) {
    if (!s.data.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.setLineDash(s.dash || []); ctx.beginPath();
    s.data.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(X(x), Y(y));
      else { if (step) ctx.lineTo(X(x), Y(s.data[i - 1][1])); ctx.lineTo(X(x), Y(y)); }
    });
    ctx.stroke(); ctx.lineWidth = 1; ctx.setLineDash([]);
    const [lx, ly] = s.data[s.data.length - 1];
    ctx.fillStyle = s.color; ctx.font = "600 15px Segoe UI, sans-serif"; ctx.fillText(s.name, X(lx) + 8, Y(ly) + 5);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// €1,250 or −€60 (a real minus sign before the euro sign)
const euro = x => (x < 0 ? "−€" : "€") + Math.abs(Math.round(x)).toLocaleString("en");
