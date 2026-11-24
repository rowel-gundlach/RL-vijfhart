const HEIGHT = 420, TRACK = 372, CART_W = 58, CART_H = 26, POLE = 110;
const INPUTS = ["position", "speed", "angle", "pole speed"];
const INPUT_RANGE = [2.4, 2, 0.21, 2];       // what counts as a large input, for the colours only

const spread = (count, top, bottom) => Array.from({ length: count }, (_, i) => count === 1 ? (top + bottom) / 2 : top + i * (bottom - top) / (count - 1));

function showInput(value, i) {
  return i === 2 ? `${(value * 180 / Math.PI).toFixed(1)}°` : value.toFixed(2);
}

function stage(title, game) {
  const canvas = h("canvas");
  const status = h("div", { class: "status" }), detail = h("div", { class: "small muted" });
  const card = h("div", { class: "card col", style: "flex-basis:400px" }, h("strong", { text: title }), canvas, status, detail);
  const last = game.states.length - 1, end = last + game.fall.length;
  const signals = step => [game.states[step], game.layers[0][step], game.layers[1][step]];

  // One colour scale for the whole game, so you can see signals grow and fade.
  const peaks = game.layers.map(layer => layer[0].map((_, i) => Math.max(1e-6, ...layer.map(row => row[i]))));
  const strongest = game.weights.map(() => 1e-6);
  for (let t = 0; t <= last; t++) {
    const from = signals(t);
    game.weights.forEach((matrix, l) => matrix.forEach((row, i) => row.forEach(w => {
      strongest[l] = Math.max(strongest[l], Math.abs(from[l][i] * w));
    })));
  }

  function network(ctx, width, step, over) {
    const xs = [100, 100 + (width - 200) / 3, 100 + 2 * (width - 200) / 3, width - 100];
    const ys = [spread(4, 70, 260), spread(game.layers[0][0].length, 48, 282), spread(game.layers[1][0].length, 48, 282), spread(2, 120, 210)];
    const from = signals(step), chosen = game.actions[step];
    const fade = over ? 0.3 : 1;

    game.weights.forEach((matrix, l) => matrix.forEach((row, i) => row.forEach((w, j) => {
      const signal = from[l][i] * w, strength = Math.min(1, Math.abs(signal) / strongest[l]);
      ctx.globalAlpha = fade * (0.1 + 0.6 * Math.sqrt(strength));
      ctx.strokeStyle = strength < 0.01 ? LINE : signal > 0 ? GOOD : BAD;
      ctx.lineWidth = 0.6 + 3 * strength;
      ctx.beginPath(); ctx.moveTo(xs[l], ys[l][i]); ctx.lineTo(xs[l + 1], ys[l + 1][j]); ctx.stroke();
    })));

    const node = (x, y, radius, fill, strength) => {
      ctx.globalAlpha = fade;
      ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.globalAlpha = fade * strength; ctx.fillStyle = fill; ctx.fill();
      ctx.globalAlpha = fade; ctx.strokeStyle = "#aab7c4"; ctx.lineWidth = 1; ctx.stroke();
    };
    const label = (text, x, y, align, bold = false) => {
      ctx.globalAlpha = fade; ctx.fillStyle = INK; ctx.textAlign = align;
      ctx.font = `${bold ? "700 " : ""}13px Segoe UI, sans-serif`;
      ctx.fillText(text, x, y);
    };

    game.states[step].forEach((value, i) => {
      node(xs[0], ys[0][i], 12, value >= 0 ? GOOD : BAD, Math.min(1, Math.abs(value) / INPUT_RANGE[i]));
      label(INPUTS[i], xs[0] - 17, ys[0][i] - 2, "right");
      label(showInput(value, i), xs[0] - 17, ys[0][i] + 13, "right", true);
    });
    game.layers.forEach((layer, l) => layer[step].forEach((value, i) => node(xs[l + 1], ys[l + 1][i], 10, WARN, value / peaks[l][i])));
    game.values[step].forEach((value, i) => {
      node(xs[3], ys[3][i], 15, ACCENT, i === chosen && !over ? 1 : 0.08);
      label(i ? "push right" : "push left", xs[3] + 20, ys[3][i] - 2, "left", i === chosen && !over);
      label(value.toFixed(2), xs[3] + 20, ys[3][i] + 13, "left", true);
    });
    label("what it sees", xs[0], 22, "center");
    label(`${game.layers[0][0].length} of ${game.neurons} neurons per layer`, (xs[1] + xs[2]) / 2, 22, "center");
    label("Q-values", xs[3], 22, "center");
    ctx.globalAlpha = 1;
  }

  function cart(ctx, width, x, angle, push, over) {
    const scale = width / 7.4, cx = width / 2 + x * scale, pivot = TRACK - CART_H;
    ctx.globalAlpha = 1; ctx.lineWidth = 3; ctx.strokeStyle = MUTED;
    ctx.beginPath(); ctx.moveTo(0, TRACK + 7); ctx.lineTo(width, TRACK + 7); ctx.stroke();
    for (const side of [-1, 1]) {                                   // the ends of the track
      const end = width / 2 + side * (game.edge * scale + CART_W / 2);
      ctx.fillStyle = MUTED; ctx.fillRect(end - (side < 0 ? 4 : 0), TRACK - 12, 4, 19);
    }

    ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5; ctx.strokeStyle = over ? BAD : MUTED;
    for (const side of [-1, 1]) {                                   // the 12° game-over lines
      ctx.beginPath(); ctx.moveTo(cx, pivot);
      ctx.lineTo(cx + side * Math.sin(game.limit) * POLE, pivot - Math.cos(game.limit) * POLE); ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.fillStyle = INK; ctx.fillRect(cx - CART_W / 2, pivot, CART_W, CART_H);
    for (const wheel of [-17, 17]) {
      ctx.fillStyle = MUTED; ctx.beginPath(); ctx.arc(cx + wheel, TRACK, 7, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.strokeStyle = over ? BAD : "#c98b4f"; ctx.lineWidth = 10; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(cx, pivot); ctx.lineTo(cx + Math.sin(angle) * POLE, pivot - Math.cos(angle) * POLE); ctx.stroke();
    ctx.lineCap = "butt";
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, pivot, 4, 0, 2 * Math.PI); ctx.fill();

    if (push !== null) {
      const side = push ? 1 : -1, tail = cx - side * (CART_W / 2 + 38), tip = cx - side * (CART_W / 2 + 4), y = pivot + CART_H / 2;
      ctx.strokeStyle = ctx.fillStyle = ACCENT; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(tail, y); ctx.lineTo(tip - side * 10, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tip, y); ctx.lineTo(tip - side * 14, y - 10); ctx.lineTo(tip - side * 14, y + 10); ctx.fill();
    }
  }

  function paint(at) {
    const i = Math.min(at, end), step = Math.min(i, last), finished = i >= last;
    const over = finished && game.reason !== null;
    const [x, angle] = i > last ? game.fall[i - last - 1] : [game.states[step][0], game.states[step][2]];
    const { ctx, width } = sizeCanvas(canvas, HEIGHT);
    ctx.clearRect(0, 0, width, HEIGHT);
    network(ctx, width, step, over);                                // behind the cart
    cart(ctx, width, x, angle, finished ? null : game.actions[step], over);

    status.textContent = !finished ? `Steps: ${step}` : over ? `Fell after ${game.score} steps` : `Balanced ${game.score} steps`;
    status.style.color = !finished ? INK : over ? BAD : GOOD;
    detail.textContent = over ? `Game over: ${game.reason}. The network stops, gravity does the rest.`
      : finished ? "It reached the end of the game." : `It pushes ${game.actions[step] ? "right" : "left"}, the higher Q-value.`;
  }

  return { card, canvas, paint, end, falling: at => at >= last && at < end };
}

function render({ model, el }) {
  const v = model.get("view");
  const speed = speedSlider(4);
  const panes = [stage("Before training: untrained network", v.before),
    stage(v.untrained ? "Not trained yet: press Train, then run this cell again" : "After training: the best network", v.after)];
  const end = Math.max(...panes.map(p => p.end));
  let at = 0, token = 0, playing = false;

  const paint = () => panes.forEach(p => p.paint(at));
  function stop() {
    token++; playing = false;
    play.textContent = "Play";
  }
  async function start() {
    if (at >= end) at = 0;
    const mine = ++token;
    playing = true; play.textContent = "Pause";
    paint();
    while (mine === token && at < end) {
      const pause = speed.delay() / 3;
      // a fall always gets at least real time, however fast the slider is
      await sleep(panes.some(p => p.falling(at)) ? Math.max(20, pause) : pause);
      if (mine !== token) return;
      at++; paint();
    }
    if (mine === token) stop();
  }
  const play = h("button", { class: "primary", text: "Play", onclick: () => playing ? stop() : start() });
  const step = h("button", { text: "1 step", onclick: () => { stop(); at = Math.min(end, at + 1); paint(); } });
  const reset = h("button", { text: "Reset", onclick: () => { stop(); at = 0; paint(); } });
  const key = (colour, text) => h("span", {}, h("span", { class: "swatch", style: `background:${colour}` }), text);
  el.append(h("div", { class: "rl" },
    h("div", { class: "row" }, play, step, reset, speed.label),
    h("div", { class: "row", style: "align-items:stretch" }, ...panes.map(p => p.card)),
    h("div", { class: "row muted small" }, key(GOOD, "positive signal"), key(BAD, "negative signal"),
      key(WARN, "active neuron"), key(ACCENT, "chosen push"), h("span", { text: "Dashed lines: game over at 12°" }))));

  const observer = new ResizeObserver(() => requestAnimationFrame(paint));
  panes.forEach(p => observer.observe(p.card));
  paint();
  return () => { stop(); observer.disconnect(); };
}

export default { render };
