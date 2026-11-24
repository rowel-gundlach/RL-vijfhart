function render({ model, el }) {
  const send = message => model.send(message);
  const speed = speedSlider(6);
  const board = h("canvas");
  const boardCard = h("div", { class: "card" }, board);
  const label = h("div", { class: "status", text: "Pick a strategy and press Play" });
  const info = h("div", { class: "muted", text: "" });
  const chart = h("canvas");
  const chartCard = h("div", { class: "card" }, chart);
  let mode = "greedy";
  const modes = [["greedy", "1. Greedy"], ["rl", "2. Learning snake"]];
  const modeButtons = modes.map(([key, text]) => h("button", { text, onclick: () => { mode = key; paintModes(); } }));
  let moment = "now";                     // which version of the learning snake plays
  const momentRow = h("div", { class: "row small" });
  const play = h("button", { class: "primary", text: "Play", onclick: () => send({ type: "play", mode, moment: moment === "now" ? null : moment }) });
  const trainBtn = h("button", { class: "good", text: "Train", onclick: () => send({ type: "train" }) });
  const reset = h("button", { text: "Reset", onclick: () => { queue.length = 0; stop = true; send({ type: "reset" }); label.textContent = "Reset: the snake knows nothing"; } });

  function paintModes() {
    modeButtons.forEach((b, i) => b.className = modes[i][0] === mode ? "on" : "");
    trainBtn.style.display = mode === "rl" ? "" : "none";
    chartCard.style.display = mode === "rl" ? "" : "none";
    paintMoments();
  }

  function paintMoments() {
    const moments = (model.get("view").moments || []);
    if (moment !== "now" && !moments.includes(moment)) moment = "now";
    momentRow.style.display = mode === "rl" && moments.length ? "" : "none";
    const button = (value, text) => h("button", { text, class: moment === value ? "on" : "",
      onclick: () => { moment = value; paintMoments(); } });
    momentRow.replaceChildren(h("span", { class: "muted", text: "Play the snake as it was:" }),
      button("now", "now"), ...moments.map(g => button(g, `after ${g.toLocaleString("en")} games`)));
  }

  el.append(h("div", { class: "rl" },
    h("div", { class: "row" }, ...modeButtons),
    momentRow,
    label,
    h("div", { class: "row" }, h("div", { class: "col" }, boardCard), h("div", { class: "col" }, chartCard, info)),
    h("div", { class: "row" }, play, trainBtn, reset, speed.label)));
  paintModes();

  function draw(frame, size) {
    const side = Math.min((boardCard.clientWidth || 420) - 24, 440);
    board.style.width = side + "px";
    const ratio = window.devicePixelRatio || 1;
    board.width = side * ratio; board.height = side * ratio; board.style.height = side + "px";
    const ctx = board.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const cell = side / size;
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, side, side);
    ctx.strokeStyle = "#eef2f6";
    for (let i = 0; i <= size; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, side); ctx.moveTo(0, i * cell); ctx.lineTo(side, i * cell); ctx.stroke();
    }
    if (frame.apple) {
      ctx.fillStyle = BAD; ctx.beginPath();
      ctx.arc((frame.apple[0] + .5) * cell, (frame.apple[1] + .5) * cell, cell * .36, 0, 7); ctx.fill();
    }
    frame.snake.forEach(([x, y], i) => {
      ctx.fillStyle = i === 0 ? INK : frame.dead ? "#9aa7b4" : ACCENT;
      ctx.fillRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
    });
    if (frame.dead) { ctx.fillStyle = "rgba(214,69,51,.16)"; ctx.fillRect(0, 0, side, side); }
    return { cell, side };
  }

  const queue = [];
  let busy = false, stop = false;

  async function run() {
    if (busy) { stop = true; while (busy) await sleep(20); }
    busy = true;
    while (queue.length) {
      const m = queue.shift();
      stop = false;
      label.textContent = m.label;
      for (let i = 0; i < m.frames.length && !stop; i++) {
        const f = m.frames[i];
        draw(f, m.size);
        if (f.reward > 0) popup(boardCard, `+${f.reward}`, true, 42, 40);
        if (f.reward < 0 && f.dead) popup(boardCard, `${f.reward}`, false, 42, 40);
        info.textContent = `Apples eaten: ${f.score}`;
        await sleep(speed.delay());
      }
      await sleep(Math.max(150, speed.delay()));
      send({ type: "shown" });
    }
    busy = false;
  }

  model.on("msg:custom", message => {
    if (message.type === "replay") { queue.push(message); run(); }
  });

  function paint() {
    const v = model.get("view");
    trainBtn.textContent = v.training ? "Pause training" : v.game >= v.total ? "Training finished" : "Train";
    trainBtn.disabled = !v.training && v.game >= v.total;
    if (v.curve.length) {
      lineChart(chart, [{ name: "apples per game", color: GOOD, data: v.curve }], { title: `Average score, game ${v.game} of ${v.total}`, height: 260, xLabel: "games played" });
    } else {
      chart.getContext("2d").clearRect(0, 0, chart.width, chart.height);
    }
  }
  model.on("change:view", () => { paint(); paintMoments(); });
  paint();
  draw({ snake: [], apple: null }, 10);
}

export default { render };
