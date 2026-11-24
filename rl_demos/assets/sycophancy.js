// You rate the answers. Each 👍 or 👎 retrains one tiny network, right in the browser.
// Orange pulls towards "you're right", blue towards a correction.
const TO_RIGHT = WARN, TO_WRONG = ACCENT;
const ASKS = 0.1, INSISTS = 1.0;
const GROUPS = 600;                     // groups of 64 simulated ratings
const percent = p => `${Math.round(p * 100)}%`;
const colour = x => x >= 0 ? TO_RIGHT : TO_WRONG;
const copy = net => ({ first: net.first.map(row => row.slice()), second: net.second.slice() });

function forward(net, inputs) {
  const hidden = net.second.map((_, j) => Math.tanh(inputs[0] * net.first[0][j] + inputs[1] * net.first[1][j]));
  const score = hidden.reduce((sum, x, j) => sum + x * net.second[j], 0);
  return { hidden, agree: 1 / (1 + Math.exp(-score)) };
}

// The same rule as part A: make the rated answer more likely after 👍, less likely after 👎.
function learn(net, inputs, agreed, reward, rate) {
  const { hidden, agree } = forward(net, inputs);
  const signal = reward * ((agreed ? 1 : 0) - agree);
  hidden.forEach((x, j) => {
    const back = signal * (1 - x * x) * net.second[j];
    net.second[j] += rate * x * signal;
    inputs.forEach((input, i) => net.first[i][j] += rate * input * back);
  });
}

function render({ model, el }) {
  const v = model.get("view"), q = v.question;
  let net = copy(v.start), claimRight = false, push = INSISTS, last = null, mine = 0, ups = 0, downs = 0, busy = false;

  const inputs = (right = claimRight, pressure = push) => [right ? q.certainty : -q.certainty, pressure];
  const choice = (options, get, set) => options.map(([value, text]) => {
    const button = h("button", { text, onclick: () => { set(value); last = null; paint(); } });
    button.check = () => button.className = get() === value ? "on" : "";
    return button;
  });
  const claimButtons = choice([[true, q.right], [false, q.wrong]], () => claimRight, x => claimRight = x);
  const pushButtons = choice([[ASKS, "asks"], [INSISTS, "insists"]], () => push, x => push = x);
  const said = h("div", { class: "answer" });
  const askBtn = h("button", { class: "primary", text: "Ask the model", onclick: ask });
  const answer = h("div", { class: "answer", style: "font-weight:700" });
  const up = h("button", { class: "good", text: "👍 Good answer", onclick: () => rate(1) });
  const down = h("button", { text: "👎 Bad answer", onclick: () => rate(-1) });
  const canvas = h("canvas");
  const grid = h("div");
  const crowd = h("button", { text: "Let thousands of people rate", onclick: letPeopleVote });
  const reset = h("button", { text: "Back to: trained on text", onclick: () => { mine++; busy = false; net = copy(v.start); last = null; ups = downs = 0; paint(); } });
  const status = h("div", { class: "muted" });

  el.append(h("div", { class: "rl" },
    h("div", { class: "row" }, h("b", { text: "The user says the capital is" }), ...claimButtons, h("b", { text: "and" }), ...pushButtons),
    said,
    h("div", { class: "row" }, askBtn),
    answer,
    h("div", { class: "row" }, up, down, h("span", { class: "muted", text: "You rate the answer, like the buttons under every ChatGPT answer." })),
    h("div", { class: "card", style: "margin-top:8px" }, canvas),
    h("div", { class: "card", style: "margin-top:8px" }, grid),
    h("div", { class: "row" }, crowd, reset),
    status));

  function ask() {
    const p = forward(net, inputs()).agree;
    last = { agreed: Math.random() < p, chance: p, inputs: inputs() };
    paint();
  }

  function rate(reward) {
    if (!last) return;
    learn(net, last.inputs, last.agreed, reward, v.thumb_strength);
    reward > 0 ? ups++ : downs++;
    last = null;
    paint();
  }

  // Simulated people, in groups of 64, as in rlhf.py: many questions, about many facts the model is more or
  // less sure of (and sometimes wrong about). A thumb up is likely for a correct answer, a little more for agreeing.
  function group() {
    const batch = [];
    for (let n = 0; n < 64; n++) {
      const userRight = Math.random() < 0.5, certainty = 0.1 + 0.9 * Math.random();
      const modelRight = Math.random() < 0.5 + 0.5 * certainty;
      const x = [userRight === modelRight ? certainty : -certainty, Math.random()];
      const agreed = Math.random() < forward(net, x).agree;
      const up = Math.random() < 0.1 + v.likes_agreement * agreed + v.likes_truth * (agreed === userRight);
      batch.push({ x, agreed, up: up ? 1 : 0 });
    }
    const average = batch.reduce((sum, b) => sum + b.up, 0) / batch.length;
    const grads = { first: net.first.map(row => row.map(() => 0)), second: net.second.map(() => 0) };
    for (const b of batch) {             // better than the group's average = 👍, worse = 👎
      const { hidden, agree } = forward(net, b.x);
      const signal = (b.up - average) * ((b.agreed ? 1 : 0) - agree);
      hidden.forEach((value, j) => {
        grads.second[j] += value * signal / batch.length;
        b.x.forEach((input, i) => grads.first[i][j] += input * signal * (1 - value * value) * net.second[j] / batch.length);
      });
    }
    net.second.forEach((_, j) => { net.second[j] += 0.25 * grads.second[j]; net.first.forEach((row, i) => row[j] += 0.25 * grads.first[i][j]); });
  }

  async function letPeopleVote() {
    const token = ++mine; busy = true;
    for (let round = 1; round <= GROUPS && token === mine; round++) {
      group();
      if (round % 10 === 0) {
        status.textContent = `${(round * 64).toLocaleString("en")} of ${(GROUPS * 64).toLocaleString("en")} simulated ratings`;
        paint(false);
        await sleep(30);
      }
    }
    busy = false;
    paint();
  }

  function drawNetwork(x, result) {
    const { ctx, width } = sizeCanvas(canvas, 240);
    ctx.clearRect(0, 0, width, 240);
    const xs = [Math.max(150, width * 0.25), width * 0.52, width - Math.max(140, width * 0.22)];
    const inY = [70, 170], midY = [45, 120, 195], outY = [70, 170];
    const line = (x1, y1, x2, y2, pull) => {
      const size = Math.abs(pull);
      ctx.globalAlpha = Math.min(1, 0.15 + size / 2);
      ctx.strokeStyle = colour(pull); ctx.lineWidth = Math.min(16, 1 + 2.5 * size);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    };
    x.forEach((input, i) => net.second.forEach((out, j) => {
      const signal = input * net.first[i][j];
      line(xs[0], inY[i], xs[1], midY[j], Math.abs(signal) * Math.sign(signal * out));
    }));
    result.hidden.forEach((value, j) => {
      const pull = value * net.second[j];
      line(xs[1], midY[j], xs[2], outY[pull >= 0 ? 0 : 1], pull);
    });
    const node = (cx, cy, radius, fill, strength) => {
      ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff"; ctx.fill();
      ctx.globalAlpha = Math.min(1, strength); ctx.fillStyle = fill; ctx.fill();
      ctx.globalAlpha = 1; ctx.strokeStyle = "#aab7c4"; ctx.lineWidth = 1.5; ctx.stroke();
    };
    const label = (text, lx, ly, align, bold = false) => {
      ctx.fillStyle = INK; ctx.textAlign = align; ctx.font = `${bold ? "700 " : ""}15px Segoe UI, sans-serif`;
      ctx.fillText(text, lx, ly);
    };
    node(xs[0], inY[0], 20, colour(x[0]), 1);
    node(xs[0], inY[1], 20, TO_RIGHT, x[1]);
    result.hidden.forEach((value, j) => node(xs[1], midY[j], 17, colour(value * net.second[j]), Math.abs(value)));
    node(xs[2], outY[0], 24, TO_RIGHT, result.agree);
    node(xs[2], outY[1], 24, TO_WRONG, 1 - result.agree);
    label(`the claim: ${claimRight ? q.right : q.wrong}`, xs[0] - 26, inY[0] - 3, "right", true);
    label(claimRight ? "matches what it knows" : "clashes with what it knows", xs[0] - 26, inY[0] + 16, "right");
    label(`the user ${push === INSISTS ? "insists" : "asks"}`, xs[0] - 26, inY[1] - 3, "right", true);
    label(push === INSISTS ? "high pressure" : "low pressure", xs[0] - 26, inY[1] + 16, "right");
    label("“you're right”", xs[2] + 30, outY[0] - 3, "left", true);
    label(percent(result.agree), xs[2] + 30, outY[0] + 16, "left");
    label("a correction", xs[2] + 30, outY[1] - 3, "left", true);
    label(percent(1 - result.agree), xs[2] + 30, outY[1] + 16, "left");
    ctx.textAlign = "left";
  }

  function paint(full = true) {
    [...claimButtons, ...pushButtons].forEach(b => b.check());
    const claim = claimRight ? q.right : q.wrong;
    said.textContent = push === INSISTS ? `User: “I'm absolutely sure the capital of the Netherlands is ${claim}. Right?”`
                                        : `User: “Is the capital of the Netherlands ${claim}?”`;
    if (last) {
      const correction = claimRight ? "" : ` No, the capital is ${q.right}. ${q.wrong} is the seat of government.`;
      const text = last.agreed ? "Yes, you're right!" : claimRight ? `No, I don't think so.` : correction.trim();
      answer.textContent = `Model: “${text}”  — rate this answer:`;
      answer.style.borderLeftColor = last.agreed === claimRight ? GOOD : BAD;
    } else {
      answer.textContent = "Press “Ask the model” to get an answer.";
      answer.style.borderLeftColor = LINE;
    }
    up.disabled = down.disabled = !last || busy;
    askBtn.disabled = crowd.disabled = busy;

    const x = inputs();
    drawNetwork(x, forward(net, x));
    const cell = (right, pressure) => {
      const p = forward(net, inputs(right, pressure)).agree, here = right === claimRight && pressure === push;
      const bad = right ? 1 - p : p;            // how often this answer is wrong
      return h("td", { text: percent(p), style: `background:rgba(214,69,51,${0.05 + 0.55 * bad});${here ? "outline:3px solid #1f2d3d;font-weight:700" : ""}` });
    };
    grid.replaceChildren(h("b", { text: "How often it says “you're right” (redder = more often wrong)" }),
      h("table", { class: "values" },
        h("tr", {}, h("th", { text: "" }), h("th", { text: "user asks" }), h("th", { text: "user insists" })),
        h("tr", {}, h("th", { text: `${q.right} (right)` }), cell(true, ASKS), cell(true, INSISTS)),
        h("tr", {}, h("th", { text: `${q.wrong} (wrong)` }), cell(false, ASKS), cell(false, INSISTS))));
    if (full) status.textContent = `Your ratings so far: ${ups} 👍, ${downs} 👎`;
  }

  const observer = new ResizeObserver(() => requestAnimationFrame(() => paint(false)));
  observer.observe(el);
  paint();
  return () => { mine++; observer.disconnect(); };
}

export default { render };
