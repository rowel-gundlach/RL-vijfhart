const BROKEN_STATE = 10;

function wearColor(wear, levels) {
  const t = wear / (levels - 1);                           // green → amber → red
  const mix = (a, b, f) => a.map((x, i) => Math.round(x + (b[i] - x) * f));
  const rgb = t < .5 ? mix([31, 157, 85], [227, 154, 18], t * 2) : mix([227, 154, 18], [214, 69, 51], (t - .5) * 2);
  return `rgb(${rgb.join(",")})`;
}

function dayEvent(d, v) {
  if (d.state === BROKEN_STATE) return { icon: "♻️", text: `Replaced with a new machine: −€${v.replace_cost}`, kind: "replace" };
  if (d.action === 1) return { icon: "🔧", text: `Planned repair: −€${v.repair_cost}`, kind: "repair" };
  if (d.next === BROKEN_STATE) return { icon: "💥", text: "Broke down! No income today", kind: "broke" };
  return { icon: "", text: `Running: +€${v.revenue}`, kind: "run" };
}

function machineView(title, showThreshold) {
  const gear = h("span", { class: "gear", text: "⚙️" });
  const smoke = h("span", { class: "smoke", text: "💨" });
  const overlay = h("span", { class: "overlay" });
  const machine = h("div", { class: "machine2" }, smoke, gear, overlay);
  const meter = h("div", { class: "meter" });
  const meterLabel = h("div", { class: "muted small" });
  const event = h("div", { class: "event" });
  const money = h("div", { class: "money", text: "€0" });
  const counts = h("div", { class: "muted small" });
  const strip = h("canvas", { class: "strip" });
  const card = h("div", { class: "card col machine-card" }, h("strong", { text: title }), machine, meter, meterLabel, event, money, counts,
    h("div", { class: "muted small", text: "Every day of the year:" }), strip);
  let cells = [], history = [], repairs = 0, breakdowns = 0;

  function build(levels, threshold) {
    cells = Array.from({ length: levels }, (_, i) => h("div", { class: "seg", title: `wear ${i}` }));
    meter.replaceChildren(...cells);
    if (showThreshold && threshold !== null) cells[threshold].classList.add("threshold");
  }

  function paintStrip(v) {
    const { ctx, width, height } = sizeCanvas(strip, 26);
    ctx.fillStyle = "#eef2f6"; ctx.fillRect(0, 0, width, height);
    const w = width / v.days;
    history.forEach((d, i) => {
      const ev = dayEvent(d, v).kind;
      ctx.fillStyle = ev === "replace" ? INK : ev === "broke" ? BAD : ev === "repair" ? ACCENT : wearColor(d.state, v.levels);
      ctx.fillRect(i * w, ev === "run" ? 6 : 0, Math.max(w, 1.5), ev === "run" ? height - 12 : height);
    });
  }

  return {
    card,
    reset(v, threshold) {
      build(v.levels, threshold); history = []; repairs = breakdowns = 0;
      machine.className = "machine2"; overlay.textContent = ""; smoke.style.opacity = 0;
      event.textContent = "Waiting to start"; money.textContent = "€0"; money.style.color = INK;
      meterLabel.textContent = "Wear: new"; counts.textContent = "";
      paintStrip(v);
    },
    show(d, v) {
      const ev = dayEvent(d, v);
      history.push(d);
      if (ev.kind === "repair") repairs++;
      if (ev.kind === "broke") breakdowns++;
      const broken = d.next === BROKEN_STATE;
      const wear = broken ? v.levels - 1 : d.next;
      cells.forEach((c, i) => {
        c.style.background = broken ? "#9aa7b4" : i <= wear ? wearColor(i, v.levels) : "";
      });
      meterLabel.textContent = broken ? "Broken: must be replaced" : `Wear: ${wear} of ${v.levels - 1} · ${v.chance[wear]}% chance it breaks today`;
      machine.className = `machine2 ${broken ? "broken" : ev.kind === "repair" || ev.kind === "replace" ? "service" : wear >= v.levels - 3 ? "shaky" : "running"}`;
      overlay.textContent = ev.icon;
      smoke.style.opacity = broken ? 1 : Math.max(0, (wear - 3) / (v.levels - 4));
      event.textContent = ev.text;
      event.className = `event ${ev.kind}`;
      money.textContent = `€${d.total.toLocaleString("en")}`;
      money.style.color = d.total >= 0 ? GOOD : BAD;
      counts.textContent = `Day ${d.day} · ${repairs} repairs · ${breakdowns} breakdowns`;
      if (ev.kind !== "run" && ev.kind !== "broke") popup(card, `−€${-d.money}`, false, 60, 18);
      paintStrip(v);
    },
  };
}

function render({ model, el }) {
  const send = message => model.send(message);
  const speed = speedSlider(6);
  const learned = machineView("Learned policy", true), fixed = machineView("Fixed rule: run until it breaks", false);
  const status = h("div", { class: "status", text: "" });
  const policy = h("div", { class: "card" });
  const canvas = h("canvas");
  const chartCard = h("div", { class: "card" }, canvas);
  const train = h("button", { class: "good", text: "Train", onclick: () => send({ type: "train" }) });
  const play = h("button", { class: "primary", text: "Play a year", onclick: () => send({ type: "play" }) });
  const reset = h("button", { text: "Reset", onclick: () => { token++; send({ type: "reset" }); } });
  const legend = h("div", { class: "row muted small" },
    h("span", {}, h("span", { class: "swatch", style: `background:${wearColor(0, 10)}` }), "running (colour = wear)"),
    h("span", {}, h("span", { class: "swatch", style: `background:${ACCENT}` }), "repair"),
    h("span", {}, h("span", { class: "swatch", style: `background:${BAD}` }), "breakdown"),
    h("span", {}, h("span", { class: "swatch", style: `background:${INK}` }), "replace"));
  el.append(h("div", { class: "rl" },
    h("div", { class: "row" }, train, play, reset, speed.label), status, policy,
    h("div", { class: "row" }, learned.card, fixed.card), legend, chartCard));

  let token = 0, lastThreshold;
  function paint() {
    const v = model.get("view");
    status.textContent = v.progress !== null ? `Training ${Math.round(v.progress * 100)}%`
      : v.trained ? "Trained: press Play a year" : "Not trained yet: the agent just does nothing. Press Train.";
    const label = v.trained ? v.threshold === null ? "The agent never repairs" : `The agent repairs from wear ${v.threshold} onwards` : "What the agent does at each wear level";
    policy.replaceChildren(h("strong", { text: label }), h("table", { class: "values policy" },
      h("tr", {}, h("th", { text: "wear" }), ...v.policy.map((_, i) => h("th", { text: i })), h("th", { text: "broken" })),
      h("tr", {}, h("th", { text: "breaks today" }), ...v.chance.map(c => h("td", { class: "muted", text: `${c}%` })), h("td", { text: "" })),
      h("tr", {}, h("th", { text: "agent does" }), ...v.policy.map((p, i) => h("td", {
        text: v.trained ? (p === "repair" ? "🔧 repair" : "nothing") : "?",
        style: `background:${!v.trained ? "#eef2f6" : p === "repair" ? "#d6e4fb" : "#dff3e7"}${v.trained && i === v.threshold ? ";outline:3px solid " + INK : ""}`,
      })), h("td", { text: "♻️ replace", style: "background:#e3e8ee" }))));
    if (v.threshold !== lastThreshold || !v.trained) { learned.reset(v, v.trained ? v.threshold : null); fixed.reset(v, null); }
    lastThreshold = v.threshold;
  }
  model.on("change:view", paint);
  paint();

  model.on("msg:custom", async m => {
    if (m.type !== "replay") return;
    const mine = ++token, v = model.get("view");
    learned.reset(v, v.trained ? v.threshold : null); fixed.reset(v, null);
    const a = [], b = [];
    for (let i = 0; i < m.learned.length && mine === token; i++) {
      learned.show(m.learned[i], v); fixed.show(m.fixed[i], v);
      a.push([i + 1, m.learned[i].total]); b.push([i + 1, m.fixed[i].total]);
      lineChart(canvas, [{ name: "learned", color: GOOD, data: a }, { name: "run until it breaks", color: BAD, data: b }], { title: "Profit so far (€)", xLabel: "days", height: 240 });
      await sleep(speed.delay());
    }
  });
}

export default { render };
