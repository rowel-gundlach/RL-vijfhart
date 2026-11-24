// The sentence and the two word lists stay in place: only the words and the bars change.
function render({ model, el }) {
  const send = message => model.send(message);
  const speed = speedSlider(5);
  const first = model.get("view");

  const sentence = h("div", { class: "sentence", style: "line-height:2;text-align:center" }, first.prompt);
  const slots = first.slots.map((slot, n) => {
    const blank = h("span", { style: "display:inline-block;min-width:130px;text-align:center;line-height:1.3;border-bottom:3px solid #2f6fde;margin:0 6px", text: "…" });
    sentence.append(` ${slot.from} `, blank);
    const card = h("div", { class: "card col", style: "flex-basis:360px;gap:2px;border:3px solid transparent" },
      h("strong", { text: `${n + 1}. The word after “${slot.from}”` }));
    const rows = slot.options.map(word => {
      const fill = h("div"), share = h("div"), change = h("div", { class: "small" });
      const row = h("div", { class: "bar-row", style: "grid-template-columns:78px minmax(0,1fr) 50px 38px" },
        h("div", { text: word }), h("div", { class: "bar" }, fill), share, change);
      card.append(row);
      return { row, fill, share, change };
    });
    return { blank, card, rows };
  });

  const generate = h("button", { class: "primary", text: "Generate a sentence", onclick: () => send({ type: "generate" }) });
  const up = h("button", { class: "good", text: "👍 Good", onclick: () => send({ type: "feedback", good: true }) });
  const down = h("button", { class: "bad", text: "👎 Bad", onclick: () => send({ type: "feedback", good: false }) });
  const reset = h("button", { text: "Reset", onclick: () => send({ type: "reset" }) });
  const rounds = h("div", { class: "muted" });
  el.append(h("div", { class: "rl" }, sentence,
    h("div", { class: "row", style: "align-items:stretch" }, ...slots.map(s => s.card)),
    h("div", { class: "row" }, generate, up, down, reset, speed.label), rounds));

  // shown = how many gaps are filled, active = the gap being chosen right now
  function draw(v, shown, active = -1) {
    slots.forEach(({ blank, card, rows }, n) => {
      const slot = v.slots[n], chosen = n < shown ? slot.chosen : -1;
      blank.textContent = chosen >= 0 ? slot.options[chosen] : "…";
      card.style.borderColor = n === active ? ACCENT : "transparent";
      rows.forEach(({ row, fill, share, change }, i) => {
        row.classList.toggle("chosen", i === chosen);
        fill.style.width = `${slot.probabilities[i] * 100}%`;
        share.textContent = `${Math.round(slot.probabilities[i] * 100)}%`;
        const points = Math.round(slot.change[i] * 100);
        change.textContent = points ? `${points > 0 ? "+" : "−"}${Math.abs(points)}` : "";
        change.style.color = points > 0 ? GOOD : BAD;
      });
    });
  }

  let token = 0, seen = first.generated;
  async function paint() {
    const v = model.get("view"), mine = ++token;
    const fresh = v.generated > seen;
    seen = v.generated;
    rounds.textContent = `Feedback rounds so far: ${v.rounds}. One thumb changes both words.`;
    up.disabled = down.disabled = true;

    if (fresh && !v.rated) {
      for (let n = 0; n < slots.length; n++) {
        draw(v, n, n);
        await sleep(speed.delay() * 4);
        if (mine !== token) return;
        draw(v, n + 1, n);
        await sleep(speed.delay() * 2);
        if (mine !== token) return;
      }
    }
    draw(v, slots.length);
    up.disabled = down.disabled = !v.generated || v.rated;
  }
  model.on("change:view", paint);
  paint();
  return () => { token++; model.off("change:view", paint); };
}

export default { render };
