function sellerView(title) {
  const tag = h("span", { class: "price-tag", text: "€–" });
  const today = h("div", { class: "small", style: "font-size:17px;margin:4px 0" });
  const seats = h("div", { class: "seats" });
  const seatLabel = h("div", { class: "muted small" });
  const money = h("div", { class: "money", text: "€0" });
  const card = h("div", { class: "card col seller" }, h("strong", { text: title }),
    h("div", { class: "row" }, h("span", { class: "muted", text: "Today's price" }), tag),
    today,
    seatLabel, seats, h("div", { class: "row" }, h("span", { class: "muted", text: "Revenue" }), money));
  let cells = [], perSeat = 1, sold = 0, lastPrice = null;

  return {
    card,
    reset(v) {
      perSeat = Math.max(1, Math.ceil(v.tickets / 300));
      const n = Math.ceil(v.tickets / perSeat);
      seats.style.gridTemplateColumns = `repeat(${Math.min(25, n)}, 1fr)`;
      cells = Array.from({ length: n }, () => h("span"));
      seats.replaceChildren(...cells);
      sold = 0; lastPrice = null;
      tag.textContent = "€–"; tag.className = "price-tag";
      today.textContent = "";
      seatLabel.textContent = `Seats: 0 of ${v.tickets} sold${perSeat > 1 ? ` (one square = ${perSeat} seats)` : ""}`;
      money.textContent = "€0";
    },
    show(d, v) {
      tag.textContent = `€${d.price}`;
      tag.className = `price-tag ${lastPrice === null || d.price === lastPrice ? "" : d.price > lastPrice ? "up" : "down"}`;
      lastPrice = d.price;
      today.textContent = `Today: ${d.sold} ${d.sold === 1 ? "person" : "people"} bought a ticket`;
      const before = Math.floor(sold / perSeat);
      sold += d.sold;
      const after = Math.floor(sold / perSeat);
      cells.forEach((c, i) => c.className = i < before ? "sold" : i < after ? "today" : "");
      seatLabel.textContent = `Seats: ${sold} of ${v.tickets} sold${perSeat > 1 ? ` (one square = ${perSeat} seats)` : ""}`;
      money.textContent = `€${d.total.toLocaleString("en")}`;
      if (d.sold) popup(card, `+€${(d.price * d.sold).toLocaleString("en")}`, true, 70, 12);
    },
  };
}

function render({ model, el }) {
  const send = message => model.send(message);
  const speed = speedSlider(4);
  const status = h("div", { class: "status", text: "" });
  const stats = h("div", { class: "muted" });
  const learned = sellerView("Learned prices"), fixed = sellerView("Fixed price");
  const price = h("canvas"), revenue = h("canvas");
  const most = model.get("view").max_concerts;
  const howMany = h("input", { type: "number", min: 1, max: most, step: 1, value: model.get("view").seasons,
    style: "width:110px;font:inherit;padding:6px", "aria-label": `Number of concerts to train on, at most ${most}` });
  const concerts = () => {                 // keep the box between 1 and the upper bound
    const n = Math.min(most, Math.max(1, Math.round(Number(howMany.value) || 1)));
    howMany.value = n;
    return n;
  };
  howMany.onchange = concerts;
  const train = h("button", { class: "good", text: "Train", onclick: () => send({ type: "train", concerts: concerts() }) });
  const play = h("button", { class: "primary", text: "Sell the concert", onclick: () => send({ type: "play" }) });
  const reset = h("button", { text: "Reset", onclick: () => { token++; learned.reset(model.get("view")); fixed.reset(model.get("view")); send({ type: "reset" }); } });
  const plan = h("div", { class: "card" });
  const legend = h("div", { class: "row muted small" },
    h("span", {}, h("span", { class: "swatch", style: `background:${GOOD}` }), "seat sold today"),
    h("span", {}, h("span", { class: "swatch", style: `background:${ACCENT}` }), "seat sold earlier"));
  el.append(h("div", { class: "rl" },
    h("div", { class: "row" }, train, howMany, h("span", { class: "muted", text: `concerts (at most ${most.toLocaleString("en")})` }), play, reset, speed.label),
    status, stats, plan,
    h("div", { class: "row" }, learned.card, fixed.card), legend,
    h("div", { class: "card" }, price), h("div", { class: "card" }, revenue)));

  function paintPlan(v) {
    if (!v.plan) { plan.style.display = "none"; return; }
    plan.style.display = "";
    const cells = v.plan.map(d => h("td", { style: "text-align:center" },
      h("div", { style: "font-size:18px;font-weight:700", text: d.price === null ? "–" : `€${d.price}` }),
      h("div", { class: "muted small", text: d.price === null ? "sold out" : `${Math.round(d.share * 100)}%` })));
    plan.replaceChildren(h("strong", { text: "The learned policy: the price for each day" }),
      h("div", { style: "overflow-x:auto" }, h("table", { class: "values" },
        h("tr", {}, ...v.plan.map(d => h("th", { style: "text-align:center", text: `Day ${d.day}` }))),
        h("tr", {}, ...cells))),
      h("div", { class: "muted small", text: `The price it picks on that day in most concerts. The % shows how often: it also looks at how many tickets are left. The fixed seller asks €${v.fixed} every day.` }));
  }

  let token = 0;
  function paint() {
    const v = model.get("view");
    train.disabled = v.progress !== null;
    status.textContent = v.progress !== null ? `Training ${Math.round(v.progress * 100)}%: ${v.concerts.toLocaleString("en")} concerts so far`
      : v.concerts ? `Trained on ${v.concerts.toLocaleString("en")} concerts: sell one, or train further`
      : "Not trained yet: type a number of concerts and press Train (or sell anyway to see an untrained seller)";
    stats.textContent = v.learned_avg !== null
      ? `Average revenue per concert: learned €${v.learned_avg.toLocaleString("en")}, fixed €${v.fixed} price €${v.fixed_avg.toLocaleString("en")}`
      : `The fixed seller always asks €${v.fixed}: the best single price, found by trying them all.`;
    paintPlan(v);
    if (!token) { learned.reset(v); fixed.reset(v); }
  }
  model.on("change:view", paint);
  paint();

  model.on("msg:custom", async m => {
    if (m.type !== "replay") return;
    const mine = ++token, v = model.get("view"), days = Math.max(m.learned.length, m.fixed.length);
    learned.reset(v); fixed.reset(v);
    for (let n = 1; n <= days && mine === token; n++) {
      const a = m.learned.slice(0, n), b = m.fixed.slice(0, n);
      if (a[n - 1]) learned.show(a[n - 1], v);
      if (b[n - 1]) fixed.show(b[n - 1], v);
      lineChart(price, [
        { name: "learned", color: GOOD, data: a.map(d => [d.day, d.price]) },
        { name: `fixed €${v.fixed}`, color: MUTED, data: b.map(d => [d.day, d.price]) },
      ], { title: "Ticket price (€)", xLabel: "days of selling", height: 220, step: true });
      lineChart(revenue, [{ name: "learned", color: GOOD, data: a.map(d => [d.day, d.total]) }, { name: "fixed", color: MUTED, data: b.map(d => [d.day, d.total]) }], { title: "Revenue so far (€)", xLabel: "days of selling", height: 220 });
      await sleep(speed.delay() * 6);
    }
  });
}

export default { render };
