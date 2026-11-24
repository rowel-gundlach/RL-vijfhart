function render({ model, el }) {
  const send = message => model.send(message);
  const status = h("div", { class: "status" });
  const canvas = h("canvas");
  const train = h("button", { class: "good", text: "Train", onclick: () => send({ type: "train" }) });
  const reset = h("button", { text: "Reset", onclick: () => send({ type: "reset" }) });
  el.append(h("div", { class: "rl" }, h("div", { class: "row" }, train, reset), status, h("div", { class: "card" }, canvas)));

  function paint() {
    const v = model.get("view");
    train.textContent = v.running ? "Pause training" : "Train";
    train.disabled = v.done && !v.running;
    status.textContent = v.done ? `Done. Run the next cell to compare before and after.` : v.running ? `Game ${v.game} of up to ${v.total}. Best test so far: ${Math.max(0, v.best).toFixed(0)} steps` : "Press Train";
    lineChart(canvas, [{ name: "steps upright", color: GOOD, data: v.curve }], { title: "Learning curve: average steps upright (500 is the maximum)", xLabel: "games played", height: 280 });
  }
  model.on("change:view", paint);
  paint();
}

export default { render };
