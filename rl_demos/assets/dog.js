const DOG_SVG = `
<svg class="dog" viewBox="0 0 340 320" role="img" aria-label="Cartoon dog">
  <g class="whole">
    <g class="tail"><path d="M72 170 C40 150 38 118 52 98" fill="none" stroke="#b9793f" stroke-width="16" stroke-linecap="round"/></g>
    <g class="back-legs">
      <rect x="84" y="205" width="28" height="68" rx="13" fill="#a96c36"/>
      <rect x="118" y="205" width="28" height="68" rx="13" fill="#b9793f"/>
    </g>
    <rect x="186" y="205" width="28" height="68" rx="13" fill="#a96c36"/>
    <rect x="218" y="205" width="28" height="68" rx="13" fill="#b9793f"/>
    <ellipse cx="160" cy="180" rx="100" ry="58" fill="#c98a4b"/>
    <ellipse cx="152" cy="196" rx="62" ry="32" fill="#e3b888"/>
    <g class="head">
      <g class="ear"><path d="M196 62 C170 58 160 100 172 128 C192 124 204 96 196 62Z" fill="#8f5a2c"/></g>
      <circle cx="230" cy="100" r="62" fill="#c98a4b"/>
      <g class="ear"><path d="M262 58 C292 54 306 96 294 128 C272 124 256 94 262 58Z" fill="#8f5a2c"/></g>
      <ellipse cx="238" cy="124" rx="38" ry="28" fill="#f0d2ad"/>
      <circle class="eye-open" cx="212" cy="88" r="8" fill="#1f2d3d"/>
      <circle class="eye-open" cx="256" cy="88" r="8" fill="#1f2d3d"/>
      <path class="eye-closed" d="M202 88 q10 8 20 0 M246 88 q10 8 20 0" stroke="#1f2d3d" stroke-width="4" fill="none" stroke-linecap="round"/>
      <ellipse cx="238" cy="110" rx="11" ry="8" fill="#1f2d3d"/>
      <path class="mouth-plain" d="M224 132 q14 8 28 0" stroke="#1f2d3d" stroke-width="4" fill="none" stroke-linecap="round"/>
      <g class="mouth-smile"><path d="M216 126 q22 30 44 0Z" fill="#7c2d2d"/><ellipse cx="238" cy="142" rx="10" ry="8" fill="#ec7f8a"/></g>
      <path class="mouth-sad" d="M222 140 q16 -14 32 0" stroke="#1f2d3d" stroke-width="4" fill="none" stroke-linecap="round"/>
    </g>
    <g class="zzz"><text x="292" y="44" font-size="30" font-weight="700" fill="#2f6fde">Z z z</text></g>
  </g>
  <g class="bubble"><path d="M20 12 h120 a14 14 0 0 1 14 14 v30 a14 14 0 0 1 -14 14 h-64 l-22 22 v-22 h-34 a14 14 0 0 1 -14 -14 v-30 a14 14 0 0 1 14 -14z" transform="translate(150 -30)" fill="#fff" stroke="#1f2d3d" stroke-width="3"/><text x="194" y="16" font-size="28" font-weight="800" fill="#1f2d3d">WOOF!</text></g>
  <g class="hearts"><text x="170" y="30" font-size="34" fill="#d64533">&#10084;</text><text x="290" y="24" font-size="26" fill="#d64533">&#10084;</text><text x="110" y="62" font-size="24" fill="#d64533">&#10084;</text></g>
  <g class="ball"><circle cx="274" cy="268" r="17" fill="#e39a12" stroke="#a86e05" stroke-width="3"/></g>
</svg>`;

const ANIMATION = { "sit": "a-sit", "roll over": "a-roll", "fetch": "a-fetch", "bark": "a-bark", "sleep": "a-sleep" };

function render({ model, el }) {
  const send = message => model.send(message);
  const stage = h("div", { class: "dog-stage" });
  stage.innerHTML = DOG_SVG;
  const dog = stage.firstElementChild;
  const said = h("div", { class: "said" });
  const choice = h("div", { class: "choice" });
  const how = h("div", { class: "how" });
  const card = h("div", { class: "card dog-card" }, said, stage, choice, how);
  const input = h("input", { type: "text", placeholder: "type any word", value: "sit" });
  const say = () => { send({ type: "command", word: input.value }); };
  input.onkeydown = e => { if (e.key === "Enter") say(); };
  const go = h("button", { class: "primary", text: "Say it", onclick: say });
  const treat = h("button", { class: "good", text: "", onclick: () => send({ type: "feedback", good: true }) });
  const no = h("button", { class: "bad", text: "", onclick: () => send({ type: "feedback", good: false }) });
  const reset = h("button", { text: "Reset", onclick: () => send({ type: "reset" }) });
  const sliders = {};
  const sendSettings = () => send({ type: "settings", ...Object.fromEntries(Object.entries(sliders).map(([k, s]) => [k, Number(s.input.value)])) });
  function slider(key, text) {
    const input = h("input", { type: "range", min: 0, max: 1, step: 0.05 });
    const value = h("span", { class: "muted" });
    input.oninput = () => { value.textContent = Number(input.value).toFixed(2); sendSettings(); };
    sliders[key] = { input, value };
    return h("label", { class: "speed" }, text, input, value);
  }
  const settings = h("div", { class: "row" },
    slider("exploration", "Exploration"), slider("learning_rate", "Learning rate"));
  let revealed = false;
  const brain = h("div", { class: "card", style: "display:none" });
  const reveal = h("button", { text: "Reveal the dog's brain", onclick: () => {
    revealed = !revealed;
    brain.style.display = revealed ? "" : "none";
    reveal.textContent = revealed ? "Hide the dog's brain" : "Reveal the dog's brain";
  } });

  el.append(h("div", { class: "rl" },
    settings, card,
    h("div", { class: "row" }, input, go, treat, no, reveal, reset),
    brain));

  let lastTrick = -1, lastMood = "";

  function paint() {
    const v = model.get("view");
    for (const [key, s] of Object.entries(sliders)) {
      if (document.activeElement !== s.input) s.input.value = v[key];
      s.value.textContent = Number(v[key]).toFixed(2);
    }
    treat.textContent = `Treat (+${v.treat})`; no.textContent = `No! (${v.no})`;
    const canFeedback = v.action !== null && !v.mood;
    treat.disabled = no.disabled = !canFeedback;

    const words = Object.keys(v.values);
    const max = Math.max(1, ...words.flatMap(w => v.values[w].map(Math.abs)));
    brain.replaceChildren(h("table", { class: "values" },
      h("tr", {}, h("th", { text: "word" }), ...v.actions.map(a => h("th", { text: a }))),
      ...words.map(w => h("tr", {}, h("th", { text: w }),
        ...v.values[w].map((value, i) => h("td", {
          text: value.toFixed(2),
          style: `background:${valueColor(value, max)};${w === v.word && i === v.action ? "outline:3px solid #1f2d3d" : ""}`,
        }))))));
    if (!words.length) brain.append(h("div", { class: "muted", text: "Empty: the dog knows nothing yet." }));

    if (v.action === null) {
      dog.setAttribute("class", "dog"); said.style.visibility = "hidden";
      choice.replaceChildren(h("span", { class: "muted", text: "Give your dog a command" })); how.replaceChildren();
      lastTrick = -1; lastMood = ""; return;
    }
    const name = v.actions[v.action];
    if (v.tricks !== lastTrick) {
      lastTrick = v.tricks;
      dog.setAttribute("class", "dog");
      void dog.getBoundingClientRect();           // restart the animation
    }
    dog.setAttribute("class", `dog ${ANIMATION[name]} ${v.mood}`);
    said.style.visibility = "";
    said.replaceChildren(h("span", { class: "muted", text: "You: " }), `"${v.word}"`);
    choice.replaceChildren(h("span", { class: "muted", text: "the dog chose " }),
      h("span", { class: `trick ${v.mood}`, text: name }));
    const exploring = v.how === "exploring";
    how.replaceChildren(h("span", { class: `badge ${exploring ? "explore" : "exploit"}`, text: exploring ? "Exploring: trying something random" : v.how === "guessing" ? "Guessing: no clear favourite yet" : "Using what it learned" }));
    if (v.mood && v.mood !== lastMood) popup(card, v.mood === "happy" ? `+${v.treat}` : `${v.no}`, v.mood === "happy", 20, 40);
    lastMood = v.mood;
  }

  model.on("change:view", paint);
  paint();
}

export default { render };
