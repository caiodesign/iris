// companion/web/app.js
const chat = document.getElementById("chat");
const statusText = document.getElementById("status-text");
const hint = document.getElementById("hint");
const conn = document.getElementById("conn");
const orb = document.getElementById("orb");
const brainSel = document.getElementById("brain");
const earsSel = document.getElementById("ears");
const pttBox = document.getElementById("ptt");
const drawer = document.getElementById("drawer");
const scrim = document.getElementById("scrim");
const memoryBtn = document.getElementById("memory-btn");
const memoryContent = document.getElementById("memory-content");
const toast = document.getElementById("toast");
const toastText = document.getElementById("toast-text");

let ws = null;
let running = false;
let memory = { durable: "", timeline: "" };
let activeTab = "durable";

const STATUS_LABELS = {
  idle: "idle",
  loading: "waking up",
  listening: "listening",
  thinking: "thinking",
  speaking: "speaking",
};

function setStatus(state) {
  document.body.dataset.state = state;
  statusText.textContent = STATUS_LABELS[state] || state;
}

function setRunning(isRunning) {
  running = isRunning;
  document.body.classList.toggle("running", isRunning);
  orb.setAttribute("aria-label", isRunning ? "End session" : "Start session");
  hint.textContent = isRunning
    ? "press the light to end the session"
    : "press the light to start talking";
  for (const el of [brainSel, earsSel, pttBox]) el.disabled = isRunning;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addTurn(role, text) {
  const turn = document.createElement("div");
  turn.className = "turn " + role;
  const head = document.createElement("div");
  head.className = "turn-head";
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = role === "user" ? "you" : "iris";
  const stamp = document.createElement("span");
  stamp.className = "stamp";
  stamp.textContent = timeNow();
  head.append(tag, stamp);
  const p = document.createElement("p");
  p.textContent = text;
  turn.append(head, p);
  chat.appendChild(turn);
  chat.scrollTop = chat.scrollHeight;
}

function addLine(kind, text) {
  const div = document.createElement("div");
  div.className = "line " + kind;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function showError(text) {
  toastText.textContent = text;
  toast.hidden = false;
}

function hideError() {
  toast.hidden = true;
}

document.getElementById("toast-close").addEventListener("click", hideError);

function handleEvent(ev) {
  switch (ev.event) {
    case "hello":
    case "status":
      setStatus(ev.state);
      setRunning(ev.state !== "idle");
      break;
    case "heard":
      addTurn("user", ev.text);
      break;
    case "reply":
      addTurn("companion", ev.text);
      break;
    case "system":
      addLine("system", ev.text);
      break;
    case "warning":
      addLine("warning", "⚠ " + ev.text);
      break;
    case "error":
      showError(ev.text);
      break;
    case "session_ended":
      setRunning(false);
      setStatus("idle");
      loadMemory();
      break;
  }
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    conn.textContent = "";
    // The server replays the session history right after hello; start from
    // a clean slate so a reconnect doesn't duplicate turns.
    chat.innerHTML = "";
    hideError();
  };
  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onclose = () => {
    conn.textContent = "reconnecting…";
    setTimeout(connect, 1500);
  };
}

orb.addEventListener("click", () => {
  hideError();
  if (running) {
    ws.send(JSON.stringify({ cmd: "stop" }));
  } else {
    // Don't lock optimistically: a rejected start (already running, unknown
    // option) emits only an error event, never session_ended. The UI locks
    // when the first non-idle status event arrives.
    ws.send(JSON.stringify({
      cmd: "start",
      brain: brainSel.value,
      ears: earsSel.value,
      ptt: pttBox.checked,
    }));
  }
});

/* ---------- memory drawer ---------- */

function setDrawer(open) {
  drawer.classList.toggle("open", open);
  scrim.hidden = !open;
  memoryBtn.setAttribute("aria-expanded", String(open));
}

memoryBtn.addEventListener("click", () => setDrawer(!drawer.classList.contains("open")));
document.getElementById("drawer-close").addEventListener("click", () => setDrawer(false));
scrim.addEventListener("click", () => setDrawer(false));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") setDrawer(false);
});

// Minimal markdown for the memory files: ## headings and "- " bullets only.
function renderMarkdown(text) {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let html = "";
  let inList = false;
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    if (t.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + esc(t.slice(2)) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (t.startsWith("## ")) html += "<h3>" + esc(t.slice(3)) + "</h3>";
      else if (t.startsWith("# ")) html += "<h3>" + esc(t.slice(2)) + "</h3>";
      else if (t) html += "<p>" + esc(t) + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function showTab(name) {
  activeTab = name;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === name);
  }
  const text = memory[name] || "";
  memoryContent.innerHTML = text
    ? renderMarkdown(text)
    : "<p class=\"empty\">Nothing here yet — it fills in after your first conversation.</p>";
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
}

async function loadMemory() {
  const res = await fetch("/api/memory");
  memory = await res.json();
  showTab(activeTab);
}

// Remember the last-used session settings across visits. localStorage can
// throw (private mode, disabled storage); settings are a nicety, so fall
// back to the server defaults silently.
const SETTINGS_KEY = "iris.settings";

function loadSavedSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
  } catch {
    return {};
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      brain: brainSel.value,
      ears: earsSel.value,
      ptt: pttBox.checked,
    }));
  } catch {
    // Storage unavailable; the session still works, settings just won't stick.
  }
}

for (const el of [brainSel, earsSel, pttBox]) {
  el.addEventListener("change", saveSettings);
}

async function loadOptions() {
  const res = await fetch("/api/options");
  const opts = await res.json();
  for (const name of opts.brains) brainSel.add(new Option(name, name));
  for (const name of opts.ears) earsSel.add(new Option(name, name));
  // Saved settings win over server defaults, but only if the option still
  // exists — a stale value from an older version must not select nothing.
  const saved = loadSavedSettings();
  brainSel.value = opts.brains.includes(saved.brain) ? saved.brain : opts.defaults.brain;
  earsSel.value = opts.ears.includes(saved.ears) ? saved.ears : opts.defaults.ears;
  pttBox.checked = typeof saved.ptt === "boolean" ? saved.ptt : opts.defaults.ptt;
}

loadOptions();
loadMemory();
connect();
