// companion/web/app.js
const chat = document.getElementById("chat");
const pill = document.getElementById("status-pill");
const conn = document.getElementById("conn");
const banner = document.getElementById("error-banner");
const brainSel = document.getElementById("brain");
const earsSel = document.getElementById("ears");
const pttBox = document.getElementById("ptt");
const btn = document.getElementById("session-btn");
const memoryContent = document.getElementById("memory-content");

let ws = null;
let running = false;
let memory = { durable: "", timeline: "" };
let activeTab = "durable";

const STATUS_LABELS = {
  idle: "Idle",
  loading: "Loading models…",
  listening: "Listening 🎤",
  thinking: "Thinking…",
  speaking: "Speaking 🔊",
};

function setStatus(state) {
  pill.textContent = STATUS_LABELS[state] || state;
  pill.className = "pill " + state;
}

function setRunning(isRunning) {
  running = isRunning;
  btn.textContent = isRunning ? "End session" : "Start";
  btn.classList.toggle("running", isRunning);
  for (const el of [brainSel, earsSel, pttBox]) el.disabled = isRunning;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addBubble(role, text) {
  const row = document.createElement("div");
  row.className = "row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const stamp = document.createElement("div");
  stamp.className = "stamp";
  stamp.textContent = timeNow();
  bubble.appendChild(stamp);
  row.appendChild(bubble);
  chat.appendChild(row);
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
  banner.textContent = text;
  banner.classList.remove("hidden");
}

function handleEvent(ev) {
  switch (ev.event) {
    case "hello":
      setStatus(ev.state);
      setRunning(ev.state !== "idle");
      break;
    case "status":
      setStatus(ev.state);
      setRunning(ev.state !== "idle");
      break;
    case "heard":
      addBubble("user", ev.text);
      break;
    case "reply":
      addBubble("companion", ev.text);
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
    // a clean slate so a reconnect doesn't duplicate bubbles.
    chat.innerHTML = "";
    banner.classList.add("hidden");
  };
  ws.onmessage = (msg) => handleEvent(JSON.parse(msg.data));
  ws.onclose = () => {
    conn.textContent = "reconnecting…";
    setTimeout(connect, 1500);
  };
}

btn.addEventListener("click", () => {
  banner.classList.add("hidden");
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
    : "<p class=\"empty\">Nothing here yet.</p>";
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
}

async function loadMemory() {
  const res = await fetch("/api/memory");
  memory = await res.json();
  showTab(activeTab);
}

async function loadOptions() {
  const res = await fetch("/api/options");
  const opts = await res.json();
  for (const name of opts.brains) brainSel.add(new Option(name, name));
  for (const name of opts.ears) earsSel.add(new Option(name, name));
  brainSel.value = opts.defaults.brain;
  earsSel.value = opts.defaults.ears;
  pttBox.checked = opts.defaults.ptt;
}

loadOptions();
loadMemory();
connect();
