# Web Interface — Design

**Date:** 2026-07-08
**Status:** Approved by Caio (pending spec review)

## Problem

All interaction with the companion today happens in a terminal: startup
questions (brain, ears, push-to-talk) are typed answers to prompts, and the
conversation is a stream of same-color `Heard:` / `Companion:` lines that is
hard to read. There is no way to see what the companion remembers without
opening the memory files by hand.

## Goal

A dark, Claude.ai-style web page served by the app itself that:

- lets the user pick session options and start/end sessions with the mouse,
- shows the conversation as a readable chat with live status,
- displays the memory files in a sidebar panel.

Interaction with the companion itself stays **voice only** — the page has no
text input for messages. Memory is **view only** in the browser.

## Decisions made during brainstorming

| Question | Decision |
| --- | --- |
| Typed chat input? | No — voice only; the page displays the session. |
| Where are options chosen? | In the browser (dropdowns + Start button). |
| Memory panel | View only, auto-refresh after each session. |
| Theme | Dark only. |
| Stack | FastAPI + one static HTML/CSS/JS page. No Node.js, no build step. |

Alternatives rejected: React frontend (adds a second toolchain for a
one-page app), Streamlit/Gradio (generic look, poor fit for live events).

## Architecture

```
browser (companion/web/index.html + app.js + style.css)
   │  WebSocket (events + commands) and GET /api/memory
   ▼
companion/server.py    FastAPI app, session thread management
   │  emit(event) callback / stop flag
   ▼
companion/session.py   extracted listen→transcribe→think→speak loop
   │
   ├─ VoiceDetector / PushToTalkRecorder (capture, + stop_check)
   ├─ Transcriber, LLMClient, Speaker, Memory, StateMachine (unchanged)
```

- **`companion/session.py` (new):** the session engine, extracted from
  `main.py`. Owns preflight checks, model loading, the conversation loop,
  and end-of-session memory writing. Reports everything through an
  `emit(event)` callback instead of `print()`. Accepts a stop flag checked
  between utterances and inside the capture wait loops.
- **`companion/server.py` (new):** FastAPI app. Serves the static page,
  exposes the WebSocket and the memory endpoint, and runs at most one
  session at a time in a background thread.
- **`companion/web/` (new):** `index.html`, `style.css`, `app.js` — plain
  static files, self-contained (no CDN dependencies).
- **`companion/main.py` (changed):** keeps its CLI flags and terminal
  prompts but delegates the loop to `session.py` with a print-based
  emitter. Terminal mode must behave as it does today.
- **Entry point:** `python -m companion.server` starts uvicorn on
  `http://localhost:8000` (localhost only), prints the link, and opens the
  default browser.

New dependencies: `fastapi`, `uvicorn`.

## Event protocol (WebSocket, JSON)

Browser → server:

- `{"cmd": "start", "brain": "...", "ears": "...", "ptt": true|false}`
- `{"cmd": "stop"}` — end the session; runs the same goodbye + memory path
  as saying the stop phrase.

Server → browser:

- `{"event": "status", "state": "idle" | "loading" | "listening" |
  "thinking" | "speaking"}`
- `{"event": "heard", "text": "..."}` — a user utterance the state machine
  forwarded (wake/sleep/cancel show as `system` lines instead).
- `{"event": "reply", "text": "..."}`
- `{"event": "system", "text": "..."}` — session started, woke up, went to
  sleep, discarded that, remembering session, etc.
- `{"event": "warning", "text": "..."}` — transcription failed, brain
  failed to reply, TTS failed; session continues.
- `{"event": "error", "text": "..."}` — a preflight check failed; session
  did not start (or died). Includes the same actionable hints the terminal
  prints today (Ollama not running, missing key, no mic, missing Kokoro
  files).
- `{"event": "session_ended"}` — browser unlocks the options and refreshes
  the memory panel.

On (re)connect, the server sends the current status and whether a session
is running, so a refreshed page shows the correct state. `GET /api/memory`
returns the contents of `memory/durable.md` and `memory/timeline.md`.

## UI layout

Dark theme, two columns:

- **Sidebar (left):**
  - Session options: brain dropdown (local / claude / openai / zai), ears
    dropdown (local / openai), push-to-talk toggle, defaults from
    `config.py`. One primary button: **Start** → becomes **End session**
    while running; options are disabled while a session runs.
  - Memory panel: two tabs, "About you" (durable.md) and "Timeline"
    (timeline.md), rendered with a minimal built-in markdown formatter
    (headings + bullet lists — all the memory files use). Refreshes on
    `session_ended`.
- **Main area:** chat transcript. User bubbles right, companion bubbles
  left, timestamps, auto-scroll to newest. Status pill at the top mirrors
  the `status` events (idle / loading models / listening / thinking /
  speaking). `system` lines render as small centered gray text; `warning`
  as amber lines; `error` as a red banner with the hint text.
- Connection loss shows a "reconnecting…" notice; the page retries the
  WebSocket automatically.

## Session lifecycle and stopping

- Start: server rejects a `start` while a session is running. Preflight
  failures emit `error` and return to idle without killing the server.
- Voice stop ("bye bye") works exactly as today; the End button triggers
  the same path: speak the goodbye, write timeline + durable memory, emit
  `session_ended`.
- Capture interruption: `VoiceDetector.listen_for_utterance` and
  `PushToTalkRecorder` gain an optional `stop_check` callable evaluated
  each frame/wait tick so End takes effect within a moment even while the
  app is waiting for speech. On stop-by-button mid-listen, the partial
  audio is discarded.
- Ctrl+C in the terminal stops the server and any running session
  (best-effort memory write, same as today's finally block).

## Error handling summary

| Failure | Behavior |
| --- | --- |
| Preflight (Ollama/key/mic/TTS files) | `error` event, red banner, back to idle; user fixes and clicks Start again. |
| Transcription/LLM/TTS mid-session | `warning` event, session continues (same recovery speech as today). |
| Memory write at session end | `warning` event, session still ends cleanly. |
| WebSocket drop | Session keeps running; events since disconnect are buffered (bounded) and replayed on reconnect. |

## Testing

- `tests/test_session.py`: drive the engine with fake capture, transcriber,
  LLM, speaker, memory; assert emitted event sequences for wake, forward,
  cancel, sleep, warning, and stop-flag paths.
- `tests/test_server.py`: FastAPI TestClient — static page served,
  `/api/memory` returns file contents, WebSocket start/stop commands and
  reject-second-session behavior, reconnect state message.
- Existing tests must keep passing; `test_main.py` updated for the
  refactored `main.py`.
- Frontend: manual smoke test (start session, talk, watch chat/status,
  end session, memory refresh).

## Out of scope

- Typed chat messages, memory editing, light theme, multi-user access,
  serving beyond localhost, mobile layout, streaming partial LLM replies.
