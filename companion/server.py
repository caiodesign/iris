# companion/server.py
"""Web front-end: serves the static page, fans session events out over a
WebSocket, and runs at most one voice session in a background thread. Run
with: python -m companion.server"""
import asyncio
import os
import threading
import webbrowser
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from companion import config, session
from companion.memory import Memory

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://localhost:{PORT}"


class SessionManager:
    """Owns the single session thread and the fan-out to websockets.

    emit() is called from the session thread; it hops onto the server's event
    loop via run_coroutine_threadsafe, so sockets are only touched from the
    loop. history is bounded and replayed to (re)connecting pages so a
    refresh shows the whole conversation so far."""

    def __init__(self):
        self.loop = None  # captured by the lifespan at startup
        self.sockets = set()
        self.history = deque(maxlen=1000)
        self.state = "idle"
        self.thread = None
        self.stop_event = threading.Event()
        # Guards history: the session thread appends via emit() while
        # connecting pages snapshot it. snapshot+register are atomic so an
        # event is either in the snapshot or broadcast live, never both.
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, brain, ears, ptt) -> None:
        if self.running:
            self.emit({"event": "error", "text": "A session is already running."})
            return
        if brain not in session.PROVIDER_NAMES or ears not in session.STT_NAMES:
            self.emit({"event": "error", "text": f"Unknown option: {brain}/{ears}."})
            return
        with self._lock:
            self.history.clear()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, args=(brain, ears, bool(ptt)), daemon=True
        )
        self.thread.start()

    def _run(self, brain, ears, ptt) -> None:
        session.run_session(brain, ears, ptt, self.emit, self.stop_event.is_set)
        # Always emitted, even when run_session failed preflight — this is
        # what unlocks the options panel in the browser.
        self.emit({"event": "status", "state": "idle"})
        self.emit({"event": "session_ended"})

    def stop(self) -> None:
        self.stop_event.set()

    def emit(self, event) -> None:
        if event["event"] == "status":
            self.state = event["state"]
        with self._lock:
            self.history.append(event)
            # Recipients are captured with the append: a socket registered
            # after this point gets the event from its snapshot instead, so
            # it can never receive the same event twice.
            recipients = list(self.sockets)
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(event, recipients), self.loop
            )

    def snapshot_and_register(self, ws) -> list:
        """Atomically copy the history and register a socket for live events.

        Done under the same lock as emit()'s append+recipient capture, so a
        concurrent event lands either in the returned snapshot or in a
        broadcast to this socket — never both (duplicate) and never
        mid-iteration (RuntimeError)."""
        with self._lock:
            snapshot = list(self.history)
            self.sockets.add(ws)
            return snapshot

    async def _broadcast(self, event, recipients) -> None:
        for ws in recipients:
            try:
                await ws.send_json(event)
            except Exception:
                # A socket that died mid-send; drop it, the page reconnects.
                self.sockets.discard(ws)


manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/options")
def get_options():
    return {
        "brains": session.PROVIDER_NAMES,
        "ears": session.STT_NAMES,
        "defaults": {
            "brain": config.LLM_PROVIDER,
            "ears": config.STT_PROVIDER,
            "ptt": False,
        },
    }


@app.get("/api/memory")
def get_memory():
    memory = Memory(config.MEMORY_DIR, config.TIMELINE_MAX_CHARS)
    timeline = ""
    if os.path.exists(memory.timeline_path):
        with open(memory.timeline_path, "r", encoding="utf-8") as f:
            timeline = f.read().strip()
    return {"durable": memory.load_durable(), "timeline": timeline}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    snapshot = manager.snapshot_and_register(ws)
    try:
        await ws.send_json(
            {"event": "hello", "running": manager.running, "state": manager.state}
        )
        for event in snapshot:
            await ws.send_json(event)
        while True:
            msg = await ws.receive_json()
            if msg.get("cmd") == "start":
                manager.start(msg.get("brain"), msg.get("ears"), msg.get("ptt"))
            elif msg.get("cmd") == "stop":
                manager.stop()
    except WebSocketDisconnect:
        pass
    finally:
        manager.sockets.discard(ws)


# Mounted last so /api and /ws win the route match.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    load_dotenv()
    print(f"Companion web interface: {URL}")
    # uvicorn.run blocks; open the browser shortly after it comes up.
    threading.Timer(1.0, webbrowser.open, args=[URL]).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
