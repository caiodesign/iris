# tests/test_server.py
import threading
import time

from fastapi.testclient import TestClient

from companion import config, server


def test_snapshot_and_register_is_atomic_with_emit():
    # emit() runs on the session thread while pages connect on the event
    # loop; snapshot+register must happen under one lock so the history
    # snapshot never races a concurrent append (no RuntimeError, snapshot
    # always ordered and duplicate-free under contention). The no-duplicate
    # -delivery guarantee itself is covered by
    # test_emit_captures_recipients_at_append_time.
    manager = server.SessionManager()  # loop stays None: no broadcasts
    fake_ws = object()
    stop = threading.Event()

    def hammer():
        i = 0
        while not stop.is_set():
            manager.emit({"event": "system", "text": str(i)})
            i += 1

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    try:
        for _ in range(300):
            snapshot = manager.snapshot_and_register(fake_ws)
            assert fake_ws in manager.sockets
            nums = [int(e["text"]) for e in snapshot]
            assert nums == sorted(set(nums))  # ordered, no duplicates
            manager.sockets.discard(fake_ws)
    finally:
        stop.set()
        t.join(timeout=2)


def test_emit_captures_recipients_at_append_time(monkeypatch):
    # The duplicate-delivery guard: the recipient list for a broadcast must
    # be captured in the same locked section as the history append. An event
    # emitted before snapshot_and_register(ws) is snapshot-only (ws not a
    # recipient); one emitted after is live-only (ws is a recipient) — an
    # event can never reach the same socket both ways, even though the
    # scheduled broadcast coroutine runs later on the loop.
    manager = server.SessionManager()
    manager.loop = object()  # non-None so emit schedules; scheduling stubbed
    calls = []

    def fake_broadcast(event, recipients):
        calls.append((event, list(recipients)))

    monkeypatch.setattr(manager, "_broadcast", fake_broadcast)
    monkeypatch.setattr(
        server.asyncio, "run_coroutine_threadsafe", lambda coro, loop: None
    )

    ws = object()
    manager.emit({"event": "system", "text": "before"})
    manager.snapshot_and_register(ws)
    manager.emit({"event": "system", "text": "after"})

    (event1, recipients1), (event2, recipients2) = calls
    assert event1["text"] == "before" and ws not in recipients1
    assert event2["text"] == "after" and ws in recipients2


def fake_run_session(brain, ears, ptt, emit, should_stop):
    emit({"event": "system", "text": f"fake session {brain}/{ears}/{ptt}"})
    for _ in range(500):
        if should_stop():
            return True
        time.sleep(0.01)
    return True


def fresh_client():
    # Each test gets its own manager so threads/history never leak between
    # tests. TestClient used as a context manager runs the lifespan, which
    # captures the event loop for cross-thread broadcasting.
    server.manager = server.SessionManager()
    return TestClient(server.app)


def test_serves_the_page():
    with fresh_client() as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "Companion" in res.text


def test_memory_endpoint_returns_both_files(tmp_path, monkeypatch):
    (tmp_path / "durable.md").write_text("## Facts\n- Likes ramen.\n", encoding="utf-8")
    (tmp_path / "timeline.md").write_text("## 2026-07-08\n- Chatted.\n", encoding="utf-8")
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    with fresh_client() as client:
        data = client.get("/api/memory").json()
    assert "Likes ramen" in data["durable"]
    assert "Chatted" in data["timeline"]


def test_memory_endpoint_tolerates_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path / "nope"))
    with fresh_client() as client:
        data = client.get("/api/memory").json()
    assert data == {"durable": "", "timeline": ""}


def test_options_endpoint_lists_choices_and_defaults():
    with fresh_client() as client:
        data = client.get("/api/options").json()
    assert data["brains"] == ["local", "claude", "openai", "zai"]
    assert data["ears"] == ["local", "openai"]
    assert data["defaults"] == {
        "brain": config.LLM_PROVIDER,
        "ears": config.STT_PROVIDER,
        "ptt": False,
    }


def test_websocket_start_and_stop_cycle(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello == {"event": "hello", "running": False, "state": "idle"}
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            assert ws.receive_json() == {
                "event": "system",
                "text": "fake session local/local/False",
            }
            ws.send_json({"cmd": "stop"})
            assert ws.receive_json() == {"event": "status", "state": "idle"}
            assert ws.receive_json() == {"event": "session_ended"}


def test_second_start_is_rejected_while_running(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            ws.receive_json()  # fake session line
            ws.send_json({"cmd": "start", "brain": "claude", "ears": "local", "ptt": False})
            assert ws.receive_json() == {
                "event": "error",
                "text": "A session is already running.",
            }
            ws.send_json({"cmd": "stop"})
            ws.receive_json()  # status idle
            ws.receive_json()  # session_ended


def test_start_with_unknown_option_is_rejected():
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "skynet", "ears": "local", "ptt": False})
            event = ws.receive_json()
    assert event["event"] == "error"
    assert "skynet" in event["text"]


def test_reconnect_replays_history(monkeypatch):
    monkeypatch.setattr(server.session, "run_session", fake_run_session)
    with fresh_client() as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"cmd": "start", "brain": "local", "ears": "local", "ptt": False})
            ws.receive_json()  # fake session line
        # Simulated page refresh: hello reports running, history replays.
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["running"] is True
            assert ws.receive_json() == {
                "event": "system",
                "text": "fake session local/local/False",
            }
            ws.send_json({"cmd": "stop"})
            ws.receive_json()  # status idle
            ws.receive_json()  # session_ended
