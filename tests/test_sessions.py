"""
Session ownership and session-ID safety tests. Fully offline: the Anthropic
client is replaced by a stub, and sessions are written to a temp folder.
"""

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

# Must be set before main / coaching_engine are imported.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")
os.environ.setdefault("SESSIONS_DIR", tempfile.mkdtemp(prefix="stratos-sessions-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import coaching_engine  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ── Stub Claude client ───────────────────────────────────────────────────────
class _StubStream:
    def __init__(self, calls):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        yield "Stub "
        yield "reply."


class _StubMessages:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        return _StubStream(self.calls)

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        text = json.dumps({
            "score": 80, "headline": "ok", "summary": "ok",
            "annotations": {}, "leader_score": 70, "skills": {}, "growth_plan": "ok",
        })
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


class _StubClient:
    def __init__(self):
        self.messages = _StubMessages()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh sessions folder inside tmp_path, stub client, empty session cache."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(coaching_engine, "SESSIONS_DIR", sessions_dir)
    stub = _StubClient()
    monkeypatch.setattr(main.engine, "client", stub)
    monkeypatch.setattr(main.engine.store, "_cache", {})
    if hasattr(main, "_rate_buckets"):  # keep the per-IP limiter from tripping across tests
        main._rate_buckets.clear()
    client = TestClient(main.app, base_url="http://localhost")
    return SimpleNamespace(client=client, dir=sessions_dir, root=tmp_path, stub=stub)


def _new(client, **body):
    r = client.post("/sessions/new", json={"coach_name": "James", **body})
    assert r.status_code == 200
    data = r.json()
    assert data["session_token"] and data["session_id"]
    return data["session_id"], data["session_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _seed_roleplay(env, sid, turns=2):
    """Give a session enough user turns for /evaluate, written straight to the store."""
    s = main.engine.store.get(sid)
    s.is_roleplay = True
    s.roleplay_character = "Alex"
    for i in range(turns):
        s.add_message("user", f"user turn {i}")
        s.add_message("assistant", f"reply {i}")
    main.engine.store.save(s)


# ── Tests ────────────────────────────────────────────────────────────────────
def test_session_listing_route_is_gone(env):
    _new(env.client)
    r = env.client.get("/sessions")
    assert r.status_code in (404, 405)
    assert "sessions" not in (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})


def test_token_hash_stored_not_token(env):
    sid, token = _new(env.client)
    data = json.loads((env.dir / f"{sid}.json").read_text())
    assert token not in json.dumps(data)
    assert len(data["owner_token_hash"]) == 64


def test_owner_can_use_session_and_others_cannot(env):
    sid_a, tok_a = _new(env.client)
    _sid_b, tok_b = _new(env.client)
    _seed_roleplay(env, sid_a)

    for headers in ({}, _auth(tok_b), _auth("nonsense"), {"Authorization": tok_a}, {"Authorization": "Basic " + tok_a}):
        assert env.client.get(f"/sessions/{sid_a}", headers=headers).status_code == 404
        assert env.client.post(f"/sessions/{sid_a}/evaluate", headers=headers).status_code == 404
        assert env.client.post(f"/sessions/{sid_a}/annotate", headers=headers).status_code == 404
        r = env.client.post("/chat/stream", json={"session_id": sid_a, "message": "intruder"}, headers=headers)
        assert r.status_code == 404
        assert r.json() == {"detail": "Session not found"}
    assert "intruder" not in (env.dir / f"{sid_a}.json").read_text()

    r = env.client.get(f"/sessions/{sid_a}", headers=_auth(tok_a))
    assert r.status_code == 200 and r.json()["session_id"] == sid_a
    assert env.client.post(f"/sessions/{sid_a}/evaluate", headers=_auth(tok_a)).status_code == 200
    assert env.client.post(f"/sessions/{sid_a}/annotate", headers=_auth(tok_a)).status_code == 200


def test_chat_stream_writes_only_with_owner_token(env):
    sid, tok = _new(env.client)
    r = env.client.post("/chat/stream", json={"session_id": sid, "message": "hello from A"}, headers=_auth(tok))
    assert r.status_code == 200
    assert "Stub " in r.text and "[DONE]" in r.text
    saved = json.loads((env.dir / f"{sid}.json").read_text())
    assert {"role": "user", "content": "hello from A"} in saved["messages"]
    assert {"role": "assistant", "content": "Stub reply."} in saved["messages"]

    r = env.client.post("/chat/stream", json={"session_id": sid, "message": "wrong token"}, headers=_auth(tok + "x"))
    assert r.status_code == 404
    assert "wrong token" not in (env.dir / f"{sid}.json").read_text()


def test_chat_stream_error_is_generic(env, monkeypatch):
    sid, tok = _new(env.client)

    def boom(**kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(env.stub.messages, "stream", boom)
    r = env.client.post("/chat/stream", json={"session_id": sid, "message": "hi"}, headers=_auth(tok))
    assert r.status_code == 200
    assert "secret internal detail" not in r.text
    assert "Something went wrong" in r.text


def test_evaluate_error_is_generic(env, monkeypatch):
    sid, tok = _new(env.client)
    _seed_roleplay(env, sid)

    def boom(session):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main.engine, "generate_feedback", boom)
    r = env.client.post(f"/sessions/{sid}/evaluate", headers=_auth(tok))
    assert r.status_code == 500
    assert "secret internal detail" not in r.text


@pytest.mark.parametrize("bad_id", [
    "../../x",
    "../outside",
    "/etc/passwd",
    "ABSOLUTE",  # replaced with an absolute path inside the test
    "UPPER",     # replaced with an upper-case UUID
    "BRACED",    # replaced with a {uuid} form
    "NOHYPHEN",  # replaced with a hex-only UUID
    "not-a-uuid",
])
def test_bad_ids_are_404_and_never_touch_outside_files(env, bad_id):
    _sid, tok = _new(env.client)
    # A session-shaped file outside SESSIONS_DIR that traversal would hit.
    outside_id = str(uuid.uuid4())
    outside = env.root / "outside.json"
    outside_payload = {
        "session_id": outside_id, "messages": [{"role": "user", "content": "outside secret"}],
        "created_at": 0, "last_active": 0, "owner_token_hash": coaching_engine._hash_token(tok),
    }
    outside.write_text(json.dumps(outside_payload))
    (env.root / "x.json").write_text(json.dumps(outside_payload))
    before = {p: p.read_text() for p in env.root.glob("*.json")}

    real = str(uuid.uuid4())
    bad_id = {
        "ABSOLUTE": str(env.root / "outside"),
        "UPPER": real.upper(),
        "BRACED": "{" + real + "}",
        "NOHYPHEN": real.replace("-", ""),
    }.get(bad_id, bad_id)

    r = env.client.post("/chat/stream", json={"session_id": bad_id, "message": "probe"}, headers=_auth(tok))
    assert r.status_code == 404
    assert r.json() == {"detail": "Session not found"}
    assert coaching_engine.session_path(bad_id) is None
    assert main.engine.store.get(bad_id) is None

    # Path-style ids that survive URL routing as a single segment.
    if "/" not in bad_id:
        assert env.client.get(f"/sessions/{bad_id}", headers=_auth(tok)).status_code == 404
        assert env.client.post(f"/sessions/{bad_id}/annotate", headers=_auth(tok)).status_code == 404
        assert env.client.post(f"/sessions/{bad_id}/evaluate", headers=_auth(tok)).status_code == 404

    r = env.client.post("/progress/generate", json={"sessions": [{"session_id": bad_id, "session_token": tok}]})
    # Very long ids are refused by validation (422); the rest find nothing.
    assert r.status_code == 422 or r.json().get("error") == "No sessions found"

    # Nothing outside SESSIONS_DIR changed, and nothing new was copied in.
    assert {p: p.read_text() for p in env.root.glob("*.json")} == before
    assert not (env.dir / f"{outside_id}.json").exists()
    assert env.stub.messages.calls == []


def test_file_with_mismatched_internal_id_is_rejected(env):
    sid, tok = _new(env.client)
    other = str(uuid.uuid4())
    data = json.loads((env.dir / f"{sid}.json").read_text())
    data["session_id"] = other
    (env.dir / f"{sid}.json").write_text(json.dumps(data))
    main.engine.store._cache.clear()
    assert env.client.get(f"/sessions/{sid}", headers=_auth(tok)).status_code == 404


def test_legacy_session_without_owner_hash_is_inaccessible(env):
    sid = str(uuid.uuid4())
    (env.dir / f"{sid}.json").write_text(json.dumps({
        "session_id": sid, "coach_name": "James", "messages": [{"role": "user", "content": "old"}],
        "created_at": 0, "last_active": 0, "turn_count": 1,
    }))
    for headers in ({}, _auth(""), _auth("anything")):
        assert env.client.get(f"/sessions/{sid}", headers=headers).status_code == 404
        r = env.client.post("/chat/stream", json={"session_id": sid, "message": "hi"}, headers=headers)
        assert r.status_code == 404
    assert main.engine.store.get_owned(sid, "") is None
    r = env.client.post("/progress/generate", json={"sessions": [{"session_id": sid, "session_token": "anything"}]})
    assert r.json().get("error") == "No sessions found"
    assert env.stub.messages.calls == []


def test_progress_never_falls_back_to_recent_sessions(env):
    sid_a, tok_a = _new(env.client)
    env.client.post("/chat/stream", json={"session_id": sid_a, "message": "A private words"}, headers=_auth(tok_a))
    env.stub.messages.calls.clear()

    for body in ({}, {"sessions": []}, {"session_ids": [sid_a]},
                 {"sessions": [{"session_id": sid_a, "session_token": "wrong"}]}):
        r = env.client.post("/progress/generate", json=body)
        assert r.status_code == 200
        assert r.json().get("error") == "No sessions found"
    assert env.stub.messages.calls == []

    # With the right token, only that session's words are analyzed.
    r = env.client.post("/progress/generate", json={"sessions": [{"session_id": sid_a, "session_token": tok_a}]})
    assert r.status_code == 200 and "error" not in r.json()
    (kind, kwargs), = env.stub.messages.calls
    assert "A private words" in kwargs["messages"][0]["content"]


def test_progress_caps_number_of_sessions(env):
    sid, tok = _new(env.client)
    body = {"sessions": [{"session_id": sid, "session_token": tok}] * 11}
    assert env.client.post("/progress/generate", json=body).status_code == 422
