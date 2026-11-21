"""Offline tests for the paid AI routes: rate limiting, /api/complete limits,
and the ElevenLabs proxies. Nothing here talks to Anthropic, ElevenLabs or
Google: every outbound call is stubbed.
"""

import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

# Dummy keys must be in place before main is imported.
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
os.environ["ELEVENLABS_API_KEY"] = "test-elevenlabs-key"
os.environ["ELEVENLABS_AGENT_ID"] = "test-agent"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

VALID_VOICE = "21m00Tcm4TlvDq8ikWAM"


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Fresh limiter state per test, and fail loudly on any unstubbed network call."""
    main._rate_buckets.clear()
    main._global_ai_calls.clear()
    monkeypatch.setattr(main, "_last_bucket_sweep", 0.0)
    monkeypatch.setattr(main, "_RATE_LIMIT_PER_MINUTE", 60)
    monkeypatch.setattr(main, "_GLOBAL_AI_CALLS_PER_MINUTE", 240)
    monkeypatch.setattr(main, "_TRUSTED_PROXY_HOPS", 1)

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(main.engine.client.messages, "create", no_network)
    yield
    main._rate_buckets.clear()
    main._global_ai_calls.clear()


@pytest.fixture
def client():
    # TrustedHostMiddleware only accepts known hosts, so use localhost.
    with TestClient(main.app, base_url="http://localhost") as c:
        yield c


@pytest.fixture
def claude(monkeypatch):
    """Stub for engine.client.messages.create that records each call."""
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="stub reply")])

    monkeypatch.setattr(main.engine.client.messages, "create", fake_create)
    return calls


class FakeUpstream:
    def __init__(self, status_code=200, body=b"", json_body=None):
        self.status_code = status_code
        self._body = body
        self._json = json_body
        self.text = body.decode("utf-8", "replace")
        self.closed = False

    def iter_content(self, chunk_size=4096):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def json(self):
        return self._json

    def close(self):
        self.closed = True


@pytest.fixture
def eleven_post(monkeypatch):
    """Stub requests.post for ElevenLabs TTS; set .response to change the reply."""
    state = SimpleNamespace(calls=[], response=FakeUpstream(200, b"ID3fake-mp3"))

    def fake_post(url, **kwargs):
        state.calls.append(SimpleNamespace(url=url, **kwargs))
        if isinstance(state.response, Exception):
            raise state.response
        return state.response

    monkeypatch.setattr(requests, "post", fake_post)
    return state


def complete(client, body, ip="203.0.113.7", spoof=None):
    forwarded = f"{spoof}, {ip}" if spoof else ip
    return client.post("/api/complete", json=body, headers={"X-Forwarded-For": forwarded})


# ── Rate limiting ─────────────────────────────────────────────────────────────
def test_rotating_leftmost_forwarded_for_is_still_limited(client, claude, monkeypatch):
    monkeypatch.setattr(main, "_RATE_LIMIT_PER_MINUTE", 3)
    for i in range(3):
        r = complete(client, {"prompt": "hi"}, spoof=f"10.0.0.{i}")
        assert r.status_code == 200
    r = complete(client, {"prompt": "hi"}, spoof="10.0.0.99")
    assert r.status_code == 429
    assert r.json() == {"error": "rate_limit_exceeded", "retry_after_seconds": 60}
    assert r.headers["Retry-After"] == "60"
    assert len(claude) == 3
    # The spoofed values never became buckets; only the proxy-added address did.
    assert list(main._rate_buckets) == ["203.0.113.7"]
    # A different real caller is unaffected.
    assert complete(client, {"prompt": "hi"}, ip="198.51.100.5").status_code == 200


def test_trusted_proxy_hops_picks_entry_from_the_right(client, claude, monkeypatch):
    monkeypatch.setattr(main, "_TRUSTED_PROXY_HOPS", 2)
    r = client.post(
        "/api/complete",
        json={"prompt": "hi"},
        headers={"X-Forwarded-For": "1.1.1.1, 198.51.100.1, 10.1.1.1"},
    )
    assert r.status_code == 200
    assert list(main._rate_buckets) == ["198.51.100.1"]


def test_no_forwarded_for_falls_back_to_peer_address(client, claude):
    assert client.post("/api/complete", json={"prompt": "hi"}).status_code == 200
    assert list(main._rate_buckets) == ["testclient"]


def test_progress_generate_is_metered(client, monkeypatch):
    monkeypatch.setattr(main, "_RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(main.engine.store, "list_recent", lambda limit=5: [])
    headers = {"X-Forwarded-For": "203.0.113.9"}
    assert client.post("/progress/generate", json={}, headers=headers).status_code == 200
    assert client.post("/progress/generate", json={}, headers=headers).status_code == 200
    r = client.post("/progress/generate", json={}, headers=headers)
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limit_exceeded"


def test_auth_google_is_metered_and_hides_error_text(client, monkeypatch):
    from google.oauth2 import id_token

    def bad_token(*args, **kwargs):
        raise ValueError("Wrong number of segments in token: secret-token-bits")

    monkeypatch.setattr(id_token, "verify_oauth2_token", bad_token)
    monkeypatch.setattr(main, "_RATE_LIMIT_PER_MINUTE", 1)
    headers = {"X-Forwarded-For": "203.0.113.10"}
    r = client.post("/auth/google", json={"credential": "x"}, headers=headers)
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid Google token"}
    assert "secret-token-bits" not in r.text
    assert client.post("/auth/google", json={"credential": "x"}, headers=headers).status_code == 429


def test_global_ceiling_applies_across_callers(client, claude, monkeypatch):
    monkeypatch.setattr(main, "_GLOBAL_AI_CALLS_PER_MINUTE", 3)
    for i in range(3):
        assert complete(client, {"prompt": "hi"}, ip=f"198.51.100.{i}").status_code == 200
    r = complete(client, {"prompt": "hi"}, ip="198.51.100.200")
    assert r.status_code == 429
    assert r.json() == {"error": "rate_limit_exceeded", "retry_after_seconds": 60}
    assert len(claude) == 3


def test_idle_buckets_are_evicted(client, claude):
    stale = time.time() - 120
    for i in range(500):
        main._rate_buckets[f"10.9.{i // 256}.{i % 256}"] = deque([stale])
    main._rate_buckets["10.8.0.1"] = deque()
    assert complete(client, {"prompt": "hi"}, ip="203.0.113.50").status_code == 200
    assert list(main._rate_buckets) == ["203.0.113.50"]


# ── /api/complete ─────────────────────────────────────────────────────────────
def test_opus_and_unknown_models_fall_back_to_default(client, claude):
    assert complete(client, {"prompt": "hi", "model": "claude-opus-4-7"}).status_code == 200
    assert complete(client, {"prompt": "hi", "model": "made-up-model"}).status_code == 200
    assert [c["model"] for c in claude] == [main.MODEL, main.MODEL]
    assert main.MODEL == "claude-sonnet-4-6"


def test_typical_prototype_call_succeeds(client, claude):
    messages = [
        {"role": "user", "content": "I need help preparing for a board update."},
        {"role": "assistant", "content": "What outcome do you want from the meeting?"},
        {"role": "user", "content": "Approval for the Q3 plan."},
    ]
    r = complete(client, {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4500,
        "system": "You are an executive coach.",
        "messages": messages,
    })
    assert r.status_code == 200
    assert r.json() == {"text": "stub reply"}
    sent = claude[0]
    assert sent["model"] == "claude-haiku-4-5-20251001"
    assert sent["max_tokens"] == 4500
    assert sent["system"] == "You are an executive coach."
    assert sent["messages"] == messages


def test_string_prompt_and_long_system_prompt(client, claude):
    r = complete(client, {"prompt": "hello", "system": "x" * 4000})
    assert r.status_code == 200
    sent = claude[0]
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["max_tokens"] == 2048
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_extra_message_fields_are_not_forwarded(client, claude):
    r = complete(client, {"messages": [{"role": "user", "content": "hi", "id": "abc"}]})
    assert r.status_code == 200
    assert claude[0]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("body", [
    {"prompt": "x" * 100_001},
    {"prompt": "hi", "system": "x" * 50_001},
    {"messages": [{"role": "user", "content": "hi"}] * 101},
    {"messages": [{"role": "user", "content": "x" * 50_001}]},
    {"messages": [{"role": "user", "content": "x" * 40_000}] * 4},
    {"prompt": "hi", "max_tokens": 5001},
    {"prompt": "hi", "max_tokens": 0},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    {"messages": [{"role": "user", "content": 42}]},
    {"messages": [{"role": "system", "content": "hi"}]},
    {"messages": [{"role": "user"}]},
], ids=[
    "prompt-too-long", "system-too-long", "too-many-messages", "message-too-long",
    "messages-total-too-long", "max-tokens-too-high", "max-tokens-zero",
    "content-blocks", "content-int", "bad-role", "missing-content",
])
def test_oversized_or_malformed_requests_are_rejected(client, claude, body):
    r = complete(client, body)
    assert r.status_code in (400, 422)
    assert claude == []


def test_missing_prompt_and_messages_is_400(client, claude):
    assert complete(client, {}).status_code == 400
    assert claude == []


def test_complete_errors_do_not_leak_details(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("upstream said: invalid x-api-key sk-ant-secret")

    monkeypatch.setattr(main.engine.client.messages, "create", boom)
    r = complete(client, {"prompt": "hi"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Claude request failed"}
    assert "sk-ant-secret" not in r.text


# ── ElevenLabs ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("voice_id", [
    "../../v1/sound-generation",
    "../../v1/user#",
    "../../v1/voices/add?x=1#",
    "abc",
    VALID_VOICE + "\n",
    "a" * 33,
])
def test_tts_rejects_bad_voice_id_without_upstream_call(client, eleven_post, voice_id):
    r = client.post("/elevenlabs/tts", json={"text": "hello", "voice_id": voice_id})
    assert r.status_code == 400
    assert eleven_post.calls == []


def test_tts_valid_voice_builds_expected_url(client, eleven_post):
    r = client.post("/elevenlabs/tts", json={"text": "hello", "voice_id": VALID_VOICE})
    assert r.status_code == 200
    assert r.content == b"ID3fake-mp3"
    assert r.headers["content-type"].startswith("audio/mpeg")
    call = eleven_post.calls[0]
    assert call.url == f"https://api.elevenlabs.io/v1/text-to-speech/{VALID_VOICE}"
    assert call.json["model_id"] == "eleven_turbo_v2_5"


def test_tts_default_voice_when_none(client, eleven_post):
    assert client.post("/elevenlabs/tts", json={"text": "hello"}).status_code == 200
    assert eleven_post.calls[0].url.endswith(f"/text-to-speech/{VALID_VOICE}")


def test_tts_unapproved_model_falls_back(client, eleven_post):
    client.post("/elevenlabs/tts", json={"text": "hi", "model_id": "eleven_multilingual_v2"})
    client.post("/elevenlabs/tts", json={"text": "hi", "model_id": "eleven_flash_v2_5"})
    assert [c.json["model_id"] for c in eleven_post.calls] == ["eleven_turbo_v2_5", "eleven_flash_v2_5"]


def test_tts_text_is_still_capped(client, eleven_post):
    client.post("/elevenlabs/tts", json={"text": "a" * 5000})
    assert len(eleven_post.calls[0].json["text"]) == 1500


def test_tts_upstream_error_is_generic_502(client, eleven_post):
    upstream = FakeUpstream(401, b'{"detail": "secret upstream account detail"}')
    eleven_post.response = upstream
    r = client.post("/elevenlabs/tts", json={"text": "hello"})
    assert r.status_code == 502
    assert r.json() == {"detail": "Voice service is unavailable"}
    assert "secret upstream" not in r.text
    assert upstream.closed


def test_tts_connection_error_is_generic_502(client, eleven_post):
    eleven_post.response = requests.ConnectionError("proxy host internal-123 refused")
    r = client.post("/elevenlabs/tts", json={"text": "hello"})
    assert r.status_code == 502
    assert "internal-123" not in r.text


def test_signed_url_rejects_bad_voice_id(client):
    # requests.get is the no-network stub here, so any upstream call would fail the test.
    r = client.post("/elevenlabs/signed-url", json={
        "scenario": {"title": "t"}, "voice_id": "../../v1/user#",
    })
    assert r.status_code == 400


def test_signed_url_valid_voice_and_generic_errors(client, monkeypatch):
    replies = [
        FakeUpstream(200, json_body={"signed_url": "wss://example.invalid/signed"}),
        FakeUpstream(500, b"secret upstream failure text"),
    ]
    monkeypatch.setattr(requests, "get", lambda *a, **k: replies.pop(0))
    body = {"scenario": {"title": "t", "character": {"name": "Alex"}}, "voice_id": VALID_VOICE}

    ok = client.post("/elevenlabs/signed-url", json=body)
    assert ok.status_code == 200
    assert ok.json()["overrides"]["tts"] == {"voice_id": VALID_VOICE}

    bad = client.post("/elevenlabs/signed-url", json=body)
    assert bad.status_code == 502
    assert bad.json() == {"detail": "Voice service is unavailable"}
    assert "secret upstream" not in bad.text
