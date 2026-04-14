"""
Stratos Coaching Engine — Core Claude integration.
Manages conversation state, RAG injection, and streaming.
"""

import os
import uuid
import json
import time
from pathlib import Path
from typing import Optional, Iterator
from anthropic import Anthropic

from system_prompt import build_system_prompt, build_roleplay_prompt
from rag import get_rag_engine


# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_HISTORY_TURNS = 20        # Keep last 20 turns in context
SESSION_TTL_SECONDS = 3600   # 1 hour session expiry
SESSIONS_DIR = Path("sessions")


# ── Session Management ───────────────────────────────────────────────────────
class Session:
    def __init__(self, session_id: str, coach_name: str = "James"):
        self.session_id = session_id
        self.coach_name = coach_name
        self.session_name: str = ""
        self.messages: list[dict] = []
        self.created_at = time.time()
        self.last_active = time.time()
        self.turn_count = 0
        # Roleplay fields
        self.is_roleplay: bool = False
        self.roleplay_character: str = ""
        self.roleplay_situation: str = ""

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_active = time.time()
        if role == "user":
            self.turn_count += 1
            # Auto-name session from first user message
            if self.turn_count == 1 and not self.session_name:
                name = content.strip().replace("\n", " ")
                self.session_name = name[:50] + ("…" if len(name) > 50 else "")

    def get_trimmed_history(self) -> list[dict]:
        """Return last N turns, always keeping the first message for context."""
        if len(self.messages) <= MAX_HISTORY_TURNS * 2:
            return self.messages
        # Keep first exchange + most recent turns
        first_two = self.messages[:2]
        recent = self.messages[-(MAX_HISTORY_TURNS * 2 - 2):]
        return first_two + recent

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "coach_name": self.coach_name,
            "session_name": self.session_name,
            "messages": self.messages,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count,
            "is_roleplay": self.is_roleplay,
            "roleplay_character": self.roleplay_character,
            "roleplay_situation": self.roleplay_situation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        s = cls(data["session_id"], data.get("coach_name", "James"))
        s.session_name = data.get("session_name", "")
        s.messages = data["messages"]
        s.created_at = data["created_at"]
        s.last_active = data["last_active"]
        s.turn_count = data.get("turn_count", 0)
        s.is_roleplay = data.get("is_roleplay", False)
        s.roleplay_character = data.get("roleplay_character", "")
        s.roleplay_situation = data.get("roleplay_situation", "")
        # Backfill name from first user message if missing
        if not s.session_name:
            for msg in s.messages:
                if msg["role"] == "user":
                    name = msg["content"].strip().replace("\n", " ")
                    s.session_name = name[:50] + ("…" if len(name) > 50 else "")
                    break
        return s


class SessionStore:
    """File-backed session store. Persists sessions across server restarts."""

    def __init__(self):
        SESSIONS_DIR.mkdir(exist_ok=True)
        self._cache: dict[str, Session] = {}

    def get(self, session_id: str) -> Optional[Session]:
        if session_id in self._cache:
            return self._cache[session_id]
        path = SESSIONS_DIR / f"{session_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            session = Session.from_dict(data)
            self._cache[session_id] = session
            return session
        return None

    def create(self, coach_name: str = "James") -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id, coach_name)
        self._cache[session_id] = session
        self._save(session)
        return session

    def save(self, session: Session):
        self._save(session)

    def _save(self, session: Session):
        path = SESSIONS_DIR / f"{session.session_id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2))

    def list_recent(self, limit: int = 10) -> list[dict]:
        sessions = []
        for path in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            data = json.loads(path.read_text())
            # Backfill session_name from first user message if not stored
            session_name = data.get("session_name", "")
            if not session_name:
                for msg in data.get("messages", []):
                    if msg["role"] == "user":
                        name = msg["content"].strip().replace("\n", " ")
                        session_name = name[:50] + ("…" if len(name) > 50 else "")
                        break
            sessions.append({
                "session_id": data["session_id"],
                "coach_name": data.get("coach_name", "James"),
                "session_name": session_name,
                "turn_count": data.get("turn_count", 0),
                "last_active": data.get("last_active", 0),
            })
        return sessions


# ── Coaching Engine ──────────────────────────────────────────────────────────
class CoachingEngine:
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.store = SessionStore()
        self.rag = get_rag_engine()

    def get_or_create_session(self, session_id: Optional[str] = None, coach_name: str = "James") -> Session:
        if session_id:
            session = self.store.get(session_id)
            if session:
                return session
        return self.store.create(coach_name)

    def chat(self, session_id: str, user_message: str) -> Iterator[str]:
        """
        Stream a coaching response for the given session and user message.
        Yields text chunks as they arrive from Claude.
        """
        session = self.store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Add user message to history
        session.add_message("user", user_message)

        # Build system prompt — roleplay mode uses character prompt, not coaching prompt
        if session.is_roleplay and session.roleplay_character:
            system = build_roleplay_prompt(session.roleplay_character, session.roleplay_situation)
        else:
            context_chunks = self.rag.retrieve(user_message, top_k=2)
            system = build_system_prompt(context_chunks if context_chunks else None)

        # Build messages list
        messages = session.get_trimmed_history()

        # Stream response
        full_response = ""
        with self.client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text

        # Save assistant response to session
        session.add_message("assistant", full_response)
        self.store.save(session)

    def get_opening_message(self, session: Session) -> str:
        """Generate a context-aware opening message for a new session."""
        if session.is_roleplay and session.roleplay_character:
            # For roleplay, generate the character's opening in-character
            return f"[Role Play Active — you are practicing with {session.roleplay_character}. Type your first message to begin. Say 'end roleplay' at any time to get coaching feedback.]"

        coach = session.coach_name
        if coach == "Alexandra":
            return (
                "Welcome. I'm Alexandra — your Stratos coaching companion. "
                "I'm here to help you navigate the leadership challenges that don't come with easy answers. "
                "What's on your mind today?"
            )
        else:
            return (
                "Good to have you here. I'm James — your Stratos coaching companion. "
                "My job is to help you think more clearly about the leadership situations that matter most. "
                "What are we working on?"
            )
