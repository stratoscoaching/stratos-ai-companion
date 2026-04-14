"""
Stratos AI Coaching Companion — FastAPI Server
Run: uvicorn main:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from coaching_engine import CoachingEngine

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stratos AI Coaching Companion",
    description="AI executive coaching grounded in CEC, ICF, and Stratos methodology",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (chat UI)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Init coaching engine
engine = CoachingEngine()


# ── Request Models ────────────────────────────────────────────────────────────
class NewSessionRequest(BaseModel):
    coach_name: str = "James"  # "James" or "Alexandra"
    is_roleplay: bool = False
    roleplay_character: str = ""
    roleplay_situation: str = ""


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    index = Path("static/index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text())
    return HTMLResponse(content="<h1>Stratos AI Coaching Companion</h1><p>Static files not found.</p>")


@app.post("/sessions/new")
async def new_session(req: NewSessionRequest):
    """Create a new coaching session."""
    if req.coach_name not in ("James", "Alexandra"):
        raise HTTPException(status_code=400, detail="coach_name must be 'James' or 'Alexandra'")

    session = engine.get_or_create_session(coach_name=req.coach_name)

    # Configure roleplay mode if requested
    if req.is_roleplay and req.roleplay_character:
        session.is_roleplay = True
        session.roleplay_character = req.roleplay_character
        session.roleplay_situation = req.roleplay_situation
        session.session_name = f"Role Play: {req.roleplay_character}"
        # Character speaks first — real API call for in-character opening
        opening = engine.generate_roleplay_opening(session)
    else:
        opening = engine.get_opening_message(session)

    # Save opening message as first assistant turn
    session.add_message("assistant", opening)
    engine.store.save(session)

    return {
        "session_id": session.session_id,
        "coach_name": session.coach_name,
        "opening_message": opening,
        "is_roleplay": session.is_roleplay,
        "roleplay_character": session.roleplay_character,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream a coaching response via Server-Sent Events.
    Frontend reads: event.data chunks, then [DONE] to close.
    """
    session = engine.store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    def generate():
        try:
            for chunk in engine.chat(req.session_id, req.message):
                # SSE format
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history."""
    session = engine.store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "coach_name": session.coach_name,
        "turn_count": session.turn_count,
        "messages": session.messages,
    }


@app.get("/sessions")
async def list_sessions():
    """List recent sessions."""
    return {"sessions": engine.store.list_recent(limit=20)}


@app.post("/sessions/{session_id}/evaluate")
async def evaluate_session(session_id: str):
    """Generate end-of-session coaching feedback for a roleplay session."""
    session = engine.store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_roleplay:
        raise HTTPException(status_code=400, detail="Feedback only available for roleplay sessions")
    if session.turn_count < 2:
        raise HTTPException(status_code=400, detail="Not enough conversation to evaluate")
    try:
        feedback = engine.generate_feedback(session_id)
        return {"feedback": feedback, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    rag_chunks = len(engine.rag.chunks) if engine.rag._loaded else 0
    return {
        "status": "ok",
        "api_key_configured": api_key_set,
        "rag_chunks_loaded": rag_chunks,
        "model": "claude-sonnet-4-6",
    }
