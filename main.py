"""
Stratos AI Coaching Companion — FastAPI Server
Run: uvicorn main:app --reload --port 8000
"""

import os
import re
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

MODEL = "claude-sonnet-4-6"

# ── Scenario Data ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "id": "difficult-feedback",
        "title": "Delivering Difficult Feedback",
        "character": "Alex Chen",
        "character_role": "Senior Engineer, 5 years at company",
        "situation": "Alex is a high-performing senior engineer who has become increasingly dismissive in team meetings, interrupting colleagues and taking credit for others' work. You're his manager and need to address this behavior directly. Alex is defensive and highly valued by leadership, making this conversation high-stakes.",
        "goal": "Address the dismissive behavior and credit-taking directly while preserving the relationship and Alex's motivation.",
        "image": "/static/images/scenario-feedback.jpg",
        "category": "Difficult Conversations",
        "difficulty": "Hard",
        "attempts": 0
    },
    {
        "id": "performance-review",
        "title": "The Underperformer",
        "character": "Jordan Mills",
        "character_role": "Marketing Manager, 2 years at company",
        "situation": "Jordan was a top performer who has been significantly underperforming for 3 months. Quality of work has dropped, deadlines are being missed, and attitude in meetings has changed. You've tried casual check-ins but nothing has changed. This is a formal performance conversation.",
        "goal": "Get to the real reason behind the performance drop and create a shared action plan with clear accountability.",
        "image": "/static/images/scenario-performance.jpg",
        "category": "Performance",
        "difficulty": "Hard",
        "attempts": 0
    },
    {
        "id": "micromanager-boss",
        "title": "The Micromanager",
        "character": "Dawn Patel",
        "character_role": "VP of Engineering, your direct manager",
        "situation": "Dawn insists on approving every decision, reviews every PR, and schedules daily check-ins even for routine work. She's risk-averse and doesn't seem to trust your judgment despite 3 years of strong performance. You need to renegotiate your working relationship.",
        "goal": "Help Dawn feel confident enough to give you more autonomy without making her feel criticized or undermined.",
        "image": "/static/images/scenario-micromanager.jpg",
        "category": "Managing Up",
        "difficulty": "Medium",
        "attempts": 0
    },
    {
        "id": "peer-conflict",
        "title": "The Credit Taker",
        "character": "Marcus Webb",
        "character_role": "Peer Director, cross-functional partner",
        "situation": "Marcus presented work from your team to the executive leadership team last week without mentioning your team's contribution — framing it entirely as his initiative. This is the second time this has happened. You need to address it directly before the next board presentation.",
        "goal": "Establish clear norms around attribution and collaboration without creating a hostile peer relationship.",
        "image": "/static/images/scenario-peer.jpg",
        "category": "Peer Dynamics",
        "difficulty": "Medium",
        "attempts": 0
    },
    {
        "id": "board-communication",
        "title": "The Board Skeptic",
        "character": "Richard Stanton",
        "character_role": "Board Member, former CFO",
        "situation": "Richard has been consistently skeptical of your division's roadmap in board meetings, asking pointed questions about ROI and timeline. He has significant influence with the CEO. You've requested a 1:1 before the next board meeting to address his concerns directly.",
        "goal": "Turn a skeptic into a supporter by addressing his real concerns, not just the surface objections.",
        "image": "/static/images/scenario-boardroom.jpg",
        "category": "Executive Presence",
        "difficulty": "Expert",
        "attempts": 0
    },
    {
        "id": "change-resistance",
        "title": "Resistance to Change",
        "character": "Sharon Torres",
        "character_role": "Senior Director, 12 years at company",
        "situation": "Sharon is openly resistant to the organizational restructuring you're leading. She's been with the company longer than anyone and has significant informal authority. She's been lobbying peers against the changes. You need to bring her on board or neutralize her opposition.",
        "goal": "Understand her real objections and find a path where she becomes an advocate, not an obstacle.",
        "image": "/static/images/scenario-change.jpg",
        "category": "Change Management",
        "difficulty": "Expert",
        "attempts": 0
    },
    {
        "id": "stakeholder-alignment",
        "title": "The Misaligned Stakeholder",
        "character": "Priya Sharma",
        "character_role": "Chief Marketing Officer",
        "situation": "Priya has different expectations about the product roadmap than what your engineering team has been building toward. The disconnect only became apparent in a cross-functional meeting where she revealed she had been telling customers features would ship that are 6 months out on your roadmap.",
        "goal": "Get to shared understanding of the roadmap without blame, and establish a process to prevent future misalignment.",
        "image": "/static/images/scenario-stakeholder.jpg",
        "category": "Stakeholder Management",
        "difficulty": "Hard",
        "attempts": 0
    },
    {
        "id": "promotion-conversation",
        "title": "The Promotion Ask",
        "character": "Catherine Lowe",
        "character_role": "SVP of Product, your skip-level manager",
        "situation": "You've been performing at the next level for 18 months and your direct manager has been non-committal about your promotion timeline. You've requested this meeting with Catherine to advocate for yourself directly. She respects directness but has high standards for executive-level communication.",
        "goal": "Make a compelling case for your promotion using evidence and strategic framing — not just tenure or effort.",
        "image": "/static/images/scenario-promotion.jpg",
        "category": "Career Development",
        "difficulty": "Medium",
        "attempts": 0
    },
    {
        "id": "team-conflict",
        "title": "The Team Conflict",
        "character": "Both: Lena K. & David R.",
        "character_role": "Two senior engineers with escalating conflict",
        "situation": "Two of your best engineers have a growing interpersonal conflict that's affecting team morale. They disagree on technical architecture and the tension has spilled into meetings. You're meeting with David first — he's convinced Lena is trying to undermine his work.",
        "goal": "De-escalate David's narrative and get to the shared interests underneath the conflict.",
        "image": "/static/images/scenario-conflict.jpg",
        "category": "Conflict Resolution",
        "difficulty": "Hard",
        "attempts": 0
    },
    {
        "id": "executive-vision",
        "title": "Setting Vision as a New Leader",
        "character": "Your new leadership team (played by AI)",
        "character_role": "5-person team you inherited 90 days ago",
        "situation": "You joined as VP 90 days ago and inherited a team that was loyal to your predecessor. You've been observing, learning, and building trust. Now it's time to share your vision for where you're taking the team over the next 12 months. The team is skeptical but open.",
        "goal": "Present your vision in a way that inspires, invites collaboration, and addresses the uncertainty the team feels.",
        "image": "/static/images/scenario-feedback.jpg",
        "category": "Executive Presence",
        "difficulty": "Hard",
        "attempts": 0
    }
]


# ── Request Models ────────────────────────────────────────────────────────────
class ProgressRequest(BaseModel):
    session_ids: list[str] = []


class NewSessionRequest(BaseModel):
    coach_name: str = "James"  # "James" or "Alexandra"
    is_roleplay: bool = False
    roleplay_character: str = ""
    roleplay_situation: str = ""


class ChatRequest(BaseModel):
    session_id: str
    message: str


class CompleteRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 2048
    model: str | None = None  # Optional override — e.g. "claude-haiku-4-5-20251001" for speed-critical calls


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
    import asyncio
    if req.coach_name not in ("James", "Alexandra"):
        raise HTTPException(status_code=400, detail="coach_name must be 'James' or 'Alexandra'")

    session = engine.get_or_create_session(coach_name=req.coach_name)

    # Configure roleplay mode if requested
    if req.is_roleplay and req.roleplay_character:
        session.is_roleplay = True
        session.roleplay_character = req.roleplay_character
        session.roleplay_situation = req.roleplay_situation
        session.session_name = f"Role Play: {req.roleplay_character}"
        # Run blocking Claude API call in thread pool with timeout
        try:
            opening = await asyncio.wait_for(
                asyncio.to_thread(engine.generate_roleplay_opening, session),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            opening = f"*takes a breath and looks up*\n\nAlright — let's get into it."
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
        import asyncio
        feedback = await asyncio.to_thread(engine.generate_feedback, session_id)
        return feedback  # structured dict with score, headline, summary, etc.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenarios")
async def get_scenarios():
    """Return all scenario metadata."""
    return {"scenarios": SCENARIOS}


@app.post("/progress/generate")
async def generate_progress(req: ProgressRequest):
    """Analyze last N sessions and generate a progress report using Claude."""
    if req.session_ids:
        sessions = [engine.store.get(sid) for sid in req.session_ids if engine.store.get(sid)]
    else:
        recent = engine.store.list_recent(limit=5)
        sessions = [engine.store.get(s["session_id"]) for s in recent if engine.store.get(s["session_id"])]

    if not sessions:
        return {"error": "No sessions found", "skills": {}, "leader_score": 0, "growth_plan": ""}

    all_user_messages = []
    for s in sessions:
        for msg in s.messages:
            if msg["role"] == "user":
                all_user_messages.append(msg["content"])

    if not all_user_messages:
        return {"error": "No messages found", "skills": {}, "leader_score": 0, "growth_plan": ""}

    transcript = "\n".join(f"- {m}" for m in all_user_messages[-30:])

    prompt = f"""You are an executive coaching analyst. Based on these user messages from coaching sessions, evaluate the leader's communication and leadership skills.

User messages from coaching sessions:
{transcript}

Provide a JSON response with exactly this structure:
{{
  "leader_score": <number 0-100>,
  "skills": {{
    "Goal Clarity": <number 1-10>,
    "Self Awareness": <number 1-10>,
    "Stakeholder Thinking": <number 1-10>,
    "Communication": <number 1-10>,
    "Accountability": <number 1-10>,
    "Strategic Thinking": <number 1-10>
  }},
  "growth_plan": "<2-3 sentence personalized growth narrative>",
  "improvement_tip": "<one specific actionable tip>",
  "completed_sessions": {len(sessions)}
}}

Return ONLY valid JSON, no other text."""

    response = engine.client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        return data
    return {"error": "Could not generate progress", "skills": {}, "leader_score": 50, "growth_plan": "Keep coaching to build your progress profile."}


@app.post("/sessions/{session_id}/annotate")
async def annotate_session(session_id: str):
    """Generate inline praise/tip annotations for a session transcript."""
    session = engine.store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_messages = [(i, msg["content"]) for i, msg in enumerate(session.messages) if msg["role"] == "user"]

    if len(user_messages) < 1:
        return {"annotations": {}}

    transcript = "\n".join(f"[Message {i}]: {content}" for i, content in user_messages[:10])

    prompt = f"""You are an executive coaching observer. Review these user messages from a coaching conversation and provide brief inline annotations.

{transcript}

For each message, provide either a "praise" (something they did well) or a "tip" (something to improve). Be specific and brief (under 15 words each).

Return JSON:
{{
  "annotations": {{
    "<message_index>": {{"type": "praise"|"tip", "text": "<annotation text>"}},
    ...
  }}
}}

Only annotate messages that have something noteworthy — skip generic or very short messages.
Return ONLY valid JSON."""

    response = engine.client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"annotations": {}}


@app.post("/api/complete")
async def api_complete(req: CompleteRequest):
    """
    Generic Claude completion endpoint — emulates the window.claude.complete()
    sandbox helper so static front-ends (ai2) can reuse our server-side key.

    Accepts either:
      - { "prompt": "..." }                       → single user message
      - { "messages": [{ role, content }, ...] }  → full messages array
    Returns: { "text": "<completion>" }
    """
    if not req.prompt and not req.messages:
        raise HTTPException(status_code=400, detail="Provide either 'prompt' or 'messages'")

    messages = req.messages or [{"role": "user", "content": req.prompt}]
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail="Each message needs role=user|assistant")

    try:
        import asyncio

        # Whitelist allowed models to prevent arbitrary model selection
        ALLOWED_MODELS = {
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-7",
        }
        chosen_model = req.model if req.model in ALLOWED_MODELS else MODEL

        def _call():
            resp = engine.client.messages.create(
                model=chosen_model,
                max_tokens=max(1, min(req.max_tokens, 8192)),
                messages=messages,
            )
            return "".join(block.text for block in resp.content if block.type == "text")

        # Haiku is fast; sonnet/opus may take longer. Timeouts reflect that.
        timeout_s = 60.0 if chosen_model.startswith("claude-haiku") else 150.0
        text = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_s)
        return {"text": text}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Claude request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "963430267713-166vehnmd1ftvd9cdp5s611ra6ool1uk.apps.googleusercontent.com",
)

USERS_FILE = Path("sessions") / "users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


class GoogleAuthIn(BaseModel):
    credential: str


@app.post("/auth/google")
async def auth_google(body: GoogleAuthIn):
    import time
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Auth library not available: {e}")

    try:
        info = id_token.verify_oauth2_token(
            body.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {e}")

    if info.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(status_code=401, detail="Invalid token issuer")

    sub = info.get("sub")
    email = info.get("email")
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Token missing required fields")

    users = _load_users()
    existing = users.get(sub)
    is_new = existing is None
    record = existing or {"sub": sub, "created_at": int(time.time())}
    record.update({
        "email": email,
        "email_verified": bool(info.get("email_verified")),
        "name": info.get("name", ""),
        "picture": info.get("picture", ""),
        "last_login": int(time.time()),
    })
    users[sub] = record
    _save_users(users)

    return {
        "is_new": is_new,
        "email": email,
        "name": info.get("name", ""),
        "picture": info.get("picture", ""),
        "sub": sub,
    }


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
