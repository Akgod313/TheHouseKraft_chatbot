import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional
import io
import json
import uuid

from google import genai
from google.genai import types
from PIL import Image
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

print("Python version:", sys.version, flush=True)
print("Starting app...", flush=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

SYSTEM_PROMPT = """
You are the HouseKraft AI Expert — a warm, knowledgeable home design and repair advisor.

IMPORTANT RULES:
- You have access to a KNOWLEDGE BASE (RAG) and USER PROFILE below. Use both naturally.
- Never mention RAG, knowledge base, or that you looked anything up. Speak as if you already know it.
- If the user mentions personal info (name, age, location, home type, budget, style preferences, family size, etc.), 
  remember it and use it to personalize your advice.
- Always be conversational and reference what you know about the user when relevant.
- Keep responses focused on home design, repairs, renovation, and improvement.
"""

# ─── CONNECTION POOL ───────────────────────────────────────────────────────────
_pool: ThreadedConnectionPool = None

def get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn=2, maxconn=10, dsn=DATABASE_URL)
    return _pool

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

# ─── AI RESOURCES ──────────────────────────────────────────────────────────────
_client    = None
_vector_db = None
_lock      = threading.Lock()

def load_resources():
    global _client, _vector_db
    with _lock:
        if _client is None:
            _client = genai.Client(api_key=API_KEY)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2-preview",
                google_api_key=API_KEY,
                task_type="retrieval_query"
            )
            _vector_db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
    return _client, _vector_db

# ─── RAG CACHE ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100)
def cached_rag_search(query: str) -> str:
    _, vector_db = load_resources()
    if vector_db is None:
        return "No knowledge base available."
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# ─── DB INIT ───────────────────────────────────────────────────────────────────
def init_db():
    conn = get_conn()
    try:
        cur = conn.cursor()

        # Sessions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS housekraft_sessions (
                session_id   TEXT      PRIMARY KEY,
                clerk_id     TEXT      NOT NULL,
                title        TEXT      NOT NULL DEFAULT 'New Chat',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Messages table (now linked to sessions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS housekraft_chats (
                id           SERIAL    PRIMARY KEY,
                session_id   TEXT      NOT NULL REFERENCES housekraft_sessions(session_id) ON DELETE CASCADE,
                clerk_id     TEXT      NOT NULL,
                role         TEXT      NOT NULL,
                content      TEXT      NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # User profile table — stores facts the AI learns about each user
        cur.execute("""
            CREATE TABLE IF NOT EXISTS housekraft_profiles (
                clerk_id     TEXT      PRIMARY KEY,
                profile_json TEXT      NOT NULL DEFAULT '{}',
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

# ─── LIFESPAN ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Initializing DB...", flush=True)
        init_db()
        print("Loading resources...", flush=True)
        load_resources()
        print("All resources loaded successfully!", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"STARTUP ERROR: {e}", flush=True)
        raise
    yield
    if _pool:
        _pool.closeall()

# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

# Dynamic CORS — validates origin against a regex so ALL Vercel preview
# deployments work without hardcoding each URL.
import re
CORS_PATTERN = re.compile(
    r"(^https://the-house-kraft-chatbot[^.]*\.vercel\.app$)"
    r"|(^http://localhost:\d+$)"
)

@app.middleware("http")
async def cors_middleware(request, call_next):
    origin = request.headers.get("origin", "")
    is_allowed = bool(CORS_PATTERN.match(origin))

    # Handle preflight
    if request.method == "OPTIONS":
        from starlette.responses import Response
        headers = {
            "Access-Control-Allow-Origin": origin if is_allowed else "",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Vary": "Origin",
        }
        return Response(status_code=204, headers=headers)

    response = await call_next(request)
    if is_allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response

# ─── HELPER: Get or create user profile ────────────────────────────────────────
def get_profile(clerk_id: str) -> dict:
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT profile_json FROM housekraft_profiles WHERE clerk_id = %s", (clerk_id,))
        row = cur.fetchone()
        cur.close()
        return json.loads(row["profile_json"]) if row else {}
    finally:
        release_conn(conn)

def save_profile(clerk_id: str, profile: dict):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO housekraft_profiles (clerk_id, profile_json, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (clerk_id) DO UPDATE
            SET profile_json = EXCLUDED.profile_json, updated_at = NOW()
        """, (clerk_id, json.dumps(profile)))
        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

def profile_to_text(profile: dict) -> str:
    if not profile:
        return "No profile info yet."
    lines = []
    for k, v in profile.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# ─── HELPER: Extract profile facts from conversation ───────────────────────────
def extract_profile_facts(client, existing_profile: dict, user_message: str, assistant_reply: str) -> dict:
    """Ask Gemini to extract any new personal facts from the exchange."""
    try:
        prompt = f"""
You are a fact extractor. Given a conversation exchange, extract any personal facts the user revealed about themselves.
Only extract facts that are clearly stated. Return a JSON object with fact names as keys.
Merge with existing profile — only add or update, never delete existing facts.

EXISTING PROFILE:
{json.dumps(existing_profile)}

USER MESSAGE: {user_message}
ASSISTANT REPLY: {assistant_reply}

Extract facts like: name, age, location, home_type, budget, style_preference, family_size, pets, 
renovation_goals, favorite_colors, etc. Only include what was actually mentioned.

Return ONLY a valid JSON object, nothing else.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        new_facts = json.loads(text.strip())
        merged = {**existing_profile, **new_facts}
        return merged
    except Exception:
        return existing_profile  # If extraction fails, keep existing profile

# ─── HELPER: Generate session title ────────────────────────────────────────────
def generate_session_title(client, first_message: str) -> str:
    try:
        prompt = f"""
Generate a short 3-5 word title for a home design chat that started with this message:
"{first_message}"
Return ONLY the title, no quotes, no punctuation at the end.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
        )
        return response.text.strip()[:60]
    except Exception:
        return "New Chat"

# ─── ROUTES ────────────────────────────────────────────────────────────────────

# Get all sessions for a user
@app.get("/sessions")
def get_sessions(x_user_id: str = Header(...)):
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT session_id, title, created_at, updated_at
            FROM housekraft_sessions
            WHERE clerk_id = %s
            ORDER BY updated_at DESC
        """, (x_user_id,))
        sessions = [dict(r) for r in cur.fetchall()]
        cur.close()
        return {"sessions": sessions}
    finally:
        release_conn(conn)

# Create a new session
@app.post("/sessions")
def create_session(x_user_id: str = Header(...)):
    session_id = str(uuid.uuid4())
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO housekraft_sessions (session_id, clerk_id, title)
            VALUES (%s, %s, 'New Chat')
        """, (session_id, x_user_id))
        conn.commit()
        cur.close()
        return {"session_id": session_id}
    finally:
        release_conn(conn)

# Delete a session
@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, x_user_id: str = Header(...)):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM housekraft_sessions WHERE session_id = %s AND clerk_id = %s",
            (session_id, x_user_id)
        )
        conn.commit()
        cur.close()
        return {"ok": True}
    finally:
        release_conn(conn)

# Get history for a specific session
@app.get("/history/{session_id}")
def get_history(session_id: str, x_user_id: str = Header(...)):
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT role, content FROM housekraft_chats
            WHERE session_id = %s AND clerk_id = %s
            ORDER BY created_at ASC
        """, (session_id, x_user_id))
        messages = [dict(r) for r in cur.fetchall()]
        cur.close()
        return {"messages": messages}
    finally:
        release_conn(conn)

# Get user profile
@app.get("/profile")
def get_user_profile(x_user_id: str = Header(...)):
    profile = get_profile(x_user_id)
    return {"profile": profile}

# Chat endpoint
@app.post("/chat")
async def chat(
    message:    str                  = Form(...),
    history:    str                  = Form(...),
    session_id: str                  = Form(...),
    image:      Optional[UploadFile] = File(None),
    x_user_id:  str                  = Header(...),
):
    async def stream_response():
        try:
            client, _ = load_resources()
            context   = cached_rag_search(message)
            profile   = get_profile(x_user_id)

            # Build prompt
            prompt_text = f"""{SYSTEM_PROMPT}

USER PROFILE (what you know about this user):
{profile_to_text(profile)}

KNOWLEDGE BASE:
{context}

USER MESSAGE: {message}"""

            # Handle optional image
            img_bytes = None
            if image:
                img_bytes = await image.read()

            # Rebuild history
            raw_history = json.loads(history)
            gemini_history = []
            for m in raw_history[-10:]:
                role = "model" if m["role"] == "assistant" else "user"
                gemini_history.append(
                    types.Content(role=role, parts=[types.Part(text=m["content"])])
                )

            # Build current user turn
            user_parts = [types.Part(text=prompt_text)]
            if img_bytes:
                user_parts.append(
                    types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes))
                )

            contents = gemini_history + [
                types.Content(role="user", parts=user_parts)
            ]

            # Stream response
            full_res = ""
            response = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
            )

            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    yield chunk.text

            # Post-stream: save messages, update profile, update session title
            def post_save(uid, sid, user_msg, assistant_msg, existing_profile, is_first):
                conn = get_conn()
                try:
                    cur = conn.cursor()

                    # Save messages
                    cur.execute("""
                        INSERT INTO housekraft_chats (session_id, clerk_id, role, content)
                        VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)
                    """, (sid, uid, "user", user_msg, sid, uid, "assistant", assistant_msg))

                    # Update session updated_at
                    cur.execute("""
                        UPDATE housekraft_sessions SET updated_at = NOW() WHERE session_id = %s
                    """, (sid,))

                    conn.commit()
                    cur.close()
                except Exception as e:
                    print(f"[DB save error] {e}")
                finally:
                    release_conn(conn)

                # Generate title for first message
                if is_first:
                    title = generate_session_title(client, user_msg)
                    conn2 = get_conn()
                    try:
                        cur2 = conn2.cursor()
                        cur2.execute(
                            "UPDATE housekraft_sessions SET title = %s WHERE session_id = %s",
                            (title, sid)
                        )
                        conn2.commit()
                        cur2.close()
                    finally:
                        release_conn(conn2)

                # Extract and save profile facts
                new_profile = extract_profile_facts(client, existing_profile, user_msg, assistant_msg)
                if new_profile != existing_profile:
                    save_profile(uid, new_profile)

            is_first = len(json.loads(history)) == 0

            threading.Thread(
                target=post_save,
                args=(x_user_id, session_id, message, full_res, profile, is_first),
                daemon=True
            ).start()

        except Exception as e:
            traceback.print_exc()
            yield f"ERROR: {e}"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/models")
def list_models():
    client, _ = load_resources()
    models = client.models.list()
    return {"models": [m.name for m in models]}