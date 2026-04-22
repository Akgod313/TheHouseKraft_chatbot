import os
import threading
from contextlib import asynccontextmanager

from google import genai
from google.genai import types
from PIL import Image
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from fastapi import FastAPI, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import json
import traceback
from typing import Optional
from functools import lru_cache

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

SYSTEM_PROMPT = """
You are the HouseKraft AI Expert.
You can see images provided by the user.
Always combine the INTERNAL KNOWLEDGE (RAG) with what you see in the IMAGE
to give design, repair, or product advice.
Do not mention anything about RAG to the user, make it sound like you already know all of this.
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
    _, vector_db = load_resources()  # this already handles it
    if vector_db is None:
        return "No knowledge base available."
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# ─── DB INIT ───────────────────────────────────────────────────────────────────
def init_db():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS housekraft_chats (
                id         SERIAL PRIMARY KEY,
                clerk_id   TEXT      NOT NULL,
                role       TEXT      NOT NULL,
                content    TEXT      NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

# ─── LIFESPAN ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    threading.Thread(target=load_resources, daemon=True).start()
    yield
    if _pool:
        _pool.closeall()

# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://the-house-kraft-chatbot-web.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/history")
def get_history(x_user_id: str = Header(...)):
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT role, content FROM housekraft_chats WHERE clerk_id = %s ORDER BY created_at ASC",
            (x_user_id,)
        )
        messages = [dict(r) for r in cur.fetchall()]
        cur.close()
        return {"messages": messages}
    finally:
        release_conn(conn)


@app.delete("/history")
def clear_history(x_user_id: str = Header(...)):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM housekraft_chats WHERE clerk_id = %s", (x_user_id,))
        conn.commit()
        cur.close()
        return {"ok": True}
    finally:
        release_conn(conn)


@app.post("/chat")
async def chat(
    message:   str                  = Form(...),
    history:   str                  = Form(...),
    image:     Optional[UploadFile] = File(None),
    x_user_id: str                  = Header(...),
):
    async def stream_response():
        try:
            client, _ = load_resources()
            context   = cached_rag_search(message)

            # Build prompt text
            prompt_text = f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nUSER QUESTION: {message}"

            # Handle optional image
            img_bytes = None
            if image:
                img_bytes = await image.read()

            # Rebuild history for the new API
            raw_history = json.loads(history)
            gemini_history = []
            for m in raw_history[-10:]:
                role = "model" if m["role"] == "assistant" else "user"
                gemini_history.append(
                    types.Content(role=role, parts=[types.Part(text=m["content"])])
                )

            # Build current user turn parts
            user_parts = [types.Part(text=prompt_text)]
            if img_bytes:
                user_parts.append(
                    types.Part(
                        inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)
                    )
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

            # Save to DB in background
            def save(uid, user_msg, assistant_msg):
                conn = get_conn()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """INSERT INTO housekraft_chats (clerk_id, role, content)
                           VALUES (%s, %s, %s), (%s, %s, %s)""",
                        (uid, "user", user_msg, uid, "assistant", assistant_msg)
                    )
                    conn.commit()
                    cur.close()
                except Exception as e:
                    print(f"[DB save error] {e}")
                finally:
                    release_conn(conn)

            threading.Thread(
                target=save,
                args=(x_user_id, message, full_res),
                daemon=True
            ).start()

        except Exception as e:
            traceback.print_exc()
            yield f"ERROR: {e}"

    return StreamingResponse(stream_response(), media_type="text/plain")