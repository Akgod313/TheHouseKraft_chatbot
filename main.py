import os
import threading
from contextlib import asynccontextmanager

import google.generativeai as genai
from PIL import Image
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import json
from typing import Optional
from functools import lru_cache

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

genai.configure(api_key=API_KEY)

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
_model      = None
_vector_db  = None
_lock       = threading.Lock()

def load_resources():
    global _model, _vector_db
    with _lock:
        if _model is None:
            genai.configure(api_key=API_KEY)
            _model = genai.GenerativeModel('gemini-2.5-flash-lite')
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
    return _model, _vector_db

# ─── RAG CACHE ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=100)
def cached_rag_search(query: str) -> str:
    _, vector_db = load_resources()
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

# ─── LIFESPAN (startup/shutdown) ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm everything on server start
    init_db()
    threading.Thread(target=load_resources, daemon=True).start()
    yield
    if _pool:
        _pool.closeall()

# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "https://the-house-kraft-chatbot-web.vercel.app/"],  # add your prod URL here when deploying
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
    message: str       = Form(...),
    history: str       = Form(...),   # JSON string of [{role, content}]
    image:   Optional[UploadFile] = File(None),
    x_user_id: str     = Header(...),
):
    """
    Streams the assistant response token by token as plain text chunks.
    The client reads via ReadableStream / EventSource.
    """

    async def stream_response():
        model, _ = load_resources()
        context  = cached_rag_search(message)

        content_list = [
            f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nUSER QUESTION: {message}"
        ]

        if image:
            img_bytes = await image.read()
            img_data  = Image.open(io.BytesIO(img_bytes))
            content_list.append(img_data)

        # Rebuild Gemini history from JSON
        raw_history = json.loads(history)
        gemini_history = []
        for m in raw_history[-10:]:
            role = "model" if m["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        chat_session = model.start_chat(history=gemini_history)
        response     = chat_session.send_message(content_list, stream=True)

        full_res = ""
        for chunk in response:
            if chunk.text:
                full_res += chunk.text
                yield chunk.text

        # Save both messages to DB in background after streaming finishes
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

    return StreamingResponse(stream_response(), media_type="text/plain")