import streamlit as st

st.set_page_config(page_title="TheHouseKraft Multimodal AI", layout="centered")

import os
import threading
import google.generativeai as genai
from PIL import Image
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ─── REACT HANDSHAKE ───────────────────────────────────────────────────────────
query_params = st.query_params
if "user_id" in query_params:
    st.session_state["user_id"] = query_params["user_id"]
    st.session_state["user_name"] = query_params.get("name", "Homeowner")
    st.query_params.clear()

# ─── BOUNCER ───────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.warning("🚨 Access Denied. Please log in via the HouseKraft Portal.")
    st.info("Direct your browser to your React site (e.g., http://localhost:5173)")
    st.stop()

user_id   = st.session_state["user_id"]
user_name = st.session_state["user_name"]

# ─── API KEY ───────────────────────────────────────────────────────────────────
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

# ─── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the HouseKraft AI Expert.
You can see images provided by the user.
Always combine the INTERNAL KNOWLEDGE (RAG) with what you see in the IMAGE
to give design, repair, or product advice.
Do not mention anything about RAG to the user, make it sound like you already know all of this.
"""

# ─── SPEED FIX 1: CONNECTION POOL ──────────────────────────────────────────────
# Instead of opening a fresh TCP connection to Neon on every message (slow!),
# we keep a pool of 2-5 reusable connections alive for the lifetime of the server.
@st.cache_resource
def get_pool():
    return ThreadedConnectionPool(
        minconn=2,
        maxconn=5,
        dsn=st.secrets["DATABASE_URL"]
    )

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

# ─── SPEED FIX 2: CACHE MODEL + VECTOR DB ──────────────────────────────────────
@st.cache_resource
def load_resources():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=API_KEY,
        task_type="retrieval_query"
    )
    vector_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return model, vector_db

# ─── SPEED FIX 3: PRE-WARM ON PAGE LOAD ───────────────────────────────────────
# Kick off resource loading the moment the page opens, in a background thread.
# By the time the user types their first message, model + FAISS are already cached.
if "resources_warmed" not in st.session_state:
    st.session_state["resources_warmed"] = True
    threading.Thread(target=load_resources, daemon=True).start()

# ─── SPEED FIX 4: CACHE RAG RESULTS PER QUERY ──────────────────────────────────
# Identical or similar queries skip the embedding + FAISS round trip entirely.
@st.cache_data(ttl=3600, max_entries=100)
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

init_db()

# ─── LOAD HISTORY ONCE INTO SESSION STATE ──────────────────────────────────────
if "messages" not in st.session_state:
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT role, content FROM housekraft_chats WHERE clerk_id = %s ORDER BY created_at ASC",
            (user_id,)
        )
        st.session_state.messages = [dict(r) for r in cur.fetchall()]
        cur.close()
    finally:
        release_conn(conn)

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 TheHouseKraft Expert")

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/home.png", width=100)
    st.title("HouseKraft Support")
    st.write(f"Welcome, **{user_name}**!")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── CHAT ENGINE ───────────────────────────────────────────────────────────────
prompt_obj = st.chat_input(
    "How can TheHouseKraft help you today?",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

if prompt_obj:
    user_text  = str(prompt_obj.text)
    user_files = prompt_obj.get("files", [])

    # Show user bubble immediately — no waiting
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        if user_files:
            for f in user_files:
                st.image(f, use_container_width=True)
        st.markdown(user_text)

    # Assistant response
    with st.chat_message("assistant"):
        res_placeholder = st.empty()
        full_res = ""

        try:
            with st.spinner("Thinking about it...."):
                # Model is instant — already cached from pre-warm
                model, _ = load_resources()

                # RAG is cached — skips embedding call on repeated/similar queries
                context = cached_rag_search(user_text)

                content_list = [
                    f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nUSER QUESTION: {user_text}"
                ]

                if user_files:
                    content_list.append(Image.open(user_files[0]))

                # Build conversation history from session state (no DB read)
                history = []
                for m in st.session_state.messages[-11:-1]:
                    role = "model" if m["role"] == "assistant" else "user"
                    history.append({"role": role, "parts": [str(m["content"])]})

                chat     = model.start_chat(history=history)
                # Initiate the streaming request while spinner is still showing
                response = chat.send_message(content_list, stream=True)

            # Spinner gone — stream tokens directly into the UI
            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    res_placeholder.markdown(full_res + "▌")

            res_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

            # ─── SPEED FIX 5: NON-BLOCKING BACKGROUND DB WRITE ────────────────
            # Both messages are saved together after streaming is done.
            # The UI never waits on Neon — zero flicker, zero delay.
            def save_to_db(uid, u_text, a_text):
                conn = get_conn()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """INSERT INTO housekraft_chats (clerk_id, role, content)
                           VALUES (%s, %s, %s), (%s, %s, %s)""",
                        (uid, "user", u_text, uid, "assistant", a_text)
                    )
                    conn.commit()
                    cur.close()
                except Exception as db_err:
                    print(f"[DB save error] {db_err}")
                finally:
                    release_conn(conn)

            threading.Thread(
                target=save_to_db,
                args=(user_id, user_text, full_res),
                daemon=True
            ).start()

        except Exception as e:
            res_placeholder.error(f"Error: {e}")