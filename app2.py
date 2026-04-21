import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
st.set_page_config(page_title="HouseKraft Pro Chat", layout="centered")

@st.cache_resource
def load_resources():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=API_KEY,
        task_type="retrieval_query"
    )
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return model, vector_db

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🏠 HouseKraft Expert")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- NATIVE CHAT INPUT ---
prompt_input = st.chat_input("Ask HouseKraft...", accept_file=True, file_type=["jpg", "jpeg", "png"])

if prompt_input:
    user_text = prompt_input.text
    user_files = prompt_input.get("files", [])

    # 1. Save User Message to UI State
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
        if user_files:
            st.image(user_files[0], width=300)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        model, vector_db = load_resources()
        
        # RAG Context Retrieval
        docs = vector_db.similarity_search(user_text, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # 3. CONVERT LIMITED HISTORY (Last 10 messages)
        # This keeps the "sliding window" of memory
        history = []
        last_messages = st.session_state.messages[-11:-1] # Get last 10, exclude current
        
        for m in last_messages:
            role = "model" if m["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [m["content"]]})

        # 4. START CHAT WITH MEMORY
        chat = model.start_chat(history=history)
        
        # Prepare the multimodal content
        full_query = f"INTERNAL KNOWLEDGE: {context}\n\nUSER QUESTION: {user_text}"
        current_parts = [full_query]
        if user_files:
            current_parts.append(Image.open(user_files[0]))

        # 5. GENERATE & STREAM
        res_placeholder = st.empty()
        full_res = ""
        try:
            response = chat.send_message(current_parts, stream=True)
            for chunk in response:
                full_res += chunk.text
                res_placeholder.markdown(full_res + "▌")
            res_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
        except Exception as e:
            st.error(f"Error: {e}")

    st.rerun()