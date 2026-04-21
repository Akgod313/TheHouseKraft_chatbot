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

st.set_page_config(page_title="TheHouseKraft Multimodal AI", layout="centered")

# --- PROMPT ---
SYSTEM_PROMPT = """
You are the HouseKraft AI Expert. 
You can see images provided by the user. 
Always combine the INTERNAL KNOWLEDGE (RAG) with what you see in the IMAGE 
to give design, repair, or product advice.
Do not mention anything about RAG to the user, make it sound like you already know all of this.
"""

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    genai.configure(api_key=API_KEY)
    # Gemini 3 Flash is the best for multimodal reasoning
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=API_KEY,
        task_type="retrieval_query"
    )
    
    # Load the local index built by engine.py
    vector_db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return model, vector_db

# --- SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🏠 TheHouseKraft Expert")
st.caption("Ask questions or upload photos of your space for AI analysis.")

# Sidebar for Image Upload
with st.sidebar:
    st.header("Visual Context")
    uploaded_file = st.file_uploader("Upload space photo", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Target Space", use_container_width=True)

# History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT ENGINE ---
if prompt := st.chat_input("How can TheHouseKraft help you today?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            #1. THE THINKING STATE
            with st.status("TheHouseKraft Expert is thinking...", expanded=False) as status:
                model, vector_db = load_resources()

                docs = vector_db.similarity_search(prompt, k=3)
                context = "\n".join([d.page_content for d in docs])

                content_list = [
                    f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nUSER QUESTION: {prompt}"
                ]

                if uploaded_file:
                    img_data = Image.open(uploaded_file)
                    content_list.append(img_data)

                status.update(label="Found the answer!", state="complete", expanded=False)
            
            #2. THE TYPING STATE
            res_placeholder = st.empty()
            full_res = ""

            
            response = model.generate_content(content_list, stream=True)
            for chunk in response:
                full_res += chunk.text
                res_placeholder.markdown(full_res + "▌")
            res_placeholder.markdown(full_res)
            
            st.session_state.messages.append({"role": "assistant", "content": full_res})

        except Exception as e:
            st.error(f"Error: {e}. If it's a 429, just wait 10 seconds and try again!")

    st.rerun()