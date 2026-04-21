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



# History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], use_container_width=True)
        st.markdown(message["content"])

# --- CHAT ENGINE ---
if prompt := st.chat_input("How can TheHouseKraft help you today?", accept_file=True, file_type=["jpg", "jpeg", "png"]):
    
    # 1. Extract data
    user_text = str(prompt.text)
    user_files = prompt.get("files", [])

    # 2. Add to session state (We'll store the text, but display the image separately)
    st.session_state.messages.append({"role": "user", "content": user_text})
    
    # 3. DISPLAY IN UI
    with st.chat_message("user"):
        # --- NEW: Image Preview Logic ---
        if user_files:
            for file in user_files:
                st.image(file, use_container_width=True) # Shows the image above text
        
        st.markdown(user_text)

    # 4. Assistant Response
    with st.chat_message("assistant"):
        res_placeholder = st.empty()
        full_res = ""
        
        try:
            with st.status("Thinking about it....", expanded=False) as status:
                model, vector_db = load_resources()
                docs = vector_db.similarity_search(user_text, k=3)
                context = "\n".join([d.page_content for d in docs])

                content_list = [
                    f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{context}\n\nUSER QUESTION: {user_text}"
                ]               

                if user_files:
                    # Open the image for the AI to see
                    img_data = Image.open(user_files[0])
                    content_list.append(img_data)

                status.update(label="I got it! One moment...", state="complete", expanded=False)
            
            # Memory and Chat
            history = []
            for m in st.session_state.messages[-11:-1]:
                role = "model" if m["role"] == "assistant" else "user"
                history.append({"role": role, "parts": [str(m["content"])]})

            chat = model.start_chat(history=history)
            response = chat.send_message(content_list, stream=True)

            for chunk in response:
                if chunk.text:
                    full_res += chunk.text
                    res_placeholder.markdown(full_res + "▌")
            
            res_placeholder.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})

        except Exception as e:
            res_placeholder.error(f"Error: {e}")
            st.stop()

    st.rerun()