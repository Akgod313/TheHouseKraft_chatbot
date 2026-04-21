import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import psycopg2
from dotenv import load_dotenv
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. INITIALIZE CLERK ---
# Securely fetch the key from Streamlit Secrets
clerk_key = None
if "CLERK_SECRET_KEY" in st.secrets:
    clerk_key = st.secrets["CLERK_SECRET_KEY"]
else:
    # Fallback to .env for local testing
    from dotenv import load_dotenv
    load_dotenv()
    clerk_key = os.getenv("CLERK_SECRET_KEY")

# Stop the app if the key is still missing
if not clerk_key:
    st.error("🔑 CLERK_SECRET_KEY not found in Secrets or .env file!")
    st.stop()
    
CLERK_KEY = st.secrets["CLERK_SECRET_KEY"]

headers = {
    'Authorization': f'Bearer {CLERK_KEY}',
    'Content-Type': 'application/json'
}

# --- 2. AUTHENTICATION LOGIC ---
if "user" not in st.session_state:
    # Main Landing Page Branding
    st.image("https://img.icons8.com/clouds/200/home.png", width=100)
    st.title("HouseKraft Expert")
    st.markdown("---")
    
    # Sidebar for Login/Signup
    with st.sidebar:
        st.header("🔐 Member Access")
        auth_mode = st.tabs(["Login", "Create Account"])

    # LOGIN TAB
    with auth_mode[0]:
        login_email = st.text_input("Email Address", key="login_email")
        if st.button("Login to Dashboard", use_container_width=True):
            res = requests.get('https://api.clerk.com/v1/users', headers=headers)
            users = res.json()
            found_user = next((u for u in users if any(e['email_address'] == login_email for e in u['email_addresses'])), None)
            
            if found_user:
                st.session_state["user"] = {"id": found_user['id'], "name": found_user.get('first_name', 'User')}
                st.success("Authenticated!")
                st.rerun()
            else:
                st.error("Account not found. Please sign up.")

    # SIGN UP TAB
    with auth_mode[1]:
        st.caption("Join HouseKraft to save your design history.")
        new_email = st.text_input("Email", key="reg_email")
        first_name = st.text_input("First Name", key="reg_name")
        password = st.text_input("Password", type="password", key="reg_pass")
        
        if st.button("Register Account", use_container_width=True):
            if not new_email or not password or not first_name:
                st.warning("All fields are required.")
            else:
                # FIXED PAYLOAD: Using the structure Clerk expects
                payload = {
                    "email_address": [new_email],
                    "password": password,
                    "first_name": first_name,
                    "skip_password_requirement": True,
                    "skip_password_checks": True
                }
                
                res = requests.post('https://api.clerk.com/v1/users', headers=headers, json=payload)
                
                if res.status_code == 200:
                    user_data = res.json()
                    st.session_state["user"] = {"id": user_data['id'], "name": first_name}
                    st.balloons()
                    st.success("Welcome to HouseKraft!")
                    st.rerun()
                else:
                    error_detail = res.json()
                    st.error(f"Error: {error_detail.get('errors', [{}])[0].get('message', 'Check Clerk Settings')}")
    
    st.stop() # Stops the rest of the app until logged in

# --- 3. DEFINE USER VARIABLES (Only reaches here if logged in) ---
user = st.session_state["user"]
user_id = st.session_state["user"]["id"]
user_name = user["first_name"]

# --- 4. SIDEBAR UI ---
with st.sidebar:
    st.success(f"✅ Connected: {user_name}")
    if st.button("Sign Out"):
        del st.session_state["user"]
        st.rerun()


# --- SETUP ---
if "GEMINI_API_KEY" in st.secrets:
    # Use this for Streamlit Cloud
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    # Use this for local testing
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

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


with st.sidebar:
    st.image("your_logo_path_or_url") # If you have a logo
    st.title("HouseKraft Support")
    st.info("This AI expert is trained on HouseKraft's internal design and repair manuals.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

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