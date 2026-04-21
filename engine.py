import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Path to your data folder
file_path = os.path.join("data", "knowledge.txt")

if not os.path.exists(file_path):
    print(f"❌ Error: {file_path} not found!")
else:
    # 1. LOAD & SPLIT
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 2. CONFIG EMBEDDINGS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=API_KEY,
        task_type="retrieval_document"
    )

    # 3. BATCHED BUILDING (Safe for Free Tier)
    print(f"🚀 Building index for {len(docs)} chunks...")
    vector_db = FAISS.from_documents([docs[0]], embeddings)
    
    for i in range(1, len(docs)):
        try:
            vector_db.add_documents([docs[i]])
            print(f"✅ Indexed {i+1}/{len(docs)}")
            time.sleep(1.5) # The "Gemini 3" sweet spot for free tier
        except Exception as e:
            if "429" in str(e):
                print("⏳ Quota reached. Cooling down for 10s...")
                time.sleep(10)
                vector_db.add_documents([docs[i]]) # Retry once
            else:
                print(f"⚠️ Skipping chunk {i}: {e}")

    # 4. SAVE
    vector_db.save_local("faiss_index")
    print("\n🎉 Indexing Complete! You can now run app.py")