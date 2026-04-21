import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
import time

# 1. Configuration
genai.configure(api_key="AIzaSyCPasRa5W_xnkyevNYh6N57wveG3igCSfI")
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. Load the Memory
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

def ask_bot(question):
    # Retrieve context
    docs = vector_db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    # THE UPDATED PERSONA: Friendly, helpful, and detailed
    prompt = f"""
    You are the friendly and expert face of TheHouseKraft. 
    Your goal is to make the user feel welcome and excited about their home project.
    
    GUIDELINES:
    - Only say "Welcome to HouseKraft" if this is the very first message.
    - Sound warm, inviting, and professional. 
    - Use "we" and "our" (e.g., "Our team loves working on...")
    - Provide a bit of extra detail or helpful context for every answer.
    - Never mention that you are reading from a file or context.
    - If information is missing, say: "That's a great question! I'd want to make sure I give you the perfect answer, so it's best to chat with one of our specialists for that specific detail."
    
    INTERNAL KNOWLEDGE:
    {context}
    
    USER QUESTION: 
    {question}
    """
    
    # 3. STREAMING LOGIC: This makes it feel much faster
    responses = model.generate_content(prompt, stream=True)
    
    print("\nTheHouseKraft: ", end="")
    for chunk in responses:
        for char in chunk.text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01) # Subtle delay for a "typing" feel
    print("\n")

# 4. Chat Interface
print("\n🏠 Welcome to HouseKraft! How can I help you today?")
print("(Type 'exit' to quit)\n")


while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    try:
        ask_bot(user_input)
    except Exception as e:
        print(f"\n❌ Oops! Something went wrong: {e}\n")