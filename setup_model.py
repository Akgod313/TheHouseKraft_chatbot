from sentence_transformers import SentenceTransformer
# This downloads the model to your disk
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./hf_model')
print("Model saved to ./hf_model folder!")