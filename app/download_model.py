from sentence_transformers import SentenceTransformer

# This will download and cache the model locally
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
print("Model downloaded successfully.")