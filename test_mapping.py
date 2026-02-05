import pickle
import faiss

# Load FAISS
index = faiss.read_index("data/vector_store/faiss.index")

# Load chunks
with open("data/vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("FAISS vectors:", index.ntotal)
print("Chunks:", len(chunks))

# Sanity check
print("\nSample chunk:")
print(chunks[0])
