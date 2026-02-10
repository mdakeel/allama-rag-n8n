<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c43a68a5-039a-4b36-b5b4-4b774de59f99" />


---

## 🧠 Telegram Bot + RAG Retriever (FastAPI)

This FastAPI app powers a Retrieval-Augmented Generation (RAG) pipeline for a Telegram bot built using n8n. It receives queries, searches vectorized YouTube transcripts, and returns concise answers via LLM.

---

### 🚀 How It Works

1. **Data Source**:  
   - Transcripts extracted from an organization's YouTube channel.  
   - Converted to vector embeddings using a multilingual model.  
   - Stored in `chunks.pkl` via Faiss indexing.

2. **Query Flow**:  
   - Telegram user sends a message.  
   - n8n forwards query to FastAPI via HTTP node.  
   - FastAPI retrieves nearest chunks using vector similarity.  
   - Filtered chunks are passed to LLM for answer generation.  
   - Final answer returned to Telegram via n8n.

---

### 🛠️ Tech Stack

- **FastAPI** – API layer for retrieval  
- **Faiss** – Vector similarity search  
- **LLM (e.g., OpenAI, HuggingFace)** – Answer generation  
- **n8n** – Workflow automation  
- **Telegram API** – Bot communication  

---



---

### ▶️ Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---
