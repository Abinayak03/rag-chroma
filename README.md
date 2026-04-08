# rag-chroma
# 🤖 HR Policy Q&A Chatbot

A RAG-based chatbot that answers employee questions from an HR policy PDF.

## Tech Stack
- 🔗 LangChain — RAG pipeline
- 🗄️ ChromaDB — Vector storage  
- 🧠 HuggingFace — Embeddings (all-MiniLM-L6-v2)
- ⚡ Groq — LLM (Llama 3.1)
- 🎨 Streamlit — Chat UI

## How to Run Locally

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`:
GROQ_API_KEY=your_key_here
4. Ingest the PDF: `python ingest.py`
5. Run the app: `streamlit run app.py`

## Features
- 📄 Answers from real HR policy PDF
- 📚 Source citations with page numbers
- 💬 Chat history
- 🔍 Semantic search via ChromaDB
