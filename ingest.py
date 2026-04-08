# ingest.py
# PURPOSE: Load PDF → Chunk → Embed → Store in ChromaDB

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── CONFIG ──────────────────────────────────────────────────
PDF_PATH      = "docs/hr_policy.pdf"
CHROMA_DIR    = "chroma_db"          # folder where ChromaDB saves data
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # free, lightweight, runs locally
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
# ────────────────────────────────────────────────────────────


def load_documents(pdf_path: str):
    """Load PDF into Document objects."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ File not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} page(s)")
    return docs


def chunk_documents(docs):
    """Split documents into overlapping chunks."""
    print(f"\n🔪 Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def create_embeddings():
    """Load the free local embedding model."""
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First run downloads ~90MB — please wait...)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded!")
    return embeddings


def store_in_chromadb(chunks, embeddings):
    """Embed all chunks and store in ChromaDB."""
    print(f"\n💾 Storing embeddings in ChromaDB at: {CHROMA_DIR}")

    # If DB already exists, delete and recreate (fresh ingest)
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("   ♻️  Cleared existing ChromaDB")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    return vectorstore


def test_retrieval(vectorstore):
    """Quick test: search ChromaDB with a sample question."""
    print("\n" + "="*60)
    print("🔍 TESTING RETRIEVAL")
    print("="*60)

    test_query = "How many leave days do employees get?"
    print(f"Query: '{test_query}'")

    results = vectorstore.similarity_search(test_query, k=3)

    print(f"\nTop {len(results)} relevant chunks found:\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Content : {doc.page_content[:200]}")
        print(f"Source  : {doc.metadata.get('source', 'N/A')}")
        print(f"Page    : {doc.metadata.get('page', 'N/A')}")
        print("-"*40)


# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: Load
    docs = load_documents(PDF_PATH)

    # Step 2: Chunk
    chunks = chunk_documents(docs)

    # Step 3: Embed model
    embeddings = create_embeddings()

    # Step 4: Store in ChromaDB
    vectorstore = store_in_chromadb(chunks, embeddings)

    # Step 5: Test it works
    test_retrieval(vectorstore)

    print("\n" + "="*60)
    print("📊 INGESTION SUMMARY")
    print("="*60)
    print(f"PDF processed          : {PDF_PATH}")
    print(f"Total chunks stored    : {len(chunks)}")
    print(f"Embedding model        : {EMBEDDING_MODEL}")
    print(f"ChromaDB saved at      : {CHROMA_DIR}/")
    print("\n✅ Ingestion complete! Ready to build RAG pipeline in Step 5.")