# rag_pipeline.py (updated for LangChain v0.2+)

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────
CHROMA_DIR      = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K_RESULTS   = 3
# ─────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """
You are a helpful HR policy assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I'm sorry, I couldn't find that information in the HR policy document."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    """Load existing ChromaDB from disk."""
    print("📦 Loading ChromaDB vectorstore...")
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError("❌ ChromaDB not found! Run ingest.py first.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print("✅ ChromaDB loaded!")
    return vectorstore


def load_llm():
    """Load Groq LLM."""
    print(f"🤖 Loading Groq LLM: {GROQ_MODEL}")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file!")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
        api_key=api_key
    )
    print("✅ Groq LLM loaded!")
    return llm


def format_docs(docs):
    """Combine retrieved chunks into single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_pipeline(vectorstore, llm):
    """Build RAG chain using modern LCEL syntax."""
    print("🔗 Building RAG pipeline...")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Modern LCEL chain: retriever → prompt → LLM → parser
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG pipeline ready!")
    return rag_chain, retriever


def ask_question(rag_chain, retriever, question: str):
    """Ask a question and return the answer with sources."""
    print(f"\n❓ Question: {question}")
    print("⏳ Thinking...")

    answer = rag_chain.invoke(question)
    sources = retriever.invoke(question)

    print(f"\n💬 Answer:\n{answer}")
    print("\n📚 Sources Used:")
    for i, doc in enumerate(sources):
        print(f"\n  Source {i+1}:")
        print(f"  Page    : {doc.metadata.get('page', 'N/A')}")
        print(f"  Content : {doc.page_content[:150]}...")

    return answer, sources


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("🚀 RAG PIPELINE TEST")
    print("="*60)

    vectorstore         = load_vectorstore()
    llm                 = load_llm()
    rag_chain, retriever = build_rag_pipeline(vectorstore, llm)

    test_questions = [
        "How many annual leave days do employees get?",
        "What is the work from home policy?",
        "When are salaries credited?",
    ]

    for question in test_questions:
        print("\n" + "─"*60)
        ask_question(rag_chain, retriever, question)

    print("\n✅ RAG Pipeline test complete! Ready for Streamlit in Step 6.")