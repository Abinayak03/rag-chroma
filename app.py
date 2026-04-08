# app.py
# PURPOSE: Streamlit Chat UI for HR Policy RAG Chatbot

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────
CHROMA_DIR      = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"
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


# ── CACHED LOADERS (load only once, not on every message) ────
@st.cache_resource
def load_rag_pipeline():
    """
    Load vectorstore + LLM + RAG chain.
    @st.cache_resource ensures this runs only ONCE
    even when user sends multiple messages.
    """
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load ChromaDB
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Load Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Build retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    # Build prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Format docs helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)

# ── HEADER ────────────────────────────────────────────────────
st.title("🤖 HR Policy Q&A Assistant")
st.markdown(
    "Ask any question about the **HR Policy Document**. "
    "Answers are generated from the actual document with source references."
)
st.divider()

# ── CHECK CHROMADB EXISTS ─────────────────────────────────────
if not os.path.exists(CHROMA_DIR):
    st.error(
        "❌ ChromaDB not found! "
        "Please run `python ingest.py` first to process the PDF."
    )
    st.stop()

# ── LOAD RAG PIPELINE ─────────────────────────────────────────
with st.spinner("⚙️ Loading RAG pipeline... (first load may take ~30 seconds)"):
    rag_chain, retriever = load_rag_pipeline()

# ── CHAT HISTORY (stored in session state) ────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hello! I'm your HR Policy Assistant.\n\n"
            "You can ask me questions like:\n"
            "- 📅 How many leave days do employees get?\n"
            "- 🏠 What is the work from home policy?\n"
            "- 💰 When are salaries credited?\n"
            "- 📋 What is the PF deduction?\n\n"
            "Go ahead and ask anything!"
        ),
        "sources": []
    })

# ── DISPLAY CHAT HISTORY ──────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if available
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}**")
                    st.markdown(
                        f"- 📄 **File:** "
                        f"`{doc.metadata.get('source', 'N/A')}`"
                    )
                    st.markdown(
                        f"- 📖 **Page:** "
                        f"{doc.metadata.get('page', 'N/A')}"
                    )
                    st.markdown(
                        f"- 💬 **Excerpt:** "
                        f"_{doc.page_content[:200]}..._"
                    )
                    if i < len(message["sources"]) - 1:
                        st.divider()

# ── CHAT INPUT ────────────────────────────────────────────────
if question := st.chat_input("💬 Ask a question about the HR policy..."):

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("⏳ Searching HR policy and generating answer..."):
            try:
                # Get answer from RAG chain
                answer  = rag_chain.invoke(question)
                sources = retriever.invoke(question)

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i+1}**")
                            st.markdown(
                                f"- 📄 **File:** "
                                f"`{doc.metadata.get('source', 'N/A')}`"
                            )
                            st.markdown(
                                f"- 📖 **Page:** "
                                f"{doc.metadata.get('page', 'N/A')}"
                            )
                            st.markdown(
                                f"- 💬 **Excerpt:** "
                                f"_{doc.page_content[:200]}..._"
                            )
                            if i < len(sources) - 1:
                                st.divider()

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This chatbot answers questions from an "
        "**HR Policy PDF** using RAG (Retrieval-Augmented Generation)."
    )

    st.divider()
    st.header("🛠️ Tech Stack")
    st.markdown("""
    - 🔗 **LangChain** — RAG pipeline
    - 🗄️ **ChromaDB** — Vector storage
    - 🧠 **HuggingFace** — Embeddings
    - ⚡ **Groq** — LLM (Llama 3.1)
    - 🎨 **Streamlit** — Chat UI
    """)

    st.divider()
    st.header("⚙️ Settings")
    st.markdown(f"- **Model:** `{GROQ_MODEL}`")
    st.markdown(f"- **Chunks retrieved:** `{TOP_K_RESULTS}`")
    st.markdown(f"- **Embedding:** `{EMBEDDING_MODEL}`")

    st.divider()
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()