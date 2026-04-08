# chunk_docs.py
# PURPOSE: Split loaded documents into overlapping chunks for embedding

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(pdf_path: str):
    """Load PDF and return list of Document objects."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from: {pdf_path}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents   : List of LangChain Document objects
        chunk_size  : Max characters per chunk (tune this later)
        chunk_overlap: Overlapping characters between chunks
    
    Returns:
        List of chunked Document objects
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,

        # It tries to split on these separators IN ORDER
        # i.e., tries paragraph → sentence → word → character
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks


def preview_chunks(chunks, num_preview=5):
    """Preview first N chunks."""
    print("\n" + "="*60)
    print("🔪 CHUNK PREVIEW")
    print("="*60)

    for i, chunk in enumerate(chunks[:num_preview]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content:\n{chunk.page_content}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")
        print("-"*40)

    if len(chunks) > num_preview:
        print(f"\n... and {len(chunks) - num_preview} more chunks.")


# ─── Run directly to test ───
if __name__ == "__main__":

    PDF_PATH = "docs/hr_policy.pdf"

    # Step 1: Load
    docs = load_documents(PDF_PATH)

    # Step 2: Chunk
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)

    # Step 3: Preview
    preview_chunks(chunks, num_preview=5)

    # Summary
    print("\n" + "="*60)
    print("📊 CHUNKING SUMMARY")
    print("="*60)
    print(f"Total pages loaded     : {len(docs)}")
    print(f"Total chunks created   : {len(chunks)}")
    print(f"Chunk size setting     : 500 characters")
    print(f"Overlap setting        : 100 characters")
    avg = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"Average chunk size     : {avg:.0f} characters")
    print("\n✅ Chunking complete! Ready for embedding in Step 4.")