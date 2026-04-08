# load_docs.py
# PURPOSE: Load PDF from docs/ folder and preview its content

from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(pdf_path: str):
    """
    Load a PDF file and return a list of Document objects.
    Each page of the PDF becomes one Document object.
    """
    print(f"\n📄 Loading PDF from: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found at: {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"✅ Successfully loaded {len(documents)} page(s) from PDF")
    return documents


def preview_documents(documents, num_chars=500):
    """
    Preview the first few characters of each page loaded.
    """
    print("\n" + "="*60)
    print("📋 DOCUMENT PREVIEW")
    print("="*60)
    
    for i, doc in enumerate(documents):
        print(f"\n--- Page {i+1} ---")
        print(f"Content Preview:\n{doc.page_content[:num_chars]}")
        print(f"\nMetadata: {doc.metadata}")
        print("-"*40)


# ─── Run this file directly to test ───
if __name__ == "__main__":
    
    # Path to your PDF
    PDF_PATH = "docs/hr_policy.pdf"
    
    # Load documents
    docs = load_documents(PDF_PATH)
    
    # Preview what was loaded
    preview_documents(docs)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Total pages loaded     : {len(docs)}")
    print(f"Total characters       : {sum(len(d.page_content) for d in docs)}")
    print(f"Source file            : {docs[0].metadata.get('source', 'N/A')}")
    print("\n✅ Document loading complete! Ready for chunking in Step 3.")