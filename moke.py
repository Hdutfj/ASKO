import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

DATA_FILE="data.txt"
VECTOR_DIR = "vector_db"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

def create_retriever()->VectorStoreRetriever:
    """Load data.txt using TextLoader and return retriever."""
    
    loader= TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks=splitter.split_documents(documents)

    print(f"✅ Loaded and split into {len(chunks)} chunks.")

    embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_DIR):
        print(f"📦 Loading existing vectorstore from {VECTOR_DIR}")
        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🛠 Creating new vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        print("💾 Saved vectorstore at", VECTOR_DIR)

    return vectorstore.as_retriever(search_kwargs={"k": 3})
