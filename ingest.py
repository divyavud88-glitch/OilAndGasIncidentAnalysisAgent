import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def ingest_documents():
    print(f"Loading PDFs from: {DATA_PATH}")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents found in the data folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120"))
    )

    chunks = splitter.split_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"FAISS vectorstore saved to: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    ingest_documents()