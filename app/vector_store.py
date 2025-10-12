# app/vector_store.py

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore_from_uploaded_files(uploaded_files):
    """
    Create an in-memory Chroma vector store from uploaded Streamlit files.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
    
    Returns:
        Chroma: In-memory vector store
    """
    if not uploaded_files:
        return None
    
    # Convert uploaded files to Document objects
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Read file content
            content = uploaded_file.read().decode('utf-8')
            
            # Create Document with metadata
            doc = Document(
                page_content=content,
                metadata={"source": uploaded_file.name}
            )
            documents.append(doc)
            
            # Reset file pointer for potential re-reading
            uploaded_file.seek(0)
        except Exception as e:
            print(f"Error reading {uploaded_file.name}: {e}")
            continue
    
    if not documents:
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Create in-memory vector store (no persist_directory)
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
        # NO persist_directory = en memoria
    )
    
    return vectorstore

def get_retriever_from_vectorstore(vectorstore, k=3):
    """
    Get retriever from an existing vector store.
    
    Args:
        vectorstore: Chroma vector store
        k: Number of documents to retrieve
    
    Returns:
        VectorStoreRetriever
    """
    if vectorstore is None:
        return None
    
    return vectorstore.as_retriever(search_kwargs={"k": k})
