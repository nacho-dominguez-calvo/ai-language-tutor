# app/rag_chain.py

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a helpful AI assistant. You should answer concisely and short.
Use only the information from the provided context.
If you don't know the answer, say you don't know.
If there is a clash between the context and your prior knowledge, trust the context but mention the clash."""

def ask_with_context(question: str, vectorstore):
    """
    Ask a question using the provided vector store.
    
    Args:
        question: User's question
        vectorstore: Chroma vector store containing documents
    
    Returns:
        tuple: (answer, sources)
    """
    if vectorstore is None:
        return "Please upload documents first.", []
    
    # Get retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create prompt template with system prompt embedded
    prompt_template = PromptTemplate(
        template=f"""{SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create RAG chain with source documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context"
        }
    )
    
    # Execute query using invoke() instead of __call__
    result = qa_chain.invoke({"query": question})
    
    # Extract answer and sources
    answer = result.get("result", "No answer available.")
    source_docs = result.get("source_documents", [])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
    
    return answer, sources
