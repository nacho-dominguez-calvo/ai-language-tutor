# app/rag_chain.py
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.vector_store import get_retriever

SYSTEM_PROMPT = """You are a helpful AI assistant. You should answer concisely and short.
Use only the information from the provided context.
If you don't know the answer, say you don't know.
If there is a clash between the context and your prior knowledge, trust the context but mention the clash."""

def build_rag_chain():
    """Create the RAG pipeline: retriever + LLM."""
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = PromptTemplate(
        template="""{system_prompt}

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["system_prompt", "context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # simplest: concatenate retrieved documents
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template.partial(system_prompt=SYSTEM_PROMPT)},
        return_source_documents=True
    )
    return qa_chain

def ask_with_context(query: str):
    """Run a query through the RAG pipeline and return answer + docs."""
    qa_chain = build_rag_chain()
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    return answer, sources
