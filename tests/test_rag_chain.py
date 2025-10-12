import pytest
from unittest.mock import patch, MagicMock

# ---- Test 1: Building the RAG chain ----
@patch("app.rag_chain.RetrievalQA.from_chain_type")
@patch("app.rag_chain.get_retriever")
@patch("app.rag_chain.ChatOpenAI")
def test_build_rag_chain(mock_llm, mock_get_retriever, mock_from_chain_type):
    from app.rag_chain import build_rag_chain

    mock_get_retriever.return_value = MagicMock()
    mock_llm.return_value = MagicMock()
    mock_from_chain_type.return_value = MagicMock(invoke=lambda _: "ok")

    chain = build_rag_chain()
    assert chain is not None
    assert hasattr(chain, "invoke")


# ---- Test 2: Asking a known query ----
@patch("app.rag_chain.build_rag_chain")
def test_ask_with_context_known(mock_build_chain):
    from app.rag_chain import ask_with_context

    # Mock chain result
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Project Alpha was developed by Laura Méndez in 2024.",
        "source_documents": [
            MagicMock(metadata={"source": "data/documents/project_alpha.txt"})
        ]
    }
    mock_build_chain.return_value = mock_chain

    answer, sources = ask_with_context("Who developed Project Alpha?")
    assert "laura" in answer.lower()
    assert len(sources) == 1
    assert "project_alpha.txt" in sources[0]


# ---- Test 3: Asking something unknown ----
@patch("app.rag_chain.build_rag_chain")
def test_ask_with_context_unknown(mock_build_chain):
    from app.rag_chain import ask_with_context

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "I don't know the answer based on the provided documents.",
        "source_documents": []
    }
    mock_build_chain.return_value = mock_chain

    answer, sources = ask_with_context("What is the capital of Mars?")
    assert "don't know" in answer.lower()
    assert sources == []
