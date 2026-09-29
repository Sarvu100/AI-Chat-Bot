from types import SimpleNamespace

from src.graph import REFUSAL, build_graph


class FakeRetriever:
    def __init__(self, matches):
        self.matches = matches

    def similarity_search_with_relevance_scores(self, query, k):
        return self.matches[:k]


class FakeLLM:
    def __init__(self, content="The eBook describes agentic systems."):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.content)


def test_grounded_question_returns_answer_and_sources():
    retriever = FakeRetriever(
        [(SimpleNamespace(page_content="Agentic systems can plan and act.", metadata={"page": 2}), 0.82)]
    )
    llm = FakeLLM("Agentic systems can plan and act.")
    graph = build_graph(retriever, llm, min_score=0.25)

    result = graph.invoke({"question": "What can agentic systems do?"})

    assert result["answer"] == "Agentic systems can plan and act."
    assert result["confidence_score"] == 0.82
    assert result["retrieved_chunks"][0]["metadata"] == {"page": 2}
    assert llm.calls == 1


def test_irrelevant_retrieval_refuses_without_calling_llm():
    retriever = FakeRetriever(
        [(SimpleNamespace(page_content="Unrelated passage.", metadata={}), 0.1)]
    )
    llm = FakeLLM()
    graph = build_graph(retriever, llm, min_score=0.25)

    result = graph.invoke({"question": "Who won the 2022 FIFA World Cup?"})

    assert result["answer"] == REFUSAL
    assert result["confidence_score"] == 0.1
    assert llm.calls == 0


def test_empty_retrieval_refuses():
    graph = build_graph(FakeRetriever([]), FakeLLM())

    result = graph.invoke({"question": "What is in the eBook?"})

    assert result["answer"] == REFUSAL
    assert result["retrieved_chunks"] == []
    assert result["confidence_score"] == 0.0
