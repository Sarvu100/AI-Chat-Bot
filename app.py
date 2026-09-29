"""FastAPI interface for the Agentic AI eBook RAG chatbot."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.graph import create_rag_graph

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description="Answers questions using retrieved passages from the Agentic AI eBook.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, description="Question about the eBook")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be blank.")
        return value


class RetrievedChunkResponse(BaseModel):
    text: str
    metadata: dict[str, Any]
    relevance_score: float = Field(ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    confidence_score: float = Field(ge=0.0, le=1.0, description="Best retrieval relevance, not a calibrated probability")


@lru_cache(maxsize=1)
def get_graph():
    return create_rag_graph()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = get_graph().invoke({"question": request.query.strip()})
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG service unavailable. Check API credentials and the Pinecone index.",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=result.get("retrieved_chunks", []),
        confidence_score=result.get("confidence_score", 0.0),
    )
