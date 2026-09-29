"""Download, parse, chunk, and index the Agentic AI reference PDF."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import gdown
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

ROOT_DIR = Path(__file__).resolve().parents[1]


def download_pdf(destination: Path | None = None) -> Path:
    """Download the assignment's Google Drive PDF if it is not already present."""
    load_dotenv(ROOT_DIR / ".env")
    output_path = destination or ROOT_DIR / os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path

    file_id = os.getenv(
        "GOOGLE_DRIVE_FILE_ID", "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
    )
    downloaded = gdown.download(id=file_id, output=str(output_path), quiet=False)
    if not downloaded or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Could not download the reference PDF from Google Drive.")
    return output_path


def ensure_pinecone_index() -> str:
    """Create the configured cosine index when it does not already exist."""
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file.")

    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")
    client = Pinecone(api_key=api_key)
    existing_indexes = client.list_indexes().names()
    if index_name not in existing_indexes:
        client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
    return index_name


def ingest_pdf(pdf_path: Path | None = None) -> int:
    """Index the PDF and return the number of chunks upserted."""
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

    source_path = pdf_path or download_pdf()
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF not found: {source_path}")

    pages = PyPDFLoader(str(source_path)).load()
    if not pages:
        raise ValueError(f"No readable pages found in {source_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text could be extracted from {source_path}")

    index_name = ensure_pinecone_index()
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    ids = [
        hashlib.sha256(
            f"{chunk.metadata.get('page', '')}:{chunk.metadata.get('start_index', '')}:"
            f"{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        for chunk in chunks
    ]
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        ids=ids,
    )
    return len(chunks)


if __name__ == "__main__":
    count = ingest_pdf()
    print(f"Indexed {count} text chunks in Pinecone.")
