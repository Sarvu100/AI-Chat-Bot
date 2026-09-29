# Agentic AI eBook RAG Chatbot

A Python retrieval-augmented generation chatbot that answers questions using passages retrieved from the provided Agentic AI eBook. It uses OpenAI embeddings and chat completions, a Pinecone vector index, LangGraph orchestration, and a FastAPI API. Questions not supported by sufficiently relevant retrieved passages receive a refusal rather than an answer from general model knowledge.

## Architecture

1. **Ingestion (`src/ingestion.py`)** downloads the assignment PDF from Google Drive (or uses an existing local PDF), extracts pages with `PyPDFLoader`, splits them into 1,000-character chunks with 200-character overlap, embeds them using `text-embedding-3-small`, and upserts them into a cosine Pinecone index (dimension 1536). Chunk IDs are deterministic, so re-ingesting the same content updates existing vectors.
2. **RAG workflow (`src/graph.py`)** implements `START -> retrieve -> generate -> END` with LangGraph. It retrieves the top four passages, uses the highest relevance score as a retrieval confidence indicator, and refuses before calling the LLM if the score is below the configured threshold. The generation prompt restricts answers to the provided context.
3. **API (`app.py`)** exposes `GET /health` and `POST /chat`. Chat responses include the answer, retrieved passage text and metadata with per-passage relevance, and the best retrieval relevance score.
4. **Tests (`tests/`)** use fake retrieval and LLM implementations; no API keys or external services are required to run the tests.

> `confidence_score` is a retrieval relevance score, not a calibrated probability that the answer is correct. Tune `MIN_RELEVANCE_SCORE` against the actual PDF and your Pinecone results.

## Requirements

- Python 3.10–3.13 (the current `langchain-pinecone` release does not support Python 3.14 yet)
- OpenAI API key
- Pinecone API key and a Pinecone account with serverless-index support
- Internet access for downloading the PDF and calling the APIs

## Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `PINECONE_API_KEY`. Do not commit `.env`; it is ignored by Git. The remaining values in `.env.example` are optional defaults and can be changed to match your Pinecone region or model preferences.

## Ingest the eBook

With the virtual environment activated and `.env` configured:

```powershell
python -m src.ingestion
```

This downloads the assignment PDF into `data/Ebook-Agentic-AI.pdf` if it is not present, creates the Pinecone index when needed, then indexes the extracted chunks. The PDF is excluded from Git. To use a local copy instead, place it at the configured `PDF_PATH`, or invoke `ingest_pdf(Path("path/to/file.pdf"))` from Python.

## Run the API

```powershell
uvicorn app:app --reload
```

- Health check: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- Chat: `POST http://127.0.0.1:8000/chat` with JSON such as `{"query":"What is Agentic AI according to the eBook?"}`

Example PowerShell request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType 'application/json' -Body '{"query":"What is Agentic AI according to the eBook?"}'
```

The response shape is:

```json
{
  "answer": "...",
  "retrieved_chunks": [
    {
      "text": "...",
      "metadata": {"source": "...", "page": 0},
      "relevance_score": 0.82
    }
  ],
  "confidence_score": 0.82
}
```

## Tests

```powershell
pytest -q
```

Tests are offline and credential-free. They cover grounded response flow, empty retrieval, and refusal for low-relevance matches.

## Benchmark questions

Run these against `/chat` after ingestion. Inspect both the returned passages and answer; a good answer should be traceable to the displayed context.

| Query | Expected verification |
| --- | --- |
| What is Agentic AI according to the eBook? | Answer should reflect the retrieved definition. |
| How do AI agents differ from traditional automation systems? | Answer should be grounded in passages describing agent behavior and automation. |
| What are the core components of an Agentic Architecture? | Answer should list only components supported by the eBook. |
| What role does memory play in Agentic AI workflows? | Answer should describe only memory functions found in retrieved passages. |
| Who won the 2022 FIFA World Cup? | Should refuse if the eBook context does not support the answer. |
| What is the capital of France? | Should refuse when no relevant eBook context is retrieved. |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | OpenAI API authentication |
| `PINECONE_API_KEY` | Required | Pinecone authentication |
| `PINECONE_INDEX_NAME` | `agentic-ai-ebook` | Pinecone index name |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Serverless index location |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Answer-generation model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536-dimensional embeddings |
| `TOP_K` | `4` | Number of passages retrieved |
| `MIN_RELEVANCE_SCORE` | `0.25` | Minimum relevance required before generation |
| `PDF_PATH` | `data/Ebook-Agentic-AI.pdf` | Local PDF path |
| `GOOGLE_DRIVE_FILE_ID` | Assignment PDF ID | Source document for download |

## Limitations

- PDF text extraction quality depends on the source PDF; scanned pages may need OCR, which is not included.
- The configured refusal threshold is a simple retrieval guardrail, not a guarantee against every unsupported model claim. Review outputs and tune the threshold for the source document.
- Pinecone and OpenAI incur usage according to their account pricing and limits.
