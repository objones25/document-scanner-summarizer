"""
FastAPI REST API for document scanner and summarizer.

Exposes document upload, summarization, and chat over HTTP with
Server-Sent Events streaming. Designed for deployment on Railway.
"""

import asyncio
import json
import os
import secrets
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .extractors import extract_text
from .ocr import MistralOCR, TesseractOCR
from .summarizer import DocumentSummarizer, create_summarizer


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

@dataclass
class SessionEntry:
    summarizer: DocumentSummarizer
    last_access: datetime


class SessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._store: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str, summarizer: DocumentSummarizer) -> None:
        with self._lock:
            self._store[session_id] = SessionEntry(
                summarizer=summarizer,
                last_access=datetime.now(timezone.utc),
            )

    def get(self, session_id: str) -> DocumentSummarizer | None:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            entry.last_access = datetime.now(timezone.utc)
            return entry.summarizer

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def purge_expired(self, ttl_seconds: int = 1800) -> int:
        """Remove sessions not accessed within ttl_seconds. Returns count removed."""
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [
                sid
                for sid, entry in self._store.items()
                if (now - entry.last_access).total_seconds() > ttl_seconds
            ]
            for sid in expired:
                del self._store[sid]
            return len(expired)


sessions = SessionStore()


# ---------------------------------------------------------------------------
# App lifespan (background cleanup task)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _cleanup_loop() -> None:  # pragma: no cover
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            sessions.purge_expired()

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

ALLOWED_ORIGINS = [
    "https://owenbeckettjones.com",
    "https://www.owenbeckettjones.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_credentials=True,
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_TOKEN: str | None = os.getenv("API_TOKEN")


def verify_token(authorization: str | None = Header(None)) -> None:
    """Require 'Authorization: Bearer <API_TOKEN>' when API_TOKEN is set."""
    if not _API_TOKEN:
        return  # token not configured — open access (local dev)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not secrets.compare_digest(authorization[7:], _API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionCreateResponse(BaseModel):
    session_id: str
    char_count: int
    preview: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class SummaryRequest(BaseModel):
    style: Literal["concise", "detailed", "bullet-points"] = "concise"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_stream(gen: Iterator[str]) -> Iterator[str]:
    """Wrap a text iterator as an SSE stream."""
    try:
        for chunk in gen:
            escaped = chunk.replace("\n", "\\n")
            yield f"data: {escaped}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = json.dumps({"detail": str(exc)})
        yield f"event: error\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sessions": sessions.count()}


@app.post("/api/sessions", status_code=201)
async def create_session(
    provider: Literal["anthropic", "gemini"] = Form("anthropic"),
    ocr_engine: Literal["tesseract", "mistral"] = Form("tesseract"),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    _: None = Depends(verify_token),
) -> JSONResponse:
    has_file = file is not None
    has_url = url is not None and url.strip() != ""

    if not has_file and not has_url:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")
    if has_file and has_url:
        raise HTTPException(status_code=400, detail="Only one of file or url may be provided")

    if ocr_engine == "mistral" and not os.getenv("MISTRAL_API_KEY"):
        raise HTTPException(status_code=400, detail="MISTRAL_API_KEY not set")

    if has_file:
        suffix = Path(file.filename).suffix if file.filename else ""
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            content = await file.read()
            tmp.write(content)
            tmp.close()

            if ocr_engine == "mistral":
                ocr: TesseractOCR | MistralOCR = MistralOCR(
                    api_key=os.getenv("MISTRAL_API_KEY", "")
                )
            else:
                ocr = TesseractOCR()

            try:
                text = extract_text(tmp.name, ocr_engine=ocr)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
        finally:
            os.unlink(tmp.name)
    else:
        try:
            text = extract_text(url.strip())  # type: ignore[union-attr]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from document")

    try:
        summarizer = create_summarizer(text, provider_name=provider)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    session_id = str(uuid.uuid4())
    sessions.create(session_id, summarizer)

    return JSONResponse(
        status_code=201,
        content={
            "session_id": session_id,
            "char_count": len(text),
            "preview": text[:200],
        },
    )


@app.post("/api/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatRequest, _: None = Depends(verify_token)) -> StreamingResponse:
    summarizer = sessions.get(session_id)
    if summarizer is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def generate() -> Iterator[str]:
        yield from _sse_stream(summarizer.ask(request.message))

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/sessions/{session_id}/summary")
async def summary(session_id: str, request: SummaryRequest, _: None = Depends(verify_token)) -> StreamingResponse:
    summarizer = sessions.get(session_id)
    if summarizer is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def generate() -> Iterator[str]:
        yield from _sse_stream(summarizer.summarize(style=request.style))

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, _: None = Depends(verify_token)) -> Response:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return Response(status_code=204)
