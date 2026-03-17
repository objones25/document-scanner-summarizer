"""Tests for src/api.py — TestClient, sessions mocked, no real API calls."""

import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import app, sessions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient; lifespan runs on enter/exit."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_sessions():
    sessions._store.clear()
    yield
    sessions._store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse_tokens(text: str) -> list[str]:
    return [
        line[6:]
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _mock_summarizer(tokens: list[str] | None = None) -> MagicMock:
    mock = MagicMock()
    tokens = tokens or ["Hello", " world"]
    mock.ask.side_effect = lambda *a, **kw: iter(tokens)
    mock.summarize.side_effect = lambda *a, **kw: iter(tokens)
    return mock


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_session_count_in_response(self, client):
        assert client.get("/health").json()["sessions"] == 0
        sessions.create("sid1", _mock_summarizer())
        assert client.get("/health").json()["sessions"] == 1


# ---------------------------------------------------------------------------
# TestCreateSession
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_file_upload_txt(self, client):
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value="Document content"):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("test.txt", b"content", "text/plain")},
                )
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["char_count"] == len("Document content")
        assert "preview" in data

    def test_file_upload_pdf_suffix(self, client):
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value="PDF content"):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("doc.pdf", b"%PDF fake", "application/pdf")},
                )
        assert response.status_code == 201

    def test_url_source(self, client):
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value="Web page content"):
                response = client.post(
                    "/api/sessions",
                    data={"url": "https://example.com"},
                )
        assert response.status_code == 201
        assert response.json()["char_count"] == len("Web page content")

    def test_url_extraction_error_returns_500(self, client):
        with patch("src.api.extract_text", side_effect=RuntimeError("Connection failed")):
            response = client.post(
                "/api/sessions",
                data={"url": "https://example.com"},
            )
        assert response.status_code == 500

    def test_missing_both_returns_400(self, client):
        response = client.post("/api/sessions", data={})
        assert response.status_code == 400

    def test_both_file_and_url_returns_400(self, client):
        response = client.post(
            "/api/sessions",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"url": "https://example.com"},
        )
        assert response.status_code == 400

    def test_empty_text_returns_422(self, client):
        with patch("src.api.extract_text", return_value="   "):
            response = client.post(
                "/api/sessions",
                files={"file": ("test.txt", b"content", "text/plain")},
            )
        assert response.status_code == 422

    def test_mistral_no_key_returns_400(self, client, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        response = client.post(
            "/api/sessions",
            files={"file": ("scan.pdf", b"fake pdf", "application/pdf")},
            data={"ocr_engine": "mistral"},
        )
        assert response.status_code == 400

    def test_preview_truncated_at_200_chars(self, client):
        long_text = "A" * 500
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value=long_text):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("test.txt", b"content", "text/plain")},
                )
        assert response.status_code == 201
        assert len(response.json()["preview"]) == 200

    def test_tempfile_deleted_on_extract_error(self, client):
        deleted = []
        real_unlink = __import__("os").unlink

        def track_unlink(path):
            deleted.append(path)

        with patch("src.api.extract_text", side_effect=RuntimeError("OCR error")):
            with patch("src.api.os.unlink", side_effect=track_unlink):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("test.txt", b"content", "text/plain")},
                )
        assert len(deleted) == 1
        assert response.status_code == 500

    def test_gemini_provider(self, client):
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer) as mock_create:
            with patch("src.api.extract_text", return_value="Content"):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("test.txt", b"content", "text/plain")},
                    data={"provider": "gemini"},
                )
        assert response.status_code == 201
        mock_create.assert_called_once_with("Content", provider_name="gemini")

    def test_provider_key_missing_returns_500(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("src.api.extract_text", return_value="Document text"):
            response = client.post(
                "/api/sessions",
                files={"file": ("test.txt", b"content", "text/plain")},
                data={"provider": "anthropic"},
            )
        assert response.status_code == 500

    def test_mistral_ocr_with_key(self, client, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value="OCR content"):
                with patch("src.api.MistralOCR") as mock_mistral_cls:
                    mock_mistral_cls.return_value = MagicMock()
                    response = client.post(
                        "/api/sessions",
                        files={"file": ("scan.pdf", b"fake pdf", "application/pdf")},
                        data={"ocr_engine": "mistral"},
                    )
        assert response.status_code == 201

    def test_session_stored_after_creation(self, client):
        mock_summarizer = _mock_summarizer()
        with patch("src.api.create_summarizer", return_value=mock_summarizer):
            with patch("src.api.extract_text", return_value="Content"):
                response = client.post(
                    "/api/sessions",
                    files={"file": ("test.txt", b"content", "text/plain")},
                )
        session_id = response.json()["session_id"]
        assert sessions.get(session_id) is not None


# ---------------------------------------------------------------------------
# TestChat
# ---------------------------------------------------------------------------

class TestChat:
    def test_sse_tokens_streamed(self, client):
        mock_summarizer = _mock_summarizer(["Hello", " world"])
        sessions.create("chat-session", mock_summarizer)

        response = client.post(
            "/api/sessions/chat-session/chat",
            json={"message": "Tell me about this"},
        )
        assert response.status_code == 200
        tokens = parse_sse_tokens(response.text)
        assert tokens == ["Hello", " world"]

    def test_done_sentinel_present(self, client):
        mock_summarizer = _mock_summarizer(["chunk"])
        sessions.create("chat-done", mock_summarizer)

        response = client.post(
            "/api/sessions/chat-done/chat",
            json={"message": "Question"},
        )
        assert "data: [DONE]" in response.text

    def test_newline_in_token_escaped(self, client):
        mock_summarizer = _mock_summarizer(["line1\nline2"])
        sessions.create("chat-newline", mock_summarizer)

        response = client.post(
            "/api/sessions/chat-newline/chat",
            json={"message": "Question"},
        )
        # The \n should be escaped to \\n in the SSE stream
        assert "line1\\nline2" in response.text

    def test_404_on_unknown_session(self, client):
        response = client.post(
            "/api/sessions/nonexistent-id/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 404

    def test_422_on_empty_message(self, client):
        sessions.create("chat-empty", _mock_summarizer())
        response = client.post(
            "/api/sessions/chat-empty/chat",
            json={"message": ""},
        )
        assert response.status_code == 422

    def test_422_on_missing_message(self, client):
        sessions.create("chat-missing", _mock_summarizer())
        response = client.post(
            "/api/sessions/chat-missing/chat",
            json={},
        )
        assert response.status_code == 422

    def test_sse_error_event_on_provider_exception(self, client):
        def _failing_ask(*args, **kw):
            raise Exception("Provider failed")
            yield  # make this a generator function

        mock_summarizer = MagicMock()
        mock_summarizer.ask.side_effect = _failing_ask
        sessions.create("chat-error", mock_summarizer)

        response = client.post(
            "/api/sessions/chat-error/chat",
            json={"message": "Question"},
        )
        assert response.status_code == 200
        assert "event: error" in response.text
        assert "Provider failed" in response.text


# ---------------------------------------------------------------------------
# TestSummary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_concise_style(self, client):
        mock_summarizer = _mock_summarizer(["Summary text"])
        sessions.create("sum-concise", mock_summarizer)

        response = client.post(
            "/api/sessions/sum-concise/summary",
            json={"style": "concise"},
        )
        assert response.status_code == 200
        tokens = parse_sse_tokens(response.text)
        assert tokens == ["Summary text"]

    def test_detailed_style(self, client):
        mock_summarizer = _mock_summarizer(["Detailed text"])
        sessions.create("sum-detailed", mock_summarizer)

        response = client.post(
            "/api/sessions/sum-detailed/summary",
            json={"style": "detailed"},
        )
        assert response.status_code == 200

    def test_bullet_points_style(self, client):
        mock_summarizer = _mock_summarizer(["• Point"])
        sessions.create("sum-bullets", mock_summarizer)

        response = client.post(
            "/api/sessions/sum-bullets/summary",
            json={"style": "bullet-points"},
        )
        assert response.status_code == 200

    def test_default_style_is_concise(self, client):
        mock_summarizer = _mock_summarizer(["Summary"])
        sessions.create("sum-default", mock_summarizer)

        response = client.post(
            "/api/sessions/sum-default/summary",
            json={},
        )
        assert response.status_code == 200

    def test_404_on_unknown_session(self, client):
        response = client.post(
            "/api/sessions/nonexistent/summary",
            json={"style": "concise"},
        )
        assert response.status_code == 404

    def test_422_on_invalid_style(self, client):
        sessions.create("sum-invalid", _mock_summarizer())
        response = client.post(
            "/api/sessions/sum-invalid/summary",
            json={"style": "wrong-style"},
        )
        assert response.status_code == 422

    def test_sse_error_event_on_exception(self, client):
        def _failing_summarize(*args, **kw):
            raise Exception("Summarize failed")
            yield  # make this a generator function

        mock_summarizer = MagicMock()
        mock_summarizer.summarize.side_effect = _failing_summarize
        sessions.create("sum-error", mock_summarizer)

        response = client.post(
            "/api/sessions/sum-error/summary",
            json={"style": "concise"},
        )
        assert response.status_code == 200
        assert "event: error" in response.text


# ---------------------------------------------------------------------------
# TestDeleteSession
# ---------------------------------------------------------------------------

class TestDeleteSession:
    def test_204_on_success(self, client):
        sessions.create("del-session", _mock_summarizer())
        response = client.delete("/api/sessions/del-session")
        assert response.status_code == 204
        assert sessions.get("del-session") is None

    def test_404_on_unknown_session(self, client):
        response = client.delete("/api/sessions/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestCORS
# ---------------------------------------------------------------------------

class TestCORS:
    def test_allowed_origin_gets_header(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "https://owenbeckettjones.com"},
        )
        assert response.headers.get("access-control-allow-origin") == "https://owenbeckettjones.com"

    def test_www_subdomain_allowed(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "https://www.owenbeckettjones.com"},
        )
        assert response.headers.get("access-control-allow-origin") == "https://www.owenbeckettjones.com"

    def test_disallowed_origin_blocked(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        acao = response.headers.get("access-control-allow-origin", "")
        assert "evil.com" not in acao

    def test_localhost_3000_allowed(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_localhost_5173_allowed(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_localhost_8080_allowed(self, client):
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:8080"},
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8080"


# ---------------------------------------------------------------------------
# TestSessionStore
# ---------------------------------------------------------------------------

class TestSessionStore:
    def test_create_and_get(self):
        mock_summarizer = _mock_summarizer()
        sessions.create("test-id", mock_summarizer)
        result = sessions.get("test-id")
        assert result is mock_summarizer

    def test_get_updates_last_access(self):
        mock_summarizer = _mock_summarizer()
        sessions.create("access-id", mock_summarizer)
        original = sessions._store["access-id"].last_access
        time.sleep(0.01)
        sessions.get("access-id")
        assert sessions._store["access-id"].last_access > original

    def test_get_nonexistent_returns_none(self):
        assert sessions.get("nonexistent") is None

    def test_delete_existing_returns_true(self):
        sessions.create("del-id", _mock_summarizer())
        assert sessions.delete("del-id") is True
        assert sessions.get("del-id") is None

    def test_delete_nonexistent_returns_false(self):
        assert sessions.delete("nonexistent") is False

    def test_count(self):
        assert sessions.count() == 0
        sessions.create("c1", _mock_summarizer())
        assert sessions.count() == 1
        sessions.create("c2", _mock_summarizer())
        assert sessions.count() == 2

    def test_purge_expired_removes_stale(self):
        sessions.create("stale-id", _mock_summarizer())
        sessions._store["stale-id"].last_access = (
            datetime.now(timezone.utc) - timedelta(seconds=3600)
        )
        removed = sessions.purge_expired(ttl_seconds=1800)
        assert removed == 1
        assert sessions.get("stale-id") is None

    def test_purge_expired_keeps_recent(self):
        sessions.create("recent-id", _mock_summarizer())
        removed = sessions.purge_expired(ttl_seconds=1800)
        assert removed == 0
        assert sessions.get("recent-id") is not None

    def test_purge_expired_mixed(self):
        sessions.create("stale", _mock_summarizer())
        sessions.create("fresh", _mock_summarizer())
        sessions._store["stale"].last_access = (
            datetime.now(timezone.utc) - timedelta(seconds=7200)
        )
        removed = sessions.purge_expired(ttl_seconds=1800)
        assert removed == 1
        assert sessions.get("stale") is None
        assert sessions.get("fresh") is not None
