# 📄 Document Scanner & Summarizer

A Python tool that extracts text from documents (images, PDFs, web pages, DOCX) and provides AI-powered analysis through streaming conversations. Available as both a CLI and a production REST API deployed on Railway.

**Live API:** `https://document-scanner-summarizer-production.up.railway.app`

---

## REST API

The API is the primary integration point for the website. All responses from AI endpoints are [Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) streams.

### Base URL

```
https://document-scanner-summarizer-production.up.railway.app
```

### CORS

Requests are accepted from:
- `https://owenbeckettjones.com`
- `https://www.owenbeckettjones.com`
- `http://localhost:3000`, `http://localhost:5173`, `http://localhost:8080`

All other origins are blocked.

---

### `GET /health`

Check that the service is up and see the number of active sessions.

**Response `200`**
```json
{ "status": "ok", "sessions": 3 }
```

---

### `POST /api/sessions`

Upload a document or provide a URL to create a session. Returns a `session_id` used by all subsequent calls.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `file` | file | one of file/url | — | Any supported format (see below) |
| `url` | string | one of file/url | — | Any `http(s)://` URL |
| `provider` | string | no | `anthropic` | `anthropic` or `gemini` |
| `ocr_engine` | string | no | `tesseract` | `tesseract` or `mistral` |

Providing both `file` and `url`, or neither, returns `400`.

**Response `201`**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "char_count": 4821,
  "preview": "First 200 characters of extracted text..."
}
```

**Error responses**

| Code | Reason |
|---|---|
| `400` | Neither file nor url provided, or both provided |
| `400` | `ocr_engine=mistral` but `MISTRAL_API_KEY` not set on server |
| `422` | Document was processed but no text could be extracted |
| `500` | Extraction failed or provider API key missing on server |

**Example — file upload**
```js
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("provider", "anthropic");

const res = await fetch(`${BASE_URL}/api/sessions`, {
  method: "POST",
  body: form,
  credentials: "include",
});
const { session_id, char_count, preview } = await res.json();
```

**Example — URL**
```js
const form = new FormData();
form.append("url", "https://example.com/article");
form.append("provider", "anthropic");

const res = await fetch(`${BASE_URL}/api/sessions`, {
  method: "POST",
  body: form,
  credentials: "include",
});
```

---

### `POST /api/sessions/{session_id}/chat`

Send a message and stream the AI response back token-by-token.

**Content-Type:** `application/json`

```json
{ "message": "What are the main conclusions?" }
```

**Response `200` — SSE stream**

```
data: The\n\n
data: main\n\n
data: conclusions\n\n
data: are...\n\n
data: [DONE]\n\n
```

On error:
```
event: error
data: {"detail": "Provider error message"}
```

**Parsing SSE in JavaScript**
```js
const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "What are the main conclusions?" }),
  credentials: "include",
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split("\n");
  buffer = lines.pop(); // keep incomplete line

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const token = line.slice(6);
      if (token === "[DONE]") break;
      // tokens have literal \n escaped — unescape if rendering as text
      outputEl.textContent += token.replace(/\\n/g, "\n");
    }
    if (line.startsWith("event: error")) {
      // next line will be: data: {"detail": "..."}
    }
  }
}
```

**Error responses**

| Code | Reason |
|---|---|
| `404` | Session not found or expired |
| `422` | Empty message |

---

### `POST /api/sessions/{session_id}/summary`

Generate a one-shot summary of the document. Streams back the same SSE format as `/chat`.

**Content-Type:** `application/json`

```json
{ "style": "concise" }
```

| `style` | Description |
|---|---|
| `concise` | 2–4 paragraph overview (default) |
| `detailed` | Comprehensive summary with sections |
| `bullet-points` | Structured bullet-point list |

**Response `200` — SSE stream** (same format as `/chat`)

**Example**
```js
const res = await fetch(`${BASE_URL}/api/sessions/${sessionId}/summary`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ style: "bullet-points" }),
  credentials: "include",
});
// consume SSE stream same as /chat
```

**Error responses**

| Code | Reason |
|---|---|
| `404` | Session not found or expired |
| `422` | Invalid `style` value |

---

### `DELETE /api/sessions/{session_id}`

Explicitly end a session and free server memory. Sessions also expire automatically after **30 minutes of inactivity**.

**Response `204`** — no body

**Response `404`** — session not found

```js
await fetch(`${BASE_URL}/api/sessions/${sessionId}`, {
  method: "DELETE",
  credentials: "include",
});
```

---

### Session lifecycle

```
POST /api/sessions                     →  session_id (valid 30 min from last use)
POST /api/sessions/:id/summary         →  stream one-shot summary
POST /api/sessions/:id/chat            →  stream answer  (repeatable, builds history)
DELETE /api/sessions/:id               →  cleanup
```

Each chat message is appended to the conversation history, so follow-up questions have full context.

---

### Supported file formats

| Category | Formats |
|---|---|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif` |
| PDF | `.pdf` (text extraction + OCR fallback for scanned docs) |
| Word | `.docx` |
| Text | `.txt`, `.md`, `.markdown` |
| Web | any `http(s)://` URL |

---

## CLI

### Installation

```bash
git clone https://github.com/objones25/document-scanner-summarizer.git
cd document_scanner_summarizer
uv sync
cp .env.example .env   # add API keys
```

### Quick start

```bash
# Interactive mode
python main.py

# Analyse a file directly
python main.py document.pdf --provider anthropic

# Quick bullet-point summary
python main.py report.pdf --summary-only --summary-style bullet-points

# OCR a handwritten image with Mistral
python main.py notes.jpg --ocr mistral --provider anthropic

# Claude with web search + code execution
python main.py data.pdf --provider anthropic --web-search --code-execution

# Gemini with thinking + grounding
python main.py research.pdf --provider gemini --thinking --grounding
```

### API keys

```bash
# At least one AI provider required
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional — only needed for Mistral OCR
MISTRAL_API_KEY=...
```

### Interactive commands

| Command | Action |
|---|---|
| `/summary` | Concise summary |
| `/summary detailed` | Detailed summary |
| `/summary bullet-points` | Bullet-point summary |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

### All CLI flags

```
positional arguments:
  source                File path or URL

optional arguments:
  --ocr {tesseract,mistral}        OCR engine (default: tesseract)
  --provider {anthropic,openai,gemini}  AI provider (default: anthropic)
  --summary-only                   Print summary and exit
  --summary-style {concise,detailed,bullet-points}
  --thinking                       Extended reasoning (Gemini)
  --grounding                      Google Search grounding (Gemini)
  --code-execution                 Code execution sandbox (Gemini + Claude)
  --web-search                     Web search with citations (Claude)
  --web-fetch                      Fetch web pages/PDFs (Claude)
```

---

## Development

### Run tests

```bash
uv sync --all-groups
uv run pytest --cov=src --cov-fail-under=98 -m "not slow" -v
```

323 tests, 99.90% coverage enforced in CI.

### Project structure

```
document_scanner_summarizer/
├── src/
│   ├── api.py              # FastAPI REST API (Railway deployment)
│   ├── cli.py              # Interactive CLI
│   ├── extractors.py       # Text extraction (PDF, DOCX, URL, images)
│   ├── ocr.py              # Tesseract + Mistral OCR engines
│   ├── preprocessing.py    # Image preprocessing pipeline
│   └── summarizer.py       # AI providers + conversation history
├── tests/                  # 323 tests, 99.90% coverage
├── Dockerfile              # Railway container build
├── railway.toml            # Railway deployment config
├── .github/workflows/ci.yml
├── pyproject.toml
└── uv.lock
```

### Docker (local)

```bash
docker build -t doc-scanner .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=... \
  -e GOOGLE_API_KEY=... \
  doc-scanner
```

---

## License

MIT
