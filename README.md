# 📄 Document Scanner & Summarizer

A powerful Python CLI tool that extracts text from various document sources (images, PDFs, web pages) and provides AI-powered analysis through interactive conversations with streaming responses.

## ✨ Features

### Document Extraction

- **Multi-format support**: Images (JPG, PNG, TIFF), PDFs, DOCX files, and web URLs
- **Dual OCR engines**:
  - Tesseract (free, local, great for printed text)
  - Mistral OCR (API-based, excellent for handwriting and complex layouts)
- **Smart preprocessing**: Automatic image enhancement for optimal OCR accuracy
- **PDF handling**: Text extraction with OCR fallback for scanned documents
- **Web scraping**: Intelligent article extraction from URLs

### AI-Powered Analysis

- **Multiple AI providers**:
  - Anthropic Claude (claude-sonnet-4-5-20250929)
  - OpenAI GPT (gpt-5.1-2025-11-13)
  - Google Gemini (gemini-3-pro-preview)
- **Streaming responses**: Real-time output as AI generates text
- **Interactive chat**: Natural conversation interface for document analysis
- **Context-aware**: Multiturn conversations with full conversation history
- **Multiple summary styles**: Concise, detailed, or bullet-point summaries

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/objones25/document-scanner-summarizer.git
cd document_scanner_summarizer

# Install dependencies with uv
uv sync

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys (at least one AI provider required)
```

### Required API Keys

Add to your `.env` file:

```bash
# AI Analysis (choose at least one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# OCR (optional - only for Mistral OCR)
MISTRAL_API_KEY=...
```

## 💻 Usage

### Interactive Mode (Recommended)

```bash
python main.py
```

The CLI will guide you through:

1. 📄 Select document source (file path or URL)
2. 🔍 Choose OCR engine (Tesseract or Mistral)
3. 🤖 Select AI provider (Anthropic, OpenAI, or Gemini)
4. 💬 Start interactive conversation

### Command Line Mode

```bash
# Analyze a specific file
python main.py document.pdf

# Analyze a web page
python main.py https://example.com/article

# Analyze an image with Mistral OCR and Anthropic
python main.py workout_notes.jpg --ocr mistral --provider anthropic

# Get a quick summary and exit
python main.py document.pdf --summary-only

# Get detailed bullet-point summary
python main.py document.pdf --summary-only --summary-style bullet-points

# Use specific providers
python main.py document.pdf --provider openai
```

### Command Line Arguments

```text
positional arguments:
  source                Document source (file path or URL)

optional arguments:
  -h, --help           Show help message
  --ocr {tesseract,mistral}
                       OCR engine to use (default: tesseract)
  --provider {anthropic,openai,gemini}
                       AI provider for analysis (default: anthropic)
  --summary-only       Generate summary and exit (no interactive mode)
  --summary-style {concise,detailed,bullet-points}
                       Summary style (default: concise)
```

## 💬 Interactive Commands

Once in interactive mode, you can use these commands:

### Ask Questions

Just type your question and press Enter:

```text
You: What are the main conclusions?
You: Can you explain the methodology in detail?
You: Who are the key people mentioned?
You: Could you list all the exercises in the workout?
```

### Get Summaries

Use the `/summary` command with optional style:

```text
You: /summary                    # Concise summary (2-4 paragraphs)
You: /summary detailed           # Comprehensive summary with sections
You: /summary bullet-points      # Structured bullet-point summary
```

### Manage Conversation

```text
You: /clear                      # Clear conversation history
You: /exit                       # Exit the program
```

**Tip**: Press `Ctrl+C` or `Ctrl+D` to exit anytime.

## 📖 Example Workflows

### Example 1: Analyze a Research Paper

```bash
python main.py research_paper.pdf --provider anthropic
```

Interactive session:

```text
You: /summary detailed
🤖 AI: This paper investigates...

You: What methodology did they use?
🤖 AI: The researchers employed a mixed-methods approach...

You: What were the main limitations?
🤖 AI: The study has three primary limitations...
```

### Example 2: OCR Handwritten Workout Notes

```bash
python main.py workout_notes.jpg --ocr mistral --provider anthropic
```

```text
You: Could you list all the exercises?
🤖 AI:
Warm-Up:
1. Uphill walk on treadmill...

Strength Set 1 (2 rounds):
1. Keiser high to low chops × 10...
[continues with full exercise list]
```

### Example 3: Quick Web Article Summary

```bash
python main.py https://example.com/tech-article --summary-only --summary-style bullet-points
```

Output:

```text
📝 Generating bullet-points summary...

- Main topic: New developments in AI reasoning systems
- Key points:
  • Breakthrough in multi-step reasoning capabilities
  • 40% improvement in complex problem-solving tasks
  • New training methodology using reinforcement learning
- Conclusion: Significant step toward more capable AI systems
```

### Example 4: Analyze Meeting Notes from DOCX

```bash
python main.py meeting_notes.docx --provider openai
```

```text
You: What action items were mentioned?
🤖 AI: Based on the meeting notes, here are the action items...

You: Who is responsible for the Q2 budget report?
🤖 AI: According to the notes, Sarah Johnson is responsible...
```

## 📁 Supported File Types

| Category | Formats | Method |
|----------|---------|--------|
| **Images** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif` | OCR (Tesseract or Mistral) |
| **Documents** | `.pdf` | Text extraction + OCR fallback |
| **Documents** | `.docx` | Direct text extraction |
| **Web** | `http://`, `https://` | Web scraping + article parsing |

## 🏗️ Project Structure

```text
document_scanner_summarizer/
├── main.py                    # CLI entry point
├── src/
│   ├── cli.py                # Interactive CLI interface
│   ├── extractors.py         # Universal text extraction
│   ├── ocr.py                # OCR engines (Tesseract, Mistral)
│   ├── preprocessing.py      # Image preprocessing pipeline
│   └── summarizer.py         # AI conversation & analysis
├── tests/
│   ├── test_ocr.py           # OCR unit tests
│   ├── test_ocr_real_images.py # Integration tests
│   ├── test_extractors.py   # Extraction tests
│   └── test_preprocessing.py # Preprocessing tests
├── pyproject.toml            # Project dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extractors.py

# Run with coverage
pytest --cov=src tests/

# Skip slow integration tests
pytest -m "not slow"
```

**Test coverage:**

- ✅ 129+ tests across all modules
- ✅ Unit tests with mocking
- ✅ Integration tests with real OCR
- ✅ Type checking with proper annotations

## 🔧 Advanced Usage

### Use as a Library

```python
from src.extractors import extract_text
from src.summarizer import create_summarizer

# Extract text
text = extract_text("document.pdf")

# Create summarizer
summarizer = create_summarizer(text, provider_name="anthropic")

# Get streaming summary
for chunk in summarizer.summarize(style="concise"):
    print(chunk, end="", flush=True)

# Ask questions
for chunk in summarizer.ask("What are the key findings?"):
    print(chunk, end="", flush=True)
```

### Custom System Prompts

```python
from src.summarizer import DocumentSummarizer, AnthropicProvider

provider = AnthropicProvider()
summarizer = DocumentSummarizer(
    provider=provider,
    document_text=text,
    system_prompt="You are a legal document analyst. Focus on contracts and terms."
)
```

### Custom Models

```python
from src.summarizer import create_summarizer

# Use a specific model version
summarizer = create_summarizer(
    document_text=text,
    provider_name="anthropic",
    model="claude-3-5-sonnet-20241022"  # Custom model
)
```

## 🛠️ Development

### Prerequisites

- Python 3.13+
- uv package manager
- Tesseract OCR (for local OCR)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/objones25/document-scanner-summarizer.git
cd document_scanner_summarizer

# Install dependencies
uv sync

# Install Tesseract (macOS)
brew install tesseract

# Run tests
pytest

# Type checking
mypy src/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/)
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract) and [Mistral AI](https://mistral.ai/)
- OpenAI and Google AI for additional LLM support
