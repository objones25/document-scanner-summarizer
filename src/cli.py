"""
Interactive CLI for document scanning and summarization.

Allows users to:
- Upload files (images, PDFs, DOCX)
- Provide URLs for web scraping
- Extract text using OCR and other methods
- Analyze documents with AI (summarize, ask questions)
- Interactive conversation mode
"""

import sys
import os
from pathlib import Path
from typing import Literal
import argparse

from .extractors import extract_text
from .summarizer import create_summarizer
from .ocr import TesseractOCR, MistralOCR


def print_header() -> None:
    """Print CLI header."""
    print("\n" + "=" * 60)
    print("📄 Document Scanner & Summarizer")
    print("=" * 60 + "\n")


def print_separator() -> None:
    """Print section separator."""
    print("\n" + "-" * 60 + "\n")


def get_source_input() -> str:
    """
    Prompt user for document source.

    Returns:
        Path to file or URL
    """
    print("Enter document source:")
    print("  • File path (image, PDF, DOCX)")
    print("  • URL (web page)")
    print()

    while True:
        source: str = input("Source: ").strip()

        if not source:
            print("❌ Please enter a valid source\n")
            continue

        # Check if it's a URL
        if source.startswith(("http://", "https://")):
            return source

        # Check if file exists
        source_path = Path(source).expanduser()
        if not source_path.exists():
            print(f"❌ File not found: {source}\n")
            continue

        return str(source_path)


def get_ocr_engine_choice() -> TesseractOCR | MistralOCR:
    """
    Prompt user to choose OCR engine.

    Returns:
        Configured OCR engine
    """
    print("Choose OCR engine:")
    print("  1. Tesseract (free, local)")
    print("  2. Mistral OCR (paid, better for handwriting)")
    print()

    while True:
        choice: str = input("Choice [1]: ").strip() or "1"

        if choice == "1":
            return TesseractOCR()
        elif choice == "2":
            api_key: str | None = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                print("❌ MISTRAL_API_KEY not found in environment")
                print("   Please set it in your .env file or environment\n")
                continue
            return MistralOCR(api_key=api_key)
        else:
            print("❌ Invalid choice. Please enter 1 or 2\n")


def get_llm_provider_choice() -> tuple[Literal["anthropic", "openai", "gemini"], dict[str, str]]:
    """
    Prompt user to choose LLM provider.

    Returns:
        Tuple of (provider_name, provider_kwargs)
    """
    print("Choose AI provider for analysis:")
    print("  1. Anthropic Claude (recommended)")
    print("  2. OpenAI GPT")
    print("  3. Google Gemini")
    print()

    while True:
        choice: str = input("Choice [1]: ").strip() or "1"

        if choice == "1":
            api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("❌ ANTHROPIC_API_KEY not found in environment")
                print("   Please set it in your .env file or environment\n")
                continue
            return "anthropic", {}

        elif choice == "2":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY not found in environment")
                print("   Please set it in your .env file or environment\n")
                continue
            return "openai", {}

        elif choice == "3":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("❌ GOOGLE_API_KEY not found in environment")
                print("   Please set it in your .env file or environment\n")
                continue
            return "gemini", {}

        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3\n")


def extract_document_text(source: str, ocr_engine: TesseractOCR | MistralOCR) -> str:
    """
    Extract text from document source.

    Args:
        source: File path or URL
        ocr_engine: OCR engine to use for images/PDFs

    Returns:
        Extracted text
    """
    print_separator()
    print("📖 Extracting text from document...")

    try:
        text: str = extract_text(source, ocr_engine=ocr_engine)

        if not text.strip():
            print("❌ No text extracted from document")
            sys.exit(1)

        print(f"✅ Extracted {len(text)} characters")
        print(f"   Preview: {text[:200]}...")

        return text

    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        sys.exit(1)


def interactive_mode(
    document_text: str,
    provider_name: Literal["anthropic", "openai", "gemini"],
    provider_kwargs: dict[str, str]
) -> None:
    """
    Enter interactive conversation mode.

    Args:
        document_text: Extracted document text
        provider_name: LLM provider to use
        provider_kwargs: Provider configuration
    """
    print_separator()
    print("🤖 Starting AI-powered document analysis...")

    try:
        summarizer = create_summarizer(
            document_text,
            provider_name=provider_name,
            **provider_kwargs
        )
    except Exception as e:
        print(f"❌ Error initializing AI: {e}")
        sys.exit(1)

    print("✅ AI ready!\n")
    print("Commands:")
    print("  /summary [style]  - Get document summary (concise/detailed/bullet-points)")
    print("  /clear           - Clear conversation history")
    print("  /exit            - Exit program")
    print("  <question>       - Ask any question about the document")
    print()

    while True:
        try:
            user_input: str = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/exit":
                print("\n👋 Goodbye!")
                sys.exit(0)

            elif user_input.lower() == "/clear":
                summarizer.clear_history()
                print("✅ Conversation history cleared\n")
                continue

            elif user_input.lower().startswith("/summary"):
                # Parse style argument
                parts: list[str] = user_input.split(maxsplit=1)
                style: str = "concise"
                if len(parts) > 1:
                    requested_style: str = parts[1].strip().lower()
                    if requested_style in ["concise", "detailed", "bullet-points"]:
                        style = requested_style
                    else:
                        print(f"⚠️  Unknown style '{requested_style}', using 'concise'")

                print(f"\n🤖 AI (generating {style} summary):\n")
                try:
                    for chunk in summarizer.summarize(style=style):
                        print(chunk, end="", flush=True)
                    print("\n")
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                continue

            # Regular question
            print("\n🤖 AI:\n")
            try:
                for chunk in summarizer.ask(user_input):
                    print(chunk, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def main() -> None:
    """Main CLI entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Document Scanner & Summarizer - Extract and analyze documents with AI"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Document source (file path or URL). If not provided, will prompt interactively."
    )
    parser.add_argument(
        "--ocr",
        choices=["tesseract", "mistral"],
        default="tesseract",
        help="OCR engine to use (default: tesseract)"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini"],
        default="anthropic",
        help="AI provider for analysis (default: anthropic)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary and exit (no interactive mode)"
    )
    parser.add_argument(
        "--summary-style",
        choices=["concise", "detailed", "bullet-points"],
        default="concise",
        help="Summary style when using --summary-only (default: concise)"
    )

    args = parser.parse_args()

    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Print header
    print_header()

    # Get source (from args or interactive prompt)
    if args.source:
        source: str = args.source
        print(f"📄 Source: {source}")
    else:
        source = get_source_input()

    print_separator()

    # Get OCR engine (from args or interactive prompt)
    if source.startswith(("http://", "https://")):
        # URLs don't need OCR
        ocr_engine: TesseractOCR | MistralOCR = TesseractOCR()
    else:
        # Check if file might need OCR (images or PDFs)
        source_lower: str = source.lower()
        needs_ocr: bool = source_lower.endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf")
        )

        if needs_ocr:
            if len(sys.argv) > 1 and args.ocr:
                # Use OCR from args
                if args.ocr == "tesseract":
                    ocr_engine = TesseractOCR()
                else:
                    api_key: str | None = os.getenv("MISTRAL_API_KEY")
                    if not api_key:
                        print("❌ MISTRAL_API_KEY not found in environment")
                        sys.exit(1)
                    ocr_engine = MistralOCR(api_key=api_key)
            else:
                ocr_engine = get_ocr_engine_choice()
        else:
            ocr_engine = TesseractOCR()

    # Extract text
    document_text: str = extract_document_text(source, ocr_engine)

    # Get LLM provider (from args or interactive prompt)
    if len(sys.argv) > 1 and args.provider:
        # Use provider from args
        provider_name: Literal["anthropic", "openai", "gemini"] = args.provider  # type: ignore[assignment]
        provider_kwargs: dict[str, str] = {}

        # Verify API key exists
        key_map: dict[str, str] = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY"
        }
        if not os.getenv(key_map[provider_name]):
            print(f"❌ {key_map[provider_name]} not found in environment")
            sys.exit(1)
    else:
        provider_name, provider_kwargs = get_llm_provider_choice()

    # Summary-only mode
    if args.summary_only:
        print_separator()
        print(f"📝 Generating {args.summary_style} summary...\n")

        try:
            summarizer = create_summarizer(
                document_text,
                provider_name=provider_name,
                **provider_kwargs
            )

            for chunk in summarizer.summarize(style=args.summary_style):
                print(chunk, end="", flush=True)
            print("\n")

            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    # Interactive mode
    interactive_mode(document_text, provider_name, provider_kwargs)


if __name__ == "__main__":
    main()
