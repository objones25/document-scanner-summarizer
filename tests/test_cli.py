"""Tests for src/cli.py — all I/O and external calls mocked."""

import sys
import pytest
from unittest.mock import MagicMock, patch

from src.cli import (
    print_header,
    print_separator,
    get_source_input,
    get_ocr_engine_choice,
    get_llm_provider_choice,
    extract_document_text,
    interactive_mode,
    main,
)
from src.ocr import TesseractOCR, MistralOCR


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

class TestPrintHelpers:
    def test_print_header_outputs_title(self, capsys):
        print_header()
        out = capsys.readouterr().out
        assert "Document Scanner" in out

    def test_print_separator_outputs_dashes(self, capsys):
        print_separator()
        out = capsys.readouterr().out
        assert "-" in out


# ---------------------------------------------------------------------------
# get_source_input
# ---------------------------------------------------------------------------

class TestGetSourceInput:
    def test_url_accepted(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "https://example.com")
        result = get_source_input()
        assert result == "https://example.com"

    def test_http_url_accepted(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "http://example.com")
        result = get_source_input()
        assert result == "http://example.com"

    def test_empty_input_retries_then_url(self, monkeypatch):
        responses = iter(["", "  ", "https://example.com"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        result = get_source_input()
        assert result == "https://example.com"

    def test_missing_file_retries_then_accepted(self, tmp_path, monkeypatch):
        real_file = tmp_path / "real.txt"
        real_file.write_text("content")
        responses = iter(["/nonexistent/file.txt", str(real_file)])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        result = get_source_input()
        assert result == str(real_file)

    def test_existing_file_accepted(self, tmp_path, monkeypatch):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        monkeypatch.setattr("builtins.input", lambda prompt="": str(f))
        result = get_source_input()
        assert result == str(f)


# ---------------------------------------------------------------------------
# get_ocr_engine_choice
# ---------------------------------------------------------------------------

class TestGetOcrEngineChoice:
    def test_default_choice_returns_tesseract(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        result = get_ocr_engine_choice()
        assert isinstance(result, TesseractOCR)

    def test_choice_1_returns_tesseract(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "1")
        result = get_ocr_engine_choice()
        assert isinstance(result, TesseractOCR)

    def test_choice_2_with_key_returns_mistral(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        result = get_ocr_engine_choice()
        assert isinstance(result, MistralOCR)

    def test_choice_2_without_key_retries_then_tesseract(self, monkeypatch, capsys):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        responses = iter(["2", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        result = get_ocr_engine_choice()
        assert isinstance(result, TesseractOCR)
        out = capsys.readouterr().out
        assert "MISTRAL_API_KEY" in out

    def test_invalid_choice_retries_then_valid(self, monkeypatch, capsys):
        responses = iter(["9", "abc", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        result = get_ocr_engine_choice()
        assert isinstance(result, TesseractOCR)
        out = capsys.readouterr().out
        assert "Invalid" in out


# ---------------------------------------------------------------------------
# get_llm_provider_choice
# ---------------------------------------------------------------------------

class TestGetLlmProviderChoice:
    def test_choice_1_anthropic_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        monkeypatch.setattr("builtins.input", lambda prompt="": "1")
        provider, kwargs = get_llm_provider_choice()
        assert provider == "anthropic"

    def test_choice_2_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        provider, kwargs = get_llm_provider_choice()
        assert provider == "openai"

    def test_choice_3_gemini_with_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake")
        monkeypatch.setattr("builtins.input", lambda prompt="": "3")
        provider, kwargs = get_llm_provider_choice()
        assert provider == "gemini"

    def test_choice_1_without_key_retries(self, monkeypatch, capsys):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake")
        responses = iter(["1", "2"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        provider, _ = get_llm_provider_choice()
        assert provider == "openai"
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().out

    def test_choice_2_without_key_retries(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["2", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        provider, _ = get_llm_provider_choice()
        assert provider == "anthropic"
        assert "OPENAI_API_KEY" in capsys.readouterr().out

    def test_choice_3_without_key_retries(self, monkeypatch, capsys):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["3", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        provider, _ = get_llm_provider_choice()
        assert provider == "anthropic"
        assert "GOOGLE_API_KEY" in capsys.readouterr().out

    def test_default_choice_is_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        provider, _ = get_llm_provider_choice()
        assert provider == "anthropic"

    def test_invalid_choice_retries(self, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["99", "abc", "1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        provider, _ = get_llm_provider_choice()
        assert provider == "anthropic"
        assert "Invalid" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# extract_document_text
# ---------------------------------------------------------------------------

class TestExtractDocumentText:
    def test_success_returns_text(self, capsys):
        mock_ocr = MagicMock(spec=TesseractOCR)
        with patch("src.cli.extract_text", return_value="Extracted content"):
            result = extract_document_text("doc.txt", mock_ocr)
        assert result == "Extracted content"
        assert "Extracted" in capsys.readouterr().out

    def test_empty_text_calls_sys_exit(self):
        mock_ocr = MagicMock(spec=TesseractOCR)
        with patch("src.cli.extract_text", return_value="   "):
            with pytest.raises(SystemExit) as exc:
                extract_document_text("doc.txt", mock_ocr)
        assert exc.value.code == 1

    def test_exception_calls_sys_exit(self):
        mock_ocr = MagicMock(spec=TesseractOCR)
        with patch("src.cli.extract_text", side_effect=RuntimeError("OCR failed")):
            with pytest.raises(SystemExit) as exc:
                extract_document_text("doc.txt", mock_ocr)
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# interactive_mode
# ---------------------------------------------------------------------------

class TestInteractiveMode:
    def _make_summarizer(self, tokens: list[str] | None = None) -> MagicMock:
        mock = MagicMock()
        mock.ask.side_effect = lambda q, **kw: iter(tokens or ["Answer"])
        mock.summarize.side_effect = lambda style="concise", **kw: iter(["Summary"])
        return mock

    def test_exit_command_exits_zero(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "/exit")
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit) as exc:
                interactive_mode("doc text", "anthropic", {})
        assert exc.value.code == 0

    def test_exit_command_case_insensitive(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "/EXIT")
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit) as exc:
                interactive_mode("doc text", "anthropic", {})
        assert exc.value.code == 0

    def test_clear_command(self, monkeypatch, capsys):
        responses = iter(["/clear", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.clear_history.assert_called_once()
        assert "cleared" in capsys.readouterr().out

    def test_summary_default_style(self, monkeypatch):
        responses = iter(["/summary", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.summarize.assert_called_once_with(style="concise")

    def test_summary_detailed_style(self, monkeypatch):
        responses = iter(["/summary detailed", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.summarize.assert_called_once_with(style="detailed")

    def test_summary_bullet_points_style(self, monkeypatch):
        responses = iter(["/summary bullet-points", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.summarize.assert_called_once_with(style="bullet-points")

    def test_summary_unknown_style_uses_concise(self, monkeypatch, capsys):
        responses = iter(["/summary badstyle", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.summarize.assert_called_once_with(style="concise")
        assert "Unknown style" in capsys.readouterr().out

    def test_regular_question(self, monkeypatch):
        responses = iter(["What is this about?", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.ask.assert_called_once_with("What is this about?")

    def test_empty_input_skipped(self, monkeypatch):
        responses = iter(["", "  ", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        mock_summarizer.ask.assert_not_called()

    def test_keyboard_interrupt_exits_zero(self, monkeypatch):
        def raise_keyboard(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_keyboard)
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit) as exc:
                interactive_mode("doc text", "anthropic", {})
        assert exc.value.code == 0

    def test_eof_error_exits_zero(self, monkeypatch):
        def raise_eof(*args, **kwargs):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        mock_summarizer = self._make_summarizer()
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit) as exc:
                interactive_mode("doc text", "anthropic", {})
        assert exc.value.code == 0

    def test_stream_error_on_ask_printed(self, monkeypatch, capsys):
        responses = iter(["What?", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.side_effect = Exception("API error")
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        assert "Error" in capsys.readouterr().out

    def test_stream_error_on_summarize_printed(self, monkeypatch, capsys):
        responses = iter(["/summary", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.side_effect = Exception("Summarize error")
        with patch("src.cli.create_summarizer", return_value=mock_summarizer):
            with pytest.raises(SystemExit):
                interactive_mode("doc text", "anthropic", {})
        assert "Error" in capsys.readouterr().out

    def test_create_summarizer_failure_exits_one(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "/exit")
        with patch("src.cli.create_summarizer", side_effect=Exception("Init error")):
            with pytest.raises(SystemExit) as exc:
                interactive_mode("doc text", "anthropic", {})
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_source_arg_interactive_mode(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(sys, "argv", ["cli", str(f)])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit):
                    main()

    def test_no_args_prompts_for_source(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["cli"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        # Provide: source URL, provider choice (empty = default anthropic), then /exit
        responses = iter(["https://example.com", "", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Web content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit):
                    main()

    def test_summary_only_exits_zero(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--summary-only"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = iter(["Summary text"])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 0

    def test_summary_only_detailed_style(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--summary-only", "--summary-style", "detailed"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = iter(["Detailed summary"])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 0
        mock_summarizer.summarize.assert_called_once_with(style="detailed")

    def test_summary_only_exception_exits_one(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--summary-only"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", side_effect=Exception("Fail")):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 1

    def test_ocr_mistral_no_key_exits_one(self, tmp_path, monkeypatch):
        f = tmp_path / "scan.jpg"
        f.write_bytes(b"fake image data")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--ocr", "mistral"])
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with patch("dotenv.load_dotenv"):  # prevent .env from overriding delenv
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1

    def test_no_args_interactive_ocr_for_image(self, tmp_path, monkeypatch):
        """Covers the interactive OCR selection path (line 362)."""
        f = tmp_path / "scan.jpg"
        f.write_bytes(b"fake image data")
        monkeypatch.setattr(sys, "argv", ["cli"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        # Input sequence: image path, OCR choice (1=tesseract), provider (1=anthropic), /exit
        responses = iter([str(f), "1", "1", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="OCR content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with patch("dotenv.load_dotenv"):
                    with pytest.raises(SystemExit):
                        main()

    def test_missing_provider_key_exits_one(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--provider", "anthropic"])
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("dotenv.load_dotenv"):  # prevent .env from overriding delenv
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code == 1

    def test_gemini_with_advanced_flags(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(
            sys, "argv",
            ["cli", str(f), "--provider", "gemini",
             "--thinking", "--grounding", "--code-execution"]
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "fake")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer) as mock_create:
                with pytest.raises(SystemExit):
                    main()
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("enable_thinking") is True
        assert call_kwargs.get("enable_grounding") is True
        assert call_kwargs.get("enable_code_execution") is True

    def test_anthropic_with_advanced_flags(self, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("Content")
        monkeypatch.setattr(
            sys, "argv",
            ["cli", str(f), "--provider", "anthropic",
             "--code-execution", "--web-search", "--web-fetch"]
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer) as mock_create:
                with pytest.raises(SystemExit):
                    main()
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("enable_code_execution") is True
        assert call_kwargs.get("enable_web_search") is True
        assert call_kwargs.get("enable_web_fetch") is True

    def test_mistral_ocr_with_key_creates_engine(self, tmp_path, monkeypatch):
        """Covers line 360: MistralOCR(api_key=api_key) when key is present."""
        f = tmp_path / "scan.jpg"
        f.write_bytes(b"fake image data")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--ocr", "mistral", "--provider", "anthropic"])
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="OCR content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with patch("dotenv.load_dotenv"):
                    with pytest.raises(SystemExit):
                        main()

    def test_pdf_source_uses_tesseract_by_default(self, tmp_path, monkeypatch):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"fake pdf")
        monkeypatch.setattr(sys, "argv", ["cli", str(f), "--provider", "anthropic"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="PDF content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit):
                    main()

    def test_url_source_skips_ocr_prompt(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["cli", "https://example.com", "--provider", "anthropic"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
        responses = iter(["/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Web content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit):
                    main()

    def test_no_args_interactive_provider_selection(self, monkeypatch):
        """No CLI args → interactive prompts for source, OCR is skipped for URL, provider selected."""
        monkeypatch.setattr(sys, "argv", ["cli"])
        monkeypatch.setenv("GOOGLE_API_KEY", "fake")
        # source (URL), then provider choice (3=gemini), then /exit
        responses = iter(["https://example.com", "3", "/exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
        mock_summarizer = MagicMock()
        mock_summarizer.ask.return_value = iter([])
        with patch("src.cli.extract_text", return_value="Content"):
            with patch("src.cli.create_summarizer", return_value=mock_summarizer):
                with pytest.raises(SystemExit):
                    main()
