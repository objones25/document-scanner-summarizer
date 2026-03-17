"""
Mock-based tests for MistralOCR, TesseractOCR, and extractor error/edge-case paths.
These tests do not require tesseract or real API keys.
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch, call

from src.ocr import MistralOCR, TesseractOCR, _image_to_string, _image_to_data
from src.extractors import (
    extract_from_docx,
    extract_from_txt,
    extract_from_markdown,
    extract_text,
    _extract_text_from_pdf,
    _ocr_pdf_pages,
    _extract_article_text,
    _detect_source_type,
)


# ---------------------------------------------------------------------------
# MistralOCR (mocked Mistral client)
# ---------------------------------------------------------------------------

class TestMistralOCR:
    def _gray_image(self) -> np.ndarray:
        return np.zeros((100, 100), dtype=np.uint8)

    def _color_image(self) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def _make_ocr_with_mock_client(self) -> tuple[MistralOCR, MagicMock]:
        """Return (ocr, mock_client) with Mistral patched at the module level."""
        mock_client = MagicMock()
        with patch("src.ocr.Mistral", return_value=mock_client):
            ocr = MistralOCR(api_key="fake-key")
        return ocr, mock_client

    def test_init_stores_key(self):
        _, _ = self._make_ocr_with_mock_client()
        ocr, _ = self._make_ocr_with_mock_client()
        assert ocr.api_key == "fake-key"
        assert ocr.model == "mistral-ocr-latest"

    def test_encode_grayscale_image(self):
        ocr, _ = self._make_ocr_with_mock_client()
        result = ocr._encode_image_base64(self._gray_image())
        assert result.startswith("data:image/png;base64,")

    def test_encode_color_image(self):
        ocr, _ = self._make_ocr_with_mock_client()
        result = ocr._encode_image_base64(self._color_image())
        assert result.startswith("data:image/png;base64,")

    def test_encode_unsupported_shape_raises(self):
        ocr, _ = self._make_ocr_with_mock_client()
        bad_image = np.zeros((100, 100, 100, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            ocr._encode_image_base64(bad_image)

    def test_extract_text_with_pages_response(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_page = MagicMock()
        mock_page.markdown = "Hello world"
        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_client.ocr.process.return_value = mock_response

        result = ocr.extract_text(self._color_image())
        assert result == "Hello world"

    def test_extract_text_no_pages_returns_empty(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_response = MagicMock()
        mock_response.pages = []
        mock_client.ocr.process.return_value = mock_response

        result = ocr.extract_text(self._color_image())
        assert result == ""

    def test_extract_text_page_no_markdown_returns_empty(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_page = MagicMock()
        mock_page.markdown = ""
        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_client.ocr.process.return_value = mock_response

        result = ocr.extract_text(self._color_image())
        assert result == ""

    def test_extract_text_no_pages_attr_returns_empty(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_response = MagicMock(spec=[])  # no 'pages' attr
        mock_client.ocr.process.return_value = mock_response

        result = ocr.extract_text(self._color_image())
        assert result == ""

    def test_extract_text_with_confidence_with_text(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_page = MagicMock()
        mock_page.markdown = "Some text"
        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_client.ocr.process.return_value = mock_response

        text, confidence = ocr.extract_text_with_confidence(self._gray_image())
        assert text == "Some text"
        assert confidence == 100.0

    def test_extract_text_with_confidence_no_text(self):
        ocr, mock_client = self._make_ocr_with_mock_client()

        mock_response = MagicMock()
        mock_response.pages = []
        mock_client.ocr.process.return_value = mock_response

        text, confidence = ocr.extract_text_with_confidence(self._gray_image())
        assert text == ""
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Extractor error and edge-case paths
# ---------------------------------------------------------------------------

class TestExtractorEdgeCases:
    def test_extract_text_pdf_exception_returns_empty(self, tmp_path):
        """_extract_text_from_pdf exception handler returns empty string."""
        # Write a corrupted (non-PDF) file
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a real pdf")
        result = _extract_text_from_pdf(str(f))
        assert result == ""

    def test_extract_article_text_no_content_area(self):
        """Returns empty string when HTML has no body or content area."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<html></html>", "lxml")
        # Remove body if present
        if soup.find("body"):
            soup.find("body").decompose()
        result = _extract_article_text(soup)
        assert result == ""

    def test_extract_from_docx_runtime_error(self, tmp_path):
        """DOCX RuntimeError when docx parsing fails."""
        f = tmp_path / "bad.docx"
        f.write_bytes(b"not a docx file")
        with pytest.raises(RuntimeError):
            extract_from_docx(str(f))

    def test_extract_from_txt_latin1_fallback(self, tmp_path):
        """TXT: latin-1 fallback when UTF-8 decoding fails."""
        f = tmp_path / "latin.txt"
        # Write bytes that are valid latin-1 but not valid UTF-8
        f.write_bytes(b"caf\xe9")  # 'café' in latin-1
        result = extract_from_txt(str(f))
        assert "caf" in result

    def test_extract_from_txt_runtime_error_on_general_exception(self, tmp_path):
        """TXT: RuntimeError raised for unexpected read failures."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        with patch("builtins.open", side_effect=OSError("disk error")):
            with pytest.raises(RuntimeError, match="Failed to extract"):
                extract_from_txt(str(f))

    def test_extract_from_markdown_latin1_fallback(self, tmp_path):
        """Markdown: latin-1 fallback when UTF-8 decoding fails."""
        f = tmp_path / "doc.md"
        f.write_bytes(b"# caf\xe9")  # 'café' in latin-1
        result = extract_from_markdown(str(f))
        assert "caf" in result

    def test_extract_from_markdown_runtime_error_on_general_exception(self, tmp_path):
        """Markdown: RuntimeError raised for unexpected read failures."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        with patch("builtins.open", side_effect=OSError("disk error")):
            with pytest.raises(RuntimeError, match="Failed to extract"):
                extract_from_markdown(str(f))

    def test_extract_text_markdown_source_type(self, tmp_path):
        """extract_text routes to markdown extractor for .md files."""
        f = tmp_path / "doc.md"
        f.write_text("# Hello\nWorld")
        result = extract_text(str(f))
        assert "Hello" in result

    def test_detect_source_type_markdown(self, tmp_path):
        """_detect_source_type returns 'markdown' for .md files."""
        result = _detect_source_type("file.md")
        assert result == "markdown"

    def test_detect_source_type_markdown_extension(self):
        result = _detect_source_type("README.markdown")
        assert result == "markdown"

    def test_detect_source_type_unsupported_raises(self):
        with pytest.raises(ValueError, match="Cannot detect source type"):
            _detect_source_type("file.xyz")

    def test_extract_text_unsupported_source_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            extract_text("file.txt", source_type="unsupported")

    def test_extract_text_from_pdf_with_text(self, tmp_path):
        """_extract_text_from_pdf returns text when pages have content."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf content")
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("src.extractors.pypdf.PdfReader", return_value=mock_reader):
            result = _extract_text_from_pdf(str(f))
        assert result == "Sample text"

    def test_extract_from_markdown_file_not_found(self):
        """extract_from_markdown raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_from_markdown("/nonexistent/path/to/file.md")

    def test_extract_from_txt_latin1_inner_exception(self, tmp_path):
        """TXT: RuntimeError when latin-1 fallback read also fails."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "reason")
            raise OSError("disk error on latin-1")

        with patch("builtins.open", side_effect=_side_effect):
            with pytest.raises(RuntimeError, match="Failed to read text file"):
                extract_from_txt(str(f))

    def test_extract_from_markdown_latin1_inner_exception(self, tmp_path):
        """Markdown: RuntimeError when latin-1 fallback read also fails."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "reason")
            raise OSError("disk error on latin-1")

        with patch("builtins.open", side_effect=_side_effect):
            with pytest.raises(RuntimeError, match="Failed to read markdown file"):
                extract_from_markdown(str(f))

    def test_ocr_pdf_pages(self):
        """_ocr_pdf_pages runs OCR over mocked PDF pages."""
        pil_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        mock_ocr = MagicMock()
        mock_ocr.extract_text.return_value = "page text"
        with patch("src.extractors.convert_from_path", return_value=[pil_img]):
            result = _ocr_pdf_pages("fake.pdf", mock_ocr)
        assert result == "page text"

    def test_ocr_pdf_pages_exception_raises_runtime(self):
        """_ocr_pdf_pages raises RuntimeError when convert_from_path fails."""
        with patch("src.extractors.convert_from_path", side_effect=RuntimeError("poppler error")):
            with pytest.raises(RuntimeError, match="Failed to OCR PDF pages"):
                _ocr_pdf_pages("fake.pdf", MagicMock())


# ---------------------------------------------------------------------------
# TesseractOCR (mocked pytesseract)
# ---------------------------------------------------------------------------

class TestTesseractOCR:
    def _gray(self) -> np.ndarray:
        return np.zeros((100, 100), dtype=np.uint8)

    def _color(self) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_image_to_string_returns_str(self):
        with patch("src.ocr.pytesseract") as mock_tess:
            mock_tess.image_to_string.return_value = "hello"
            result = _image_to_string(MagicMock(), "eng", "")
        assert result == "hello"

    def test_image_to_string_non_str_result(self):
        with patch("src.ocr.pytesseract") as mock_tess:
            mock_tess.image_to_string.return_value = 42
            result = _image_to_string(MagicMock(), "eng", "")
        assert result == "42"

    def test_image_to_data_returns_dict(self):
        with patch("src.ocr.pytesseract") as mock_tess:
            mock_tess.image_to_data.return_value = {"conf": [90], "text": ["hi"]}
            result = _image_to_data(MagicMock(), "eng", "", 2)
        assert result == {"conf": [90], "text": ["hi"]}

    def test_image_to_data_non_dict_result(self):
        with patch("src.ocr.pytesseract") as mock_tess:
            mock_tess.image_to_data.return_value = [("conf", [90])]
            result = _image_to_data(MagicMock(), "eng", "", 2)
        assert result == {"conf": [90]}

    def test_init_with_tesseract_cmd(self):
        with patch("src.ocr.pytesseract") as mock_tess:
            TesseractOCR(tesseract_cmd="/usr/local/bin/tesseract")
            assert mock_tess.pytesseract.tesseract_cmd == "/usr/local/bin/tesseract"

    def test_extract_text_grayscale(self):
        with patch("src.ocr._image_to_string", return_value="detected text"):
            ocr = TesseractOCR()
            result = ocr.extract_text(self._gray())
        assert result == "detected text"

    def test_extract_text_color(self):
        with patch("src.ocr._image_to_string", return_value="  text  "):
            ocr = TesseractOCR()
            result = ocr.extract_text(self._color())
        assert result == "text"

    def test_cv2_to_pil_grayscale(self):
        ocr = TesseractOCR()
        pil_img = ocr._cv2_to_pil(self._gray())
        assert isinstance(pil_img, Image.Image)

    def test_cv2_to_pil_color(self):
        ocr = TesseractOCR()
        pil_img = ocr._cv2_to_pil(self._color())
        assert isinstance(pil_img, Image.Image)

    def test_cv2_to_pil_unsupported_raises(self):
        ocr = TesseractOCR()
        bad_img = np.zeros((100, 100, 100, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image shape"):
            ocr._cv2_to_pil(bad_img)

    def test_extract_text_with_confidence_with_words(self):
        data = {"conf": [90, -1, 85], "text": ["hello", "", "world"]}
        with patch("src.ocr._image_to_data", return_value=data):
            ocr = TesseractOCR()
            text, conf = ocr.extract_text_with_confidence(self._gray())
        assert "hello" in text
        assert "world" in text
        assert conf > 0.0

    def test_extract_text_with_confidence_str_conf(self):
        """Confidence values given as strings (not ints) are handled."""
        data = {"conf": ["88", "-1"], "text": ["word", ""]}
        with patch("src.ocr._image_to_data", return_value=data):
            ocr = TesseractOCR()
            text, conf = ocr.extract_text_with_confidence(self._gray())
        assert text == "word"
        assert conf == 88.0

    def test_extract_text_with_confidence_non_str_text_item(self):
        """Non-str text items (e.g. int) are converted via str()."""
        data = {"conf": [90], "text": [42]}
        with patch("src.ocr._image_to_data", return_value=data):
            ocr = TesseractOCR()
            text, conf = ocr.extract_text_with_confidence(self._gray())
        assert text == "42"
        assert conf == 90.0

    def test_extract_text_with_confidence_no_text(self):
        data = {"conf": [-1, -1], "text": ["", ""]}
        with patch("src.ocr._image_to_data", return_value=data):
            ocr = TesseractOCR()
            text, conf = ocr.extract_text_with_confidence(self._gray())
        assert text == ""
        assert conf == 0.0
