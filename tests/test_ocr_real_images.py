"""
Integration tests for OCR module using real document images.

This test suite uses real document images to test both Tesseract and Mistral OCR engines.

These tests are marked as 'slow' and should be run explicitly:
    pytest tests/test_ocr_real_images.py -m slow

Or to run all tests including slow ones:
    pytest tests/test_ocr_real_images.py
"""

import pytest
import cv2
import os
from pathlib import Path
from dotenv import load_dotenv

from src.ocr import TesseractOCR, MistralOCR
from src.preprocessing import preprocess_for_ocr


# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Fixtures - Download and Cache Real Images
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a session-scoped directory for test images."""
    return tmp_path_factory.mktemp("ocr_test_images")


@pytest.fixture(scope="session")
def simple_text_image(test_data_dir: Path) -> Path:
    """
    Download a simple text image from OpenCV samples.
    This is a handwritten digits grid - simple and clean text.
    """
    import urllib.request

    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png"
    save_path = test_data_dir / "digits.png"

    if save_path.exists():
        return save_path

    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        pytest.skip(f"Could not download test image from {url}: {e}")

    return save_path


@pytest.fixture(scope="session")
def text_motion_image(test_data_dir: Path) -> Path:
    """Download text with motion blur from OpenCV samples."""
    import urllib.request

    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/text_motion.jpg"
    save_path = test_data_dir / "text_motion.jpg"

    if save_path.exists():
        return save_path

    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        pytest.skip(f"Could not download test image from {url}: {e}")

    return save_path


@pytest.fixture
def tesseract_ocr() -> TesseractOCR:
    """Create a Tesseract OCR engine instance."""
    return TesseractOCR(language="eng")


@pytest.fixture
def mistral_ocr() -> MistralOCR:
    """
    Create a Mistral OCR engine instance.
    Requires MISTRAL_API_KEY in environment or .env file.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not found in environment")

    return MistralOCR(api_key=api_key)


# ============================================================================
# Tesseract OCR Tests
# ============================================================================

def test_tesseract_extract_text(tesseract_ocr: TesseractOCR, simple_text_image: Path):
    """Test Tesseract text extraction on simple clean image."""
    # Load and preprocess image
    img = cv2.imread(str(simple_text_image))
    preprocessed = preprocess_for_ocr(
        str(simple_text_image),
        denoise_method="gaussian",
        binarize_method="adaptive"
    )

    # Extract text
    text = tesseract_ocr.extract_text(preprocessed)

    # Should extract some text
    assert isinstance(text, str)
    assert len(text) > 0, "Should extract some text from image"
    print(f"\nExtracted text (Tesseract): {text[:100]}...")


def test_tesseract_extract_text_with_confidence(tesseract_ocr: TesseractOCR, simple_text_image: Path):
    """Test Tesseract text extraction with confidence scores."""
    # Load and preprocess image
    preprocessed = preprocess_for_ocr(
        str(simple_text_image),
        denoise_method="gaussian",
        binarize_method="adaptive"
    )

    # Extract text with confidence
    text, confidence = tesseract_ocr.extract_text_with_confidence(preprocessed)

    # Should extract text and confidence
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 100.0, f"Confidence should be between 0-100, got {confidence}"
    print(f"\nExtracted text: {text[:100]}...")
    print(f"Confidence: {confidence:.2f}%")


@pytest.mark.slow
def test_tesseract_motion_blur_text(tesseract_ocr: TesseractOCR, text_motion_image: Path):
    """Test Tesseract on motion blurred text."""
    # Load and preprocess with aggressive denoising
    preprocessed = preprocess_for_ocr(
        str(text_motion_image),
        denoise_method="nlm",  # Heavy denoising for motion blur
        binarize_method="adaptive"
    )

    # Extract text
    text, confidence = tesseract_ocr.extract_text_with_confidence(preprocessed)

    # Should extract something, even if degraded
    assert isinstance(text, str)
    print(f"\nExtracted text from motion blur: {text[:100]}...")
    print(f"Confidence: {confidence:.2f}%")


# ============================================================================
# Mistral OCR Tests
# ============================================================================

@pytest.mark.slow
def test_mistral_extract_text(mistral_ocr: MistralOCR, simple_text_image: Path):
    """Test Mistral OCR text extraction on simple clean image."""
    # Load image
    img = cv2.imread(str(simple_text_image))

    # Extract text (Mistral works on raw images, no preprocessing needed)
    text = mistral_ocr.extract_text(img)

    # Should extract some text
    assert isinstance(text, str)
    assert len(text) > 0, "Should extract some text from image"
    print(f"\nExtracted text (Mistral): {text[:200]}...")


@pytest.mark.slow
def test_mistral_extract_text_with_confidence(mistral_ocr: MistralOCR, simple_text_image: Path):
    """Test Mistral OCR text extraction with confidence scores."""
    # Load image
    img = cv2.imread(str(simple_text_image))

    # Extract text with confidence
    text, confidence = mistral_ocr.extract_text_with_confidence(img)

    # Should extract text and confidence
    assert isinstance(text, str)
    assert len(text) > 0
    assert isinstance(confidence, float)
    assert confidence == 100.0 or confidence == 0.0, "Mistral returns 100 if text found, 0 otherwise"
    print(f"\nExtracted text (Mistral): {text[:200]}...")
    print(f"Confidence: {confidence:.2f}%")


@pytest.mark.slow
def test_mistral_motion_blur_text(mistral_ocr: MistralOCR, text_motion_image: Path):
    """Test Mistral OCR on motion blurred text."""
    # Load image (no preprocessing - Mistral handles degradation)
    img = cv2.imread(str(text_motion_image))

    # Extract text
    text, confidence = mistral_ocr.extract_text_with_confidence(img)

    # Should extract something
    assert isinstance(text, str)
    print(f"\nExtracted text from motion blur (Mistral): {text[:200]}...")
    print(f"Confidence: {confidence:.2f}%")


# ============================================================================
# Comparison Tests
# ============================================================================

@pytest.mark.slow
def test_compare_tesseract_vs_mistral(
    tesseract_ocr: TesseractOCR,
    mistral_ocr: MistralOCR,
    simple_text_image: Path
):
    """Compare Tesseract and Mistral OCR on the same image."""
    # Tesseract with preprocessing
    preprocessed = preprocess_for_ocr(
        str(simple_text_image),
        denoise_method="gaussian",
        binarize_method="adaptive"
    )
    tesseract_text, tesseract_conf = tesseract_ocr.extract_text_with_confidence(preprocessed)

    # Mistral without preprocessing
    img = cv2.imread(str(simple_text_image))
    mistral_text, mistral_conf = mistral_ocr.extract_text_with_confidence(img)

    # Both should extract text
    assert len(tesseract_text) > 0
    assert len(mistral_text) > 0

    print(f"\nTesseract extracted: {tesseract_text[:100]}...")
    print(f"Tesseract confidence: {tesseract_conf:.2f}%")
    print(f"\nMistral extracted: {mistral_text[:100]}...")
    print(f"Mistral confidence: {mistral_conf:.2f}%")


# ============================================================================
# Edge Cases
# ============================================================================

def test_tesseract_empty_image(tesseract_ocr: TesseractOCR, test_data_dir: Path):
    """Test Tesseract on empty/blank image."""
    import numpy as np

    # Create blank image
    blank = np.ones((500, 500), dtype=np.uint8) * 255

    # Extract text
    text = tesseract_ocr.extract_text(blank)

    # Should return empty string or whitespace
    assert isinstance(text, str)
    assert len(text.strip()) == 0, "Blank image should extract no text"


@pytest.mark.slow
def test_mistral_empty_image(mistral_ocr: MistralOCR, test_data_dir: Path):
    """Test Mistral OCR on empty/blank image."""
    import numpy as np

    # Create blank image
    blank = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Extract text
    text, confidence = mistral_ocr.extract_text_with_confidence(blank)

    # Should return empty or minimal text
    assert isinstance(text, str)
    print(f"\nMistral result on blank image: '{text}'")
    print(f"Confidence: {confidence:.2f}%")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
def test_tesseract_performance(tesseract_ocr: TesseractOCR, simple_text_image: Path):
    """Test that Tesseract completes in reasonable time."""
    import time

    preprocessed = preprocess_for_ocr(str(simple_text_image))

    start = time.time()
    text = tesseract_ocr.extract_text(preprocessed)
    elapsed = time.time() - start

    assert len(text) > 0
    assert elapsed < 10.0, f"Tesseract took too long: {elapsed:.2f}s"
    print(f"\nTesseract processing time: {elapsed:.3f}s")


@pytest.mark.slow
def test_mistral_performance(mistral_ocr: MistralOCR, simple_text_image: Path):
    """Test that Mistral OCR completes in reasonable time."""
    import time

    img = cv2.imread(str(simple_text_image))

    start = time.time()
    text = mistral_ocr.extract_text(img)
    elapsed = time.time() - start

    assert len(text) > 0
    # Mistral is API-based, so allow more time for network latency
    assert elapsed < 30.0, f"Mistral OCR took too long: {elapsed:.2f}s"
    print(f"\nMistral OCR processing time: {elapsed:.3f}s")
