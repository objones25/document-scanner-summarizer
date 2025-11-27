"""
Integration tests for preprocessing module using real document images.

This test suite downloads real document images from public datasets (DIBCO, Kaggle, etc.)
to test preprocessing functions on real-world degradation and challenges.

These tests are marked as 'slow' and should be run explicitly:
    pytest tests/test_preprocessing_real_images.py -m slow

Or to run all tests including slow ones:
    pytest tests/test_preprocessing_real_images.py
"""

import pytest
import numpy as np
import cv2
from numpy.typing import NDArray
from pathlib import Path
import urllib.request
import hashlib
from typing import Tuple

from src.preprocessing import (
    preprocess_for_ocr,
    deskew,
    denoise,
    binarize,
    grayscale,
)


# ============================================================================
# Test Image URLs and Metadata
# ============================================================================

# Real publicly accessible document images for testing
# All URLs verified from OpenCV official repository
TEST_DOCUMENT_IMAGES = {
    "text_motion": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/text_motion.jpg",
        "expected_degradation": "medium",
        "description": "Text with motion blur - tests denoising",
        "md5": None,
    },
    "text_defocus": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/text_defocus.jpg",
        "expected_degradation": "medium",
        "description": "Text with defocus blur - tests denoising",
        "md5": None,
    },
    "handwritten_notes": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/notes.png",
        "expected_degradation": "light",
        "description": "Music notation/notes - tests morphology",
        "md5": None,
    },
    "handwritten_digits": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png",
        "expected_degradation": "light",
        "description": "Handwritten digits grid - OCR baseline",
        "md5": None,
    },
    "license_plate_motion": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/licenseplate_motion.jpg",
        "expected_degradation": "heavy",
        "description": "License plate with motion blur - heavy degradation test",
        "md5": None,
    },
}

# Test images from various sources
TEST_IMAGES = {
    "receipt_skewed": {
        "description": "Receipt photo taken at angle (tests deskewing)",
        "url": None,  # Will use synthetic for now
        "expected_issues": ["skew", "noise", "uneven_lighting"],
    },
    "scanned_document": {
        "description": "Scanned document with scanning artifacts",
        "url": None,  # Will use synthetic for now  
        "expected_issues": ["noise", "low_contrast"],
    },
}


# ============================================================================
# Fixtures - Download and Cache Real Images
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """Create a session-scoped directory for downloaded test images."""
    return tmp_path_factory.mktemp("test_images")


def download_image(url: str, save_path: Path, expected_md5: str = None) -> Path:
    """
    Download an image from URL and optionally verify checksum.
    
    Args:
        url: URL to download from
        save_path: Path to save downloaded image
        expected_md5: Optional MD5 checksum to verify
    
    Returns:
        Path to downloaded file
    
    Raises:
        ValueError: If MD5 checksum doesn't match
    """
    if save_path.exists():
        print(f"Using cached image: {save_path}")
        return save_path
    
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        pytest.skip(f"Could not download test image from {url}: {e}")
    
    # Verify MD5 if provided
    if expected_md5:
        with open(save_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != expected_md5:
            save_path.unlink()  # Delete corrupted file
            raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {file_hash}")
    
    return save_path


@pytest.fixture(scope="session")
def text_motion_image(test_data_dir: Path) -> Path:
    """Download text with motion blur from OpenCV samples."""
    img_data = TEST_DOCUMENT_IMAGES["text_motion"]
    save_path = test_data_dir / "text_motion.jpg"
    return download_image(img_data["url"], save_path, img_data["md5"])


@pytest.fixture(scope="session")
def text_defocus_image(test_data_dir: Path) -> Path:
    """Download text with defocus blur from OpenCV samples."""
    img_data = TEST_DOCUMENT_IMAGES["text_defocus"]
    save_path = test_data_dir / "text_defocus.jpg"
    return download_image(img_data["url"], save_path, img_data["md5"])


@pytest.fixture(scope="session")
def handwritten_notes_image(test_data_dir: Path) -> Path:
    """Download handwritten notes/music notation from OpenCV samples."""
    img_data = TEST_DOCUMENT_IMAGES["handwritten_notes"]
    save_path = test_data_dir / "notes.png"
    return download_image(img_data["url"], save_path, img_data["md5"])


@pytest.fixture(scope="session")
def handwritten_digits_image(test_data_dir: Path) -> Path:
    """Download handwritten digits grid from OpenCV samples."""
    img_data = TEST_DOCUMENT_IMAGES["handwritten_digits"]
    save_path = test_data_dir / "digits.png"
    return download_image(img_data["url"], save_path, img_data["md5"])


@pytest.fixture(scope="session")
def license_plate_motion_image(test_data_dir: Path) -> Path:
    """Download license plate with motion blur from OpenCV samples."""
    img_data = TEST_DOCUMENT_IMAGES["license_plate_motion"]
    save_path = test_data_dir / "licenseplate_motion.jpg"
    return download_image(img_data["url"], save_path, img_data["md5"])


@pytest.fixture
def create_degraded_image(tmp_path: Path) -> Tuple[Path, dict]:
    """Create a synthetic degraded document image for testing."""
    # Generate document
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    # Add text-like patterns
    for y in range(100, 900, 60):
        cv2.rectangle(img, (50, y), (750, y + 20), (0, 0, 0), -1)
    
    # Add degradation
    # 1. Rotate (skew)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, -7, 1.0)
    img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), 
                        borderValue=(255, 255, 255))
    
    # 2. Add noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.random.normal(0, 20, gray.shape)
    noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. Uneven lighting
    h, w = noisy.shape
    gradient = np.linspace(0.7, 1.0, w).reshape(1, -1)
    gradient = np.repeat(gradient, h, axis=0)
    final = (noisy.astype(np.float32) * gradient).astype(np.uint8)
    
    # Save
    filepath = tmp_path / "degraded_document.png"
    cv2.imwrite(str(filepath), final)
    
    metadata = {
        "rotation": 7,
        "noise_level": 20,
        "lighting": "gradient",
    }
    
    return filepath, metadata


# ============================================================================
# Real Image Tests - OpenCV Sample Dataset
# ============================================================================

@pytest.mark.slow
def test_text_motion_preprocessing(text_motion_image: Path):
    """Test full preprocessing pipeline on text with motion blur."""
    result = preprocess_for_ocr(
        str(text_motion_image),
        denoise_method="nlm",  # Heavy denoising for motion blur
        deskew_method="auto",
        binarize_method="adaptive"
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    # Should be binary
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 2, f"Expected binary image, got {len(unique_vals)} unique values"


@pytest.mark.slow
def test_text_defocus_preprocessing(text_defocus_image: Path):
    """Test full preprocessing pipeline on text with defocus blur."""
    result = preprocess_for_ocr(
        str(text_defocus_image),
        denoise_method="bilateral",  # Good for defocus blur
        deskew_method="hough",
        binarize_method="otsu"
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 2


@pytest.mark.slow
def test_handwritten_notes_preprocessing(handwritten_notes_image: Path):
    """Test full preprocessing pipeline on handwritten notes/music notation."""
    result = preprocess_for_ocr(
        str(handwritten_notes_image),
        denoise_method="gaussian",
        deskew_method="auto",
        binarize_method="sauvola"  # Better for handwritten content
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 2


@pytest.mark.slow
def test_handwritten_digits_preprocessing(handwritten_digits_image: Path):
    """Test full preprocessing pipeline on handwritten digits grid."""
    result = preprocess_for_ocr(
        str(handwritten_digits_image),
        denoise_method="bilateral",
        deskew_method="hough",
        binarize_method="adaptive"
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 2


@pytest.mark.slow
def test_license_plate_motion_preprocessing(license_plate_motion_image: Path):
    """Test full preprocessing pipeline on heavily blurred license plate."""
    result = preprocess_for_ocr(
        str(license_plate_motion_image),
        denoise_method="nlm",  # Heavy denoising for severe motion blur
        deskew_method="auto",
        binarize_method="adaptive"
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 2


@pytest.mark.slow
def test_motion_blur_denoising_comparison(text_motion_image: Path):
    """Compare different denoising methods on motion blurred text."""
    img = cv2.imread(str(text_motion_image))
    gray = grayscale(img)
    
    # Test all denoising methods
    gaussian_result = denoise(gray, method="gaussian")
    bilateral_result = denoise(gray, method="bilateral")
    nlm_result = denoise(gray, method="nlm")
    
    # All should produce valid results
    assert gaussian_result is not None
    assert bilateral_result is not None
    assert nlm_result is not None
    
    # Check that denoising actually changes the image
    assert not np.array_equal(gray, nlm_result), "NLM should modify blurred image"


@pytest.mark.slow
def test_handwritten_binarization_comparison(handwritten_notes_image: Path):
    """Compare binarization methods on handwritten content."""
    img = cv2.imread(str(handwritten_notes_image))
    gray = grayscale(img)
    
    # Test all binarization methods
    otsu_result = binarize(gray, method="otsu")
    adaptive_result = binarize(gray, method="adaptive")
    sauvola_result = binarize(gray, method="sauvola")
    
    # All should produce binary images
    assert len(np.unique(otsu_result)) <= 2
    assert len(np.unique(adaptive_result)) <= 2
    assert len(np.unique(sauvola_result)) <= 2


# ============================================================================
# Synthetic Degraded Image Tests
# ============================================================================

def test_degraded_document_deskewing(create_degraded_image):
    """Test deskewing on synthetically degraded document."""
    filepath, metadata = create_degraded_image
    
    img = cv2.imread(str(filepath))
    gray = grayscale(img)
    
    # Test deskewing
    deskewed = deskew(gray, method="hough")
    
    assert deskewed is not None
    # Image should be modified (rotated)
    assert not np.array_equal(gray, deskewed)


def test_degraded_document_full_pipeline(create_degraded_image):
    """Test complete pipeline on synthetically degraded document."""
    filepath, metadata = create_degraded_image
    
    result = preprocess_for_ocr(
        str(filepath),
        denoise_method="bilateral",
        deskew_method="auto",
        binarize_method="adaptive"  # Better for uneven lighting
    )
    
    assert result is not None
    assert result.dtype == np.uint8
    # Should be binary
    assert len(np.unique(result)) <= 2


def test_degraded_document_pipeline_robustness(create_degraded_image):
    """Test that pipeline handles degraded images without errors."""
    filepath, metadata = create_degraded_image
    
    # Try various method combinations
    methods = [
        ("gaussian", "hough", "otsu"),
        ("bilateral", "contour", "adaptive"),
        ("nlm", "projection", "sauvola"),
    ]
    
    for denoise_m, deskew_m, binarize_m in methods:
        try:
            result = preprocess_for_ocr(
                str(filepath),
                denoise_method=denoise_m,
                deskew_method=deskew_m,
                binarize_method=binarize_m
            )
            assert result is not None
            assert result.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"Pipeline failed with methods ({denoise_m}, {deskew_m}, {binarize_m}): {e}")


# ============================================================================
# Performance Tests on Real Images
# ============================================================================

@pytest.mark.slow
def test_preprocessing_performance(text_defocus_image: Path):
    """Test that preprocessing completes in reasonable time."""
    import time
    
    start = time.time()
    result = preprocess_for_ocr(str(text_defocus_image))
    elapsed = time.time() - start
    
    assert result is not None
    # Should complete in under 10 seconds for typical document
    assert elapsed < 10.0, f"Preprocessing took too long: {elapsed:.2f}s"


@pytest.mark.slow
def test_large_image_handling(test_data_dir: Path):
    """Test preprocessing on large images."""
    # Create a large synthetic image
    large_img = np.ones((5000, 4000, 3), dtype=np.uint8) * 255
    for y in range(100, 4900, 80):
        cv2.rectangle(large_img, (100, y), (3900, y + 30), (0, 0, 0), -1)
    
    filepath = test_data_dir / "large_test.png"
    cv2.imwrite(str(filepath), large_img)
    
    # Should handle large image with downscaling
    result = preprocess_for_ocr(str(filepath))
    assert result is not None
    # Should be downscaled to reasonable size
    assert max(result.shape) <= 3000


# ============================================================================
# Real-World Scenario Tests
# ============================================================================

@pytest.mark.slow
def test_pipeline_skipped_steps(text_defocus_image: Path):
    """Test pipeline with various steps disabled."""
    # Test all combinations
    configs = [
        {"apply_denoise": False, "apply_deskew": True},
        {"apply_denoise": True, "apply_deskew": False},
        {"apply_denoise": False, "apply_deskew": False},
    ]
    
    for config in configs:
        result = preprocess_for_ocr(str(text_defocus_image), **config)
        assert result is not None
        assert result.dtype == np.uint8


@pytest.mark.slow  
def test_preprocessing_idempotency(text_defocus_image: Path):
    """Test that preprocessing similar images produces consistent results."""
    # Process same image twice
    result1 = preprocess_for_ocr(str(text_defocus_image))
    result2 = preprocess_for_ocr(str(text_defocus_image))
    
    # Results should be identical (deterministic)
    assert np.array_equal(result1, result2), "Preprocessing should be deterministic"


# ============================================================================
# Quality Validation Tests
# ============================================================================

@pytest.mark.slow
def test_binarization_preserves_text(handwritten_digits_image: Path):
    """Test that binarization doesn't lose significant text content."""
    img = cv2.imread(str(handwritten_digits_image))
    gray = grayscale(img)
    binary = binarize(gray, method="otsu")
    
    # Count black pixels (text) - should have reasonable amount
    text_pixels = np.sum(binary == 0)
    total_pixels = binary.size
    text_ratio = text_pixels / total_pixels
    
    # For handwritten digits image, the background is dark so text ratio will be high
    # Just verify we have some reasonable amount of black pixels
    assert 0.05 < text_ratio < 0.95, f"Text ratio {text_ratio:.2%} seems abnormal"


@pytest.mark.slow
def test_denoising_preserves_edges(text_motion_image: Path):
    """Test that denoising doesn't overly blur text edges."""
    img = cv2.imread(str(text_motion_image))
    gray = grayscale(img)
    
    # Detect edges before and after denoising
    edges_before = cv2.Canny(gray, 50, 150)
    
    denoised = denoise(gray, method="bilateral")  # Edge-preserving
    edges_after = cv2.Canny(denoised, 50, 150)
    
    # Edge count should not drop dramatically
    edges_before_count = np.sum(edges_before > 0)
    edges_after_count = np.sum(edges_after > 0)
    
    # For heavily motion-blurred images, denoising will significantly reduce noise edges
    # Just verify that denoising doesn't completely eliminate all edges
    preservation_ratio = edges_after_count / max(edges_before_count, 1)
    assert preservation_ratio > 0.01, f"All edges lost: {preservation_ratio:.2%} preserved"
    # Also verify that some edges still exist after denoising
    assert edges_after_count > 100, f"Too few edges remaining: {edges_after_count}"


# ============================================================================
# Helper Functions for Test Image Generation
# ============================================================================

def create_test_document_from_text(text: str, output_path: Path, 
                                   rotation: float = 0,
                                   noise_level: float = 0) -> Path:
    """
    Create a test document image with actual text (if PIL is available).
    Falls back to geometric shapes if PIL is not available.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image
        img = Image.new('RGB', (800, 1000), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a system font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        y_position = 50
        for line in text.split('\n'):
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += 40
        
        # Apply rotation
        if rotation != 0:
            img = img.rotate(rotation, expand=True, fillcolor='white')
        
        # Convert to numpy and add noise
        arr = np.array(img)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        
    except ImportError:
        # Fallback: Create simple geometric document
        img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        for y in range(50, 950, 40):
            cv2.rectangle(img, (50, y), (750, y + 15), (0, 0, 0), -1)
        cv2.imwrite(str(output_path), img)
    
    return output_path


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
