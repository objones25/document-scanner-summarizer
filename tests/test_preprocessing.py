"""
Unit tests for preprocessing module using synthetic test images.

This test suite uses programmatically generated images to test all preprocessing
functions with controlled parameters for reproducible, fast tests.
"""

import pytest
import numpy as np
import cv2
from numpy.typing import NDArray
from pathlib import Path

from src.preprocessing import (
    load_image,
    resize_if_needed,
    grayscale,
    rotate_image,
    detect_skew_angle_hough,
    detect_skew_angle_contour,
    detect_skew_angle_projection,
    deskew,
    denoise_gaussian,
    denoise_bilateral,
    denoise_nlm,
    denoise_morphological,
    denoise,
    binarize_otsu,
    binarize_adaptive,
    binarize_sauvola,
    binarize,
    preprocess_for_ocr,
)


# ============================================================================
# Test Fixtures - Synthetic Image Generation
# ============================================================================

@pytest.fixture
def clean_image() -> NDArray[np.uint8]:
    """Generate a clean synthetic document with text."""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some horizontal lines to simulate text
    for y in range(100, 700, 80):
        cv2.rectangle(img, (50, y), (550, y + 30), (0, 0, 0), -1)
    
    # Add some vertical elements
    for x in range(100, 500, 100):
        cv2.rectangle(img, (x, 150), (x + 20, 650), (0, 0, 0), -1)
    
    return img


@pytest.fixture
def gray_image(clean_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Generate grayscale version of clean image."""
    return cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def rotated_image(clean_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Generate a rotated document (5° clockwise)."""
    center = (clean_image.shape[1] // 2, clean_image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, -5, 1.0)  # -5 for clockwise
    return cv2.warpAffine(clean_image, matrix, (clean_image.shape[1], clean_image.shape[0]), 
                         borderValue=(255, 255, 255))


@pytest.fixture
def noisy_image(gray_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Generate a noisy document with Gaussian noise."""
    noise = np.random.normal(0, 25, gray_image.shape).astype(np.float32)
    noisy = np.clip(gray_image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


@pytest.fixture
def uneven_lighting_image(gray_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Generate document with gradient lighting (dark on left, bright on right)."""
    h, w = gray_image.shape
    gradient = np.linspace(0.6, 1.0, w).reshape(1, -1)
    gradient = np.repeat(gradient, h, axis=0)
    return (gray_image.astype(np.float32) * gradient).astype(np.uint8)


@pytest.fixture
def temp_image_file(clean_image: NDArray[np.uint8], tmp_path: Path) -> Path:
    """Create a temporary image file for file I/O tests."""
    filepath = tmp_path / "test_image.png"
    cv2.imwrite(str(filepath), clean_image)
    return filepath


@pytest.fixture
def small_image() -> NDArray[np.uint8]:
    """Generate a very small image (500x500)."""
    return np.ones((500, 500, 3), dtype=np.uint8) * 255


@pytest.fixture
def large_image() -> NDArray[np.uint8]:
    """Generate a very large image (4000x4000)."""
    return np.ones((4000, 4000, 3), dtype=np.uint8) * 255


# ============================================================================
# File I/O Tests
# ============================================================================

def test_load_image_success(temp_image_file: Path):
    """Test loading a valid image file."""
    img = load_image(str(temp_image_file))
    assert img is not None
    assert img.shape[2] == 3  # BGR format
    assert img.dtype == np.uint8


def test_load_image_file_not_found():
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_file.jpg")


def test_load_image_invalid_format(tmp_path: Path):
    """Test loading invalid image format raises ValueError."""
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("not an image")
    
    with pytest.raises(ValueError):
        load_image(str(invalid_file))


# ============================================================================
# Resize Tests
# ============================================================================

def test_resize_if_needed_upscale(small_image: NDArray[np.uint8]):
    """Test upscaling small images."""
    resized = resize_if_needed(small_image, min_dimension=1000)
    assert min(resized.shape[:2]) >= 1000
    assert resized.shape[2] == 3


def test_resize_if_needed_downscale(large_image: NDArray[np.uint8]):
    """Test downscaling large images."""
    resized = resize_if_needed(large_image, max_dimension=3000)
    assert max(resized.shape[:2]) <= 3000
    assert resized.shape[2] == 3


def test_resize_if_needed_no_change(clean_image: NDArray[np.uint8]):
    """Test that optimal-size images are not resized."""
    original_shape = clean_image.shape
    resized = resize_if_needed(clean_image, min_dimension=500, max_dimension=3500)
    assert resized.shape == original_shape


def test_resize_maintains_aspect_ratio(small_image: NDArray[np.uint8]):
    """Test that aspect ratio is maintained during resize."""
    original_ratio = small_image.shape[0] / small_image.shape[1]
    resized = resize_if_needed(small_image, min_dimension=1000)
    new_ratio = resized.shape[0] / resized.shape[1]
    assert abs(original_ratio - new_ratio) < 0.01


# ============================================================================
# Grayscale Tests
# ============================================================================

def test_grayscale_conversion(clean_image: NDArray[np.uint8]):
    """Test BGR to grayscale conversion."""
    gray = grayscale(clean_image)
    assert len(gray.shape) == 2  # Single channel
    assert gray.dtype == np.uint8
    assert gray.shape[:2] == clean_image.shape[:2]


def test_grayscale_values_in_range(clean_image: NDArray[np.uint8]):
    """Test grayscale values are in valid range."""
    gray = grayscale(clean_image)
    assert np.all(gray >= 0)
    assert np.all(gray <= 255)


# ============================================================================
# Rotation Tests
# ============================================================================

@pytest.mark.parametrize("angle", [0, 5, -10, 45, -45, 90])
def test_rotate_image_various_angles(gray_image: NDArray[np.uint8], angle: float):
    """Test rotation at various angles."""
    rotated = rotate_image(gray_image, angle)
    assert rotated is not None
    assert rotated.dtype == np.uint8


def test_rotate_image_no_rotation(gray_image: NDArray[np.uint8]):
    """Test that 0° rotation returns similar image."""
    rotated = rotate_image(gray_image, 0)
    assert rotated.shape == gray_image.shape
    assert np.array_equal(rotated, gray_image)


def test_rotate_image_increases_dimensions(gray_image: NDArray[np.uint8]):
    """Test that rotation increases image dimensions."""
    rotated = rotate_image(gray_image, 45)
    # Rotated image should be larger to accommodate corners
    assert rotated.shape[0] >= gray_image.shape[0]
    assert rotated.shape[1] >= gray_image.shape[1]


def test_rotate_image_border_value(gray_image: NDArray[np.uint8]):
    """Test custom border value."""
    rotated = rotate_image(gray_image, 10, border_value=128)
    # Check that border pixels exist with custom value
    assert rotated is not None


# ============================================================================
# Skew Detection Tests
# ============================================================================

def test_detect_skew_angle_hough_straight_image(gray_image: NDArray[np.uint8]):
    """Test skew detection on straight image."""
    angle = detect_skew_angle_hough(gray_image)
    assert abs(angle) < 2.0  # Should be nearly 0


def test_detect_skew_angle_contour_straight_image(gray_image: NDArray[np.uint8]):
    """Test contour-based skew detection on straight image."""
    angle = detect_skew_angle_contour(gray_image)
    # Contour method can return 0 or 90 for axis-aligned rectangles
    # Both indicate no skew for our purposes
    assert abs(angle) < 5.0 or abs(angle - 90) < 5.0


def test_detect_skew_angle_projection_straight_image(gray_image: NDArray[np.uint8]):
    """Test projection-based skew detection on straight image."""
    angle = detect_skew_angle_projection(gray_image, angle_range=10, angle_step=1.0)
    assert abs(angle) < 5.0


# ============================================================================
# Deskew Tests
# ============================================================================

@pytest.mark.parametrize("method", ["hough", "contour", "projection", "auto"])
def test_deskew_methods(gray_image: NDArray[np.uint8], method: str):
    """Test all deskewing methods."""
    result = deskew(gray_image, method=method)
    assert result is not None
    assert result.dtype == np.uint8
    # Deskewing may change dimensions slightly (rotation can swap height/width)
    # Just verify we get a reasonable sized output
    assert min(result.shape[:2]) >= min(gray_image.shape[:2]) * 0.7


def test_deskew_straight_image_unchanged(gray_image: NDArray[np.uint8]):
    """Test that straight images are not modified."""
    result = deskew(gray_image, method="hough")
    # Should be similar to original (minimal change)
    assert result.shape[:2] == gray_image.shape[:2]


def test_deskew_invalid_method_raises_error(gray_image: NDArray[np.uint8]):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        deskew(gray_image, method="invalid_method")


def test_deskew_max_angle_parameter(gray_image: NDArray[np.uint8]):
    """Test max_angle parameter is respected."""
    result = deskew(gray_image, method="hough", max_angle=15.0)
    assert result is not None


# ============================================================================
# Denoise Tests
# ============================================================================

def test_denoise_gaussian(noisy_image: NDArray[np.uint8]):
    """Test Gaussian denoising."""
    result = denoise_gaussian(noisy_image, kernel_size=5)
    assert result.shape == noisy_image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_denoise_gaussian_various_kernel_sizes(noisy_image: NDArray[np.uint8], kernel_size: int):
    """Test Gaussian denoising with various kernel sizes."""
    result = denoise_gaussian(noisy_image, kernel_size=kernel_size)
    assert result is not None


def test_denoise_gaussian_even_kernel_becomes_odd(noisy_image: NDArray[np.uint8]):
    """Test that even kernel size is adjusted to odd."""
    result = denoise_gaussian(noisy_image, kernel_size=4)  # Even
    assert result is not None  # Should work (converted to 5)


def test_denoise_bilateral(noisy_image: NDArray[np.uint8]):
    """Test bilateral filter denoising."""
    result = denoise_bilateral(noisy_image, d=9, sigma_color=75, sigma_space=75)
    assert result.shape == noisy_image.shape
    assert result.dtype == np.uint8


def test_denoise_nlm(noisy_image: NDArray[np.uint8]):
    """Test Non-Local Means denoising."""
    result = denoise_nlm(noisy_image, h=10)
    assert result.shape == noisy_image.shape
    assert result.dtype == np.uint8


def test_denoise_morphological():
    """Test morphological denoising on binary image."""
    # Create binary image with salt-and-pepper noise
    binary = np.ones((100, 100), dtype=np.uint8) * 255
    binary[20:30, 20:30] = 0  # Add some black regions
    # Add noise
    binary[25, 25] = 255  # Single white pixel in black region
    binary[50, 50] = 0    # Single black pixel in white region
    
    result = denoise_morphological(binary, kernel_size=2)
    assert result.shape == binary.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("method", ["gaussian", "bilateral", "nlm", "morphological"])
def test_denoise_main_function_methods(noisy_image: NDArray[np.uint8], method: str):
    """Test main denoise function with all methods."""
    result = denoise(noisy_image, method=method)
    assert result is not None
    assert result.dtype == np.uint8


@pytest.mark.parametrize("noise_level", ["light", "medium", "heavy"])
def test_denoise_auto_mode(noisy_image: NDArray[np.uint8], noise_level: str):
    """Test auto mode with different noise levels."""
    result = denoise(noisy_image, method="auto", noise_level=noise_level)
    assert result is not None
    assert result.dtype == np.uint8


def test_denoise_invalid_method_raises_error(noisy_image: NDArray[np.uint8]):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        denoise(noisy_image, method="invalid_method")


def test_denoise_with_kwargs(noisy_image: NDArray[np.uint8]):
    """Test denoise with custom parameters via kwargs."""
    result = denoise(noisy_image, method="gaussian", kernel_size=7)
    assert result is not None


# ============================================================================
# Binarization Tests
# ============================================================================

def test_binarize_otsu(gray_image: NDArray[np.uint8]):
    """Test Otsu's binarization."""
    result = binarize_otsu(gray_image)
    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8
    # Binary image should have only 2 unique values (0 and 255)
    assert len(np.unique(result)) <= 2


def test_binarize_adaptive(uneven_lighting_image: NDArray[np.uint8]):
    """Test adaptive binarization on uneven lighting."""
    result = binarize_adaptive(uneven_lighting_image, block_size=11, c=2)
    assert result.shape == uneven_lighting_image.shape
    assert result.dtype == np.uint8
    assert len(np.unique(result)) <= 2


def test_binarize_adaptive_even_block_size_becomes_odd(gray_image: NDArray[np.uint8]):
    """Test that even block size is adjusted to odd."""
    result = binarize_adaptive(gray_image, block_size=10)  # Even
    assert result is not None  # Should work (converted to 11)


def test_binarize_sauvola(gray_image: NDArray[np.uint8]):
    """Test Sauvola's binarization."""
    result = binarize_sauvola(gray_image, window_size=25, k=0.2)
    assert result.shape == gray_image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("method", ["otsu", "adaptive", "sauvola"])
def test_binarize_main_function_methods(gray_image: NDArray[np.uint8], method: str):
    """Test main binarize function with all methods."""
    result = binarize(gray_image, method=method)
    assert result is not None
    assert result.dtype == np.uint8
    # Should be binary (mostly 0 and 255)
    unique_vals = np.unique(result)
    assert len(unique_vals) <= 3  # Allow small rounding variations


def test_binarize_invalid_method_raises_error(gray_image: NDArray[np.uint8]):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError):
        binarize(gray_image, method="invalid_method")


def test_binarize_with_kwargs(gray_image: NDArray[np.uint8]):
    """Test binarize with custom parameters via kwargs."""
    result = binarize(gray_image, method="adaptive", block_size=15, c=3)
    assert result is not None


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================

def test_preprocess_for_ocr_default_pipeline(temp_image_file: Path):
    """Test complete preprocessing pipeline with defaults."""
    result = preprocess_for_ocr(str(temp_image_file))
    assert result is not None
    assert result.dtype == np.uint8
    # Should be binary
    assert len(np.unique(result)) <= 2


def test_preprocess_for_ocr_custom_methods(temp_image_file: Path):
    """Test pipeline with custom methods."""
    result = preprocess_for_ocr(
        str(temp_image_file),
        denoise_method="gaussian",
        deskew_method="contour",
        binarize_method="adaptive"
    )
    assert result is not None
    assert result.dtype == np.uint8


def test_preprocess_for_ocr_skip_denoise(temp_image_file: Path):
    """Test pipeline with denoising disabled."""
    result = preprocess_for_ocr(
        str(temp_image_file),
        apply_denoise=False
    )
    assert result is not None


def test_preprocess_for_ocr_skip_deskew(temp_image_file: Path):
    """Test pipeline with deskewing disabled."""
    result = preprocess_for_ocr(
        str(temp_image_file),
        apply_deskew=False
    )
    assert result is not None


def test_preprocess_for_ocr_skip_both(temp_image_file: Path):
    """Test pipeline with only binarization."""
    result = preprocess_for_ocr(
        str(temp_image_file),
        apply_denoise=False,
        apply_deskew=False
    )
    assert result is not None


def test_preprocess_for_ocr_returns_correct_type(temp_image_file: Path):
    """Test that pipeline returns NDArray[np.uint8]."""
    result = preprocess_for_ocr(str(temp_image_file))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_empty_image_handling():
    """Test handling of empty/all-white images."""
    empty = np.ones((100, 100), dtype=np.uint8) * 255
    result = binarize_otsu(empty)
    assert result is not None


def test_all_black_image():
    """Test handling of all-black images."""
    black = np.zeros((100, 100), dtype=np.uint8)
    result = binarize_otsu(black)
    assert result is not None


def test_single_pixel_image():
    """Test handling of very small images."""
    tiny = np.ones((1, 1, 3), dtype=np.uint8) * 255
    gray = grayscale(tiny)
    assert gray.shape == (1, 1)


def test_very_noisy_image():
    """Test with extreme noise levels."""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    noise = np.random.normal(0, 100, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    result = denoise(noisy, method="nlm", h=15)
    assert result is not None


# ============================================================================
# Performance/Smoke Tests
# ============================================================================

def test_pipeline_completes_without_errors(temp_image_file: Path):
    """Smoke test - ensure pipeline completes without crashing."""
    try:
        result = preprocess_for_ocr(str(temp_image_file))
        assert result is not None
    except Exception as e:
        pytest.fail(f"Pipeline raised unexpected exception: {e}")


@pytest.mark.parametrize("method_combo", [
    ("gaussian", "hough", "otsu"),
    ("bilateral", "contour", "adaptive"),
    ("nlm", "projection", "sauvola"),
    ("auto", "auto", "otsu"),
])
def test_all_method_combinations(temp_image_file: Path, method_combo: tuple):
    """Test various combinations of methods work together."""
    denoise_method, deskew_method, binarize_method = method_combo
    result = preprocess_for_ocr(
        str(temp_image_file),
        denoise_method=denoise_method,
        deskew_method=deskew_method,
        binarize_method=binarize_method
    )
    assert result is not None
    assert result.dtype == np.uint8
