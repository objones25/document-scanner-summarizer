import cv2
import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike
import os

def binarize_otsu(gray_img: MatLike) -> MatLike:
    """
    Apply Otsu's automatic thresholding for binarization.
    Works well when image has bimodal histogram (distinct foreground/background).
    
    Args:
        gray_img: Grayscale input image
    
    Returns:
        Binary image (black text on white background)
    """
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(gray_img: MatLike,
                      block_size: int = 11,
                      c: int = 2) -> MatLike:
    """
    Apply adaptive thresholding - excellent for uneven lighting.
    Threshold value calculated for local neighborhood of each pixel.
    
    Args:
        gray_img: Grayscale input image
        block_size: Size of pixel neighborhood (must be odd, 11-15 typical)
        c: Constant subtracted from mean (2-5 typical, higher = more aggressive)
    
    Returns:
        Binary image (black text on white background)
    """
    if block_size % 2 == 0:
        block_size += 1  # Ensure odd block size
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def binarize_sauvola(gray_img: MatLike,
                     window_size: int = 25,
                     k: float = 0.2) -> MatLike:
    """
    Apply Sauvola's local thresholding - excellent for degraded documents.
    Better than Otsu for documents with variable contrast and degradation.
    
    Args:
        gray_img: Grayscale input image
        window_size: Local window size (must be odd, 15-25 typical)
        k: Sensitivity parameter (0.2-0.5 typical, lower = more aggressive)
    
    Returns:
        Binary image (black text on white background)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate local mean and standard deviation
    mean = cv2.blur(gray_img.astype(np.float32), (window_size, window_size))
    sqr_mean = cv2.blur(gray_img.astype(np.float32) ** 2, (window_size, window_size))
    std = np.sqrt(sqr_mean - mean ** 2)
    
    # Sauvola threshold
    R = 128  # Dynamic range of standard deviation (128 for grayscale)
    threshold = mean * (1 + k * ((std / R) - 1))
    
    # Apply threshold
    binary = np.where(gray_img > threshold, 255, 0).astype(np.uint8)
    return binary


def binarize(gray_img: MatLike,
             method: str = "otsu",
             **kwargs: int) -> MatLike:
    """
    Binarize a grayscale document image using various methods.
    
    Args:
        gray_img: Grayscale input image
        method: Binarization method to use:
            - "otsu": Otsu's automatic thresholding (default, fast and reliable)
            - "adaptive": Adaptive thresholding (best for uneven lighting)
            - "sauvola": Sauvola's local thresholding (best for degraded documents)
        **kwargs: Additional integer parameters for specific methods:
            - block_size, c (int): For adaptive
            - window_size (int), k will be 0.2: For sauvola
    
    Returns:
        Binary image (black text on white background)
    """
    if method == "otsu":
        return binarize_otsu(gray_img)
    elif method == "adaptive":
        block_size: int = kwargs.get("block_size", 11)
        c: int = kwargs.get("c", 2)
        return binarize_adaptive(gray_img, block_size, c)
    elif method == "sauvola":
        window_size: int = kwargs.get("window_size", 25)
        # Note: k is float but we accept int kwargs, so we use default
        return binarize_sauvola(gray_img, window_size)
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def preprocess_for_ocr(image_path: str,
                       denoise_method: str = "bilateral",
                       deskew_method: str = "hough",
                       binarize_method: str = "otsu",
                       apply_denoise: bool = True,
                       apply_deskew: bool = True) -> NDArray[np.uint8]:
    """
    Complete preprocessing pipeline for OCR.
    
    Pipeline steps:
    1. Load image
    2. Resize if needed (for optimal OCR performance)
    3. Convert to grayscale
    4. Denoise (optional, default: bilateral filter)
    5. Deskew (optional, default: Hough transform)
    6. Binarize (Otsu's thresholding)
    
    Args:
        image_path: Path to the image file
        denoise_method: Denoising method ("gaussian", "bilateral", "nlm", or "auto")
        deskew_method: Deskewing method ("hough", "contour", "projection", or "auto")
        binarize_method: Binarization method ("otsu", "adaptive", or "sauvola")
        apply_denoise: Whether to apply denoising (default: True)
        apply_deskew: Whether to apply deskewing (default: True)
    
    Returns:
        Preprocessed binary image ready for OCR
    """
    # Load image
    img: MatLike = load_image(image_path)
    
    # Resize if needed for optimal OCR performance
    img = resize_if_needed(img)
    
    # Convert to grayscale
    gray: MatLike = grayscale(img)
    
    # Denoise (optional)
    if apply_denoise:
        gray = denoise(gray, method=denoise_method)
    
    # Deskew (optional)
    if apply_deskew:
        gray = deskew(gray, method=deskew_method)
    
    # Binarize
    binary: MatLike = binarize(gray, method=binarize_method)
    
    # Ensure return type is NDArray[np.uint8]
    return np.asarray(binary, dtype=np.uint8)


def load_image(image_path: str) -> MatLike:
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array containing the image data in BGR format
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file exists but isn't a valid image format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Invalid image format or corrupted file: {image_path}")
    
    return img

def resize_if_needed(img: MatLike, 
                     min_dimension: int = 1000,
                     max_dimension: int = 3000) -> MatLike:
    """
    Resize image only if too small (upscale) or too large (downscale).
    Maintains aspect ratio.
    
    Args:
        img: Input image
        min_dimension: Minimum height/width threshold
        max_dimension: Maximum height/width threshold for performance
    
    Returns:
        Resized image or original if already optimal size
    """
    height, width = img.shape[:2]
    min_side = min(height, width)
    max_side = max(height, width)
    
    # Too small - upscale for quality
    if min_side < min_dimension:
        scale = min_dimension / min_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Too large - downscale for performance
    elif max_side > max_dimension:
        scale = max_dimension / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Just right - no resize needed
    return img

def grayscale(img: MatLike) -> MatLike:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_skew_angle_hough(gray_img: MatLike, angle_range: int = 45) -> float:
    """
    Detect skew angle using Hough Line Transform.
    
    Args:
        gray_img: Grayscale input image
        angle_range: Maximum angle deviation to check (default ±45 degrees)
    
    Returns:
        Detected skew angle in degrees (positive = clockwise rotation needed)
    """
    # Edge detection
    edges: MatLike = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Hough Line Transform - returns array of shape (N, 1, 4) where N is number of lines
    # Type stubs say it returns MatLike, but runtime can return None when no lines detected
    lines: MatLike = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=min(gray_img.shape) // 4,
        maxLineGap=20
    )
    
    # Handle runtime None return (despite type stubs)
    try:
        if len(lines) == 0:
            return 0.0
    except TypeError:
        # lines is None (no lines detected)
        return 0.0
    
    angles: list[float] = []
    for line in lines:
        # Type narrowing: line is from MatLike (ndarray-like)
        assert isinstance(line, np.ndarray)
        # HoughLinesP returns shape (N, 1, 4), so line[0] gives us the 4 coordinates
        line_coords: NDArray[np.int32] = line[0]
        
        x1: float = float(line_coords[0])
        y1: float = float(line_coords[1])
        x2: float = float(line_coords[2])
        y2: float = float(line_coords[3])
        
        # Calculate angle in degrees
        angle: float = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        
        # Normalize angle to [-90, 90]
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
            
        # Filter angles within range
        if abs(angle) <= angle_range:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Use median to be robust against outliers
    angles_array: NDArray[np.float64] = np.array(angles, dtype=np.float64)
    return float(np.median(angles_array))


def detect_skew_angle_contour(gray_img: MatLike) -> float:
    """
    Detect skew angle using contour analysis.
    
    Args:
        gray_img: Grayscale input image
    
    Returns:
        Detected skew angle in degrees
    """
    # Threshold the image
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Get the largest contour (assumed to be the document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Normalize angle
    # cv2.minAreaRect returns angle in [-90, 0]
    if angle < -45:
        angle = 90 + angle
    
    return float(angle)


def detect_skew_angle_projection(gray_img: MatLike, 
                                  angle_range: int = 45,
                                  angle_step: float = 0.5) -> float:
    """
    Detect skew angle using projection profile analysis.
    Tests different rotation angles and finds the one with maximum variance
    in horizontal projection (indicates best alignment of text lines).
    
    Args:
        gray_img: Grayscale input image
        angle_range: Range of angles to test (±angle_range)
        angle_step: Step size for angle testing
    
    Returns:
        Detected skew angle in degrees
    """
    def calculate_projection_variance(img: MatLike) -> float:
        """Calculate variance of horizontal projection profile."""
        # Binary threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Horizontal projection (sum of pixels in each row)
        projection = np.sum(binary, axis=1)
        return float(np.var(projection))
    
    best_angle = 0.0
    max_variance = calculate_projection_variance(gray_img)
    
    # Test different angles
    angles_to_test: NDArray[np.float64] = np.arange(-angle_range, angle_range + angle_step, angle_step, dtype=np.float64)
    
    for angle_val in angles_to_test:
        angle: float = float(angle_val)
        if angle == 0:
            continue
            
        # Rotate image
        rotated = rotate_image(gray_img, angle)
        
        # Calculate variance
        variance = calculate_projection_variance(rotated)
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    return float(best_angle)


def rotate_image(img: MatLike, angle: float, border_value: int = 255) -> MatLike:
    """
    Rotate image by specified angle around its center.
    
    Args:
        img: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        border_value: Value for border pixels (default: 255 for white)
    
    Returns:
        Rotated image
    """
    if abs(angle) < 0.01:  # No rotation needed
        return img
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(
        img,
        rotation_matrix,
        (new_width, new_height),
        borderValue=border_value
    )
    
    return rotated


def denoise_gaussian(gray_img: MatLike, kernel_size: int = 5) -> MatLike:
    """
    Apply Gaussian blur to reduce noise. Fast but may blur edges slightly.
    
    Args:
        gray_img: Grayscale input image
        kernel_size: Blur kernel size (must be odd: 3, 5, 7, etc.). Larger = more blur
    
    Returns:
        Denoised image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    return cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)


def denoise_bilateral(gray_img: MatLike,
                      d: int = 9,
                      sigma_color: int = 75,
                      sigma_space: int = 75) -> MatLike:
    """
    Apply bilateral filter - reduces noise while preserving edges.
    Excellent for keeping text sharp while removing noise.
    
    Args:
        gray_img: Grayscale input image
        d: Diameter of pixel neighborhood (9-15 typical)
        sigma_color: Filter sigma in color space (higher = more colors mixed)
        sigma_space: Filter sigma in coordinate space (higher = farther pixels influence)
    
    Returns:
        Denoised image with preserved edges
    """
    return cv2.bilateralFilter(gray_img, d, sigma_color, sigma_space)


def denoise_nlm(gray_img: MatLike, h: int = 10) -> MatLike:
    """
    Apply Non-Local Means denoising - best quality but slower.
    Excellent for heavy noise from poor quality scans or photos.
    
    Args:
        gray_img: Grayscale input image
        h: Filter strength (10-15 typical, higher = more denoising but may blur text)
    
    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoising(gray_img, None, h, 7, 21)


def denoise_morphological(binary_img: MatLike, kernel_size: int = 2) -> MatLike:
    """
    Use morphological opening to remove small noise spots.
    Best applied AFTER binarization for salt-and-pepper noise.
    
    Args:
        binary_img: Binary image (black text on white background)
        kernel_size: Size of morphological kernel (2-3 typical)
    
    Returns:
        Cleaned binary image
    """
    kernel: NDArray[np.uint8] = np.ones((kernel_size, kernel_size), np.uint8)
    # Opening = erosion followed by dilation (removes small white spots)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)


def denoise(gray_img: MatLike,
            method: str = "bilateral",
            noise_level: str = "medium",
            **kwargs: int) -> MatLike:
    """
    Denoise a document image using various methods.
    
    Args:
        gray_img: Grayscale input image (or binary for morphological method)
        method: Denoising method to use:
            - "gaussian": Fast Gaussian blur (best for light noise, fastest)
            - "bilateral": Edge-preserving filter (default, balanced quality/speed)
            - "nlm": Non-Local Means (best quality, slowest)
            - "morphological": Removes isolated pixels (binary images only)
            - "auto": Automatically select method based on noise_level
        noise_level: Noise level for auto method selection:
            - "light": Uses Gaussian blur
            - "medium": Uses bilateral filter (default)
            - "heavy": Uses Non-Local Means
        **kwargs: Additional integer parameters passed to specific denoising functions:
            - kernel_size (int): For gaussian and morphological
            - d, sigma_color, sigma_space (int): For bilateral
            - h (int): For nlm
    
    Returns:
        Denoised image
    """
    if method == "gaussian":
        kernel_size: int = kwargs.get("kernel_size", 5)
        return denoise_gaussian(gray_img, kernel_size)
    elif method == "bilateral":
        d: int = kwargs.get("d", 9)
        sigma_color: int = kwargs.get("sigma_color", 75)
        sigma_space: int = kwargs.get("sigma_space", 75)
        return denoise_bilateral(gray_img, d, sigma_color, sigma_space)
    elif method == "nlm":
        h: int = kwargs.get("h", 10)
        return denoise_nlm(gray_img, h)
    elif method == "morphological":
        kernel_size: int = kwargs.get("kernel_size", 2)
        return denoise_morphological(gray_img, kernel_size)
    elif method == "auto":
        # Select method based on noise level
        if noise_level == "light":
            kernel_size: int = kwargs.get("kernel_size", 3)
            return denoise_gaussian(gray_img, kernel_size)
        elif noise_level == "medium":
            d: int = kwargs.get("d", 9)
            sigma_color: int = kwargs.get("sigma_color", 75)
            sigma_space: int = kwargs.get("sigma_space", 75)
            return denoise_bilateral(gray_img, d, sigma_color, sigma_space)
        else:  # heavy
            h: int = kwargs.get("h", 10)
            return denoise_nlm(gray_img, h)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def deskew(gray_img: MatLike,
           method: str = "hough",
           max_angle: float = 45.0) -> MatLike:
    """
    Deskew (straighten) a document image by detecting and correcting rotation.
    
    Args:
        gray_img: Grayscale input image
        method: Deskewing method to use:
            - "hough": Hough line transform (default, fast and reliable)
            - "contour": Contour-based detection (good for whole documents)
            - "projection": Projection profile analysis (most accurate but slower)
            - "auto": Try multiple methods and use consensus
        max_angle: Maximum expected skew angle in degrees (default: 45)
    
    Returns:
        Deskewed grayscale image
    """
    if method == "hough":
        angle = detect_skew_angle_hough(gray_img, angle_range=int(max_angle))
    elif method == "contour":
        angle = detect_skew_angle_contour(gray_img)
    elif method == "projection":
        angle = detect_skew_angle_projection(gray_img, angle_range=int(max_angle))
    elif method == "auto":
        # Try multiple methods and use median
        angles: list[float] = []
        try:
            angles.append(detect_skew_angle_hough(gray_img, angle_range=int(max_angle)))
        except:
            pass
        try:
            angles.append(detect_skew_angle_contour(gray_img))
        except:
            pass
        
        if angles:
            angles_array: NDArray[np.float64] = np.array(angles, dtype=np.float64)
            angle = float(np.median(angles_array))
        else:
            return gray_img
    else:
        raise ValueError(f"Unknown deskewing method: {method}")
    
    # Only rotate if angle is significant (> 0.1 degrees)
    if abs(angle) < 0.1:
        return gray_img
    
    # Rotate to correct the skew
    # Negate angle because we want to rotate in opposite direction to correct
    return rotate_image(gray_img, -angle, border_value=255)
