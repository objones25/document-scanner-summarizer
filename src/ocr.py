from abc import ABC, abstractmethod
from cv2.typing import MatLike
import base64
import io
import cv2
import pytesseract  # type: ignore[import]
from PIL import Image
import numpy as np
from mistralai import Mistral

# Typed wrappers for pytesseract functions to satisfy type checker
def _image_to_string(image: Image.Image, lang: str, config: str) -> str:
    """Typed wrapper for pytesseract.image_to_string."""
    result = pytesseract.image_to_string(image, lang=lang, config=config)  # type: ignore[attr-defined]
    if isinstance(result, str):
        return result
    return str(result)  # type: ignore[arg-type]

def _image_to_data(image: Image.Image, lang: str, config: str, output_type: int) -> dict[str, list[str | int]]:
    """Typed wrapper for pytesseract.image_to_data."""
    result = pytesseract.image_to_data(image, lang=lang, config=config, output_type=output_type)  # type: ignore[attr-defined]
    if isinstance(result, dict):
        return result  # type: ignore[return-value]
    return dict(result)  # type: ignore[arg-type]

# Get the OUTPUT.DICT constant
_OUTPUT_DICT: int = getattr(getattr(pytesseract, 'Output'), 'DICT')


class OCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: MatLike) -> str:
        """
        Extract text from preprocessed image.
        
        Args:
            image: Preprocessed image (typically binary, grayscale acceptable)
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_text_with_confidence(self, image: MatLike) -> tuple[str, float]:
        """
        Extract text with confidence score.
        
        Returns:
            Tuple of (text, confidence_score) where confidence is 0-100
        """
        pass


class TesseractOCR(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, 
                 language: str = "eng",
                 config: str = "",
                 tesseract_cmd: str | None = None):
        """
        Initialize Tesseract OCR.
        
        Args:
            language: Tesseract language code (default: "eng")
            config: Custom Tesseract config string (e.g., "--psm 6")
            tesseract_cmd: Path to tesseract executable (None = use default)
        """
        self.language = language
        self.config = config
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_text(self, image: MatLike) -> str:
        """Extract text using Tesseract."""
        # Convert OpenCV image to PIL Image for pytesseract
        pil_image = self._cv2_to_pil(image)

        # Extract text using typed wrapper
        text: str = _image_to_string(pil_image, self.language, self.config)

        return text.strip()
    
    def extract_text_with_confidence(self, image: MatLike) -> tuple[str, float]:
        """Extract text with confidence scores."""
        pil_image = self._cv2_to_pil(image)

        # Get detailed data including confidence using typed wrapper
        data: dict[str, list[str | int]] = _image_to_data(
            pil_image,
            self.language,
            self.config,
            _OUTPUT_DICT
        )

        # Extract text and average confidence
        texts: list[str] = []
        confidences: list[float] = []

        conf_list: list[str | int] = data['conf']
        text_list: list[str | int] = data['text']

        for i, conf in enumerate(conf_list):
            if isinstance(conf, int):
                conf_int: int = conf
            else:
                conf_int = int(conf)

            if conf_int != -1:  # -1 means no text detected
                text_item = text_list[i]
                if isinstance(text_item, str):
                    text_str: str = text_item
                else:
                    text_str = str(text_item)

                if text_str.strip():  # Only non-empty text
                    texts.append(text_str)
                    confidences.append(float(conf_int))

        full_text: str = ' '.join(texts)
        avg_confidence: float = float(np.mean(confidences)) if confidences else 0.0

        return full_text.strip(), avg_confidence
    
    def _cv2_to_pil(self, image: MatLike) -> Image.Image:
        """Convert OpenCV image to PIL Image."""
        # Handle different image types
        if len(image.shape) == 2:  # Grayscale
            return Image.fromarray(image)
        elif len(image.shape) == 3:  # Color
            # OpenCV is BGR, PIL expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


class MistralOCR(OCREngine):
    """
    Mistral OCR implementation using Pixtral vision models.

    Requires Mistral API key and uses their vision model for OCR.
    Better for handwriting and complex layouts.
    """

    def __init__(self, api_key: str, model: str = "mistral-ocr-latest"):
        """
        Initialize Mistral OCR.

        Args:
            api_key: Mistral API key
            model: Mistral OCR model to use (default: mistral-ocr-latest)
        """
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)

    def _encode_image_base64(self, image: MatLike) -> str:
        """
        Encode OpenCV image to base64 string for API transmission.

        Args:
            image: OpenCV image (MatLike)

        Returns:
            Base64 encoded image string with data URI prefix
        """
        # Convert OpenCV image to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        elif len(image.shape) == 3:  # Color
            # OpenCV is BGR, PIL expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Convert PIL Image to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        return f"data:image/png;base64,{base64_image}"

    def extract_text(self, image: MatLike) -> str:
        """
        Extract text using Mistral OCR API.

        Args:
            image: OpenCV image (MatLike)

        Returns:
            Extracted text as string
        """
        # Encode image to base64
        image_url = self._encode_image_base64(image)

        # Call Mistral OCR API
        ocr_response = self.client.ocr.process(
            model=self.model,
            document={
                "type": "image_url",
                "image_url": image_url
            }
        )

        # Extract text from response
        if hasattr(ocr_response, 'pages') and ocr_response.pages and len(ocr_response.pages) > 0:
            first_page = ocr_response.pages[0]
            if hasattr(first_page, 'markdown') and first_page.markdown:
                text: str = first_page.markdown
                return text.strip()

        return ""

    def extract_text_with_confidence(self, image: MatLike) -> tuple[str, float]:
        """
        Extract text with confidence (Mistral doesn't provide per-word confidence).

        Args:
            image: OpenCV image (MatLike)

        Returns:
            Tuple of (text, confidence_score) where confidence is 100 if text found, 0 otherwise
        """
        text = self.extract_text(image)
        # Mistral doesn't provide confidence scores, return 100 if successful
        return text, 100.0 if text else 0.0
