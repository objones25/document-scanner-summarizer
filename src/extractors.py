"""
Text extraction from various document sources.

Supports:
- Images (JPG, PNG, etc.) → preprocessing + OCR
- PDFs → text extraction or OCR fallback
- URLs → web scraping + article parsing
- DOCX files → direct text extraction
- TXT files → direct text extraction with encoding fallback
- Markdown files → direct text extraction with encoding fallback
"""

from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np

# PDF handling
import pypdf
from pdf2image import convert_from_path  # type: ignore[import-untyped]

# Web scraping
import requests
from bs4 import BeautifulSoup

# Document parsing
from docx import Document

# Local imports
from .preprocessing import preprocess_for_ocr
from .ocr import OCREngine, TesseractOCR


def extract_from_image(
    image_path: str,
    ocr_engine: OCREngine,
    preprocess: bool = True,
    **preprocessing_kwargs: Any
) -> str:
    """
    Extract text from an image file.
    
    Args:
        image_path: Path to image file
        ocr_engine: OCR engine instance to use
        preprocess: Whether to apply preprocessing (default: True)
        **preprocessing_kwargs: Additional arguments for preprocess_for_ocr
            (denoise_method, deskew_method, binarize_method, etc.)
    
    Returns:
        Extracted text
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if preprocess:
        # Apply full preprocessing pipeline
        processed_image = preprocess_for_ocr(image_path, **preprocessing_kwargs)
    else:
        # Load image directly without preprocessing
        processed_image = cv2.imread(image_path)
        if processed_image is None:
            raise ValueError(f"Could not load image: {image_path}")
    
    # Extract text
    text = ocr_engine.extract_text(processed_image)
    return text


def extract_from_pdf(
    pdf_path: str,
    ocr_engine: OCREngine,
    force_ocr: bool = False,
    dpi: int = 300,
    **preprocessing_kwargs: Any
) -> str:
    """
    Extract text from a PDF file.
    
    Strategy:
    1. Try text extraction with pypdf (fast, for text-based PDFs)
    2. If empty or force_ocr=True, convert pages to images and OCR
    
    Args:
        pdf_path: Path to PDF file
        ocr_engine: OCR engine instance to use
        force_ocr: Force OCR even if PDF has extractable text (default: False)
        dpi: DPI for PDF to image conversion (default: 300)
        **preprocessing_kwargs: Arguments for preprocessing (if OCR is used)
    
    Returns:
        Extracted text from all pages
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try text extraction first (unless forced to OCR)
    if not force_ocr:
        text = _extract_text_from_pdf(pdf_path)
        if text.strip():  # If we got meaningful text
            return text
    
    # Fallback to OCR: convert PDF pages to images
    text = _ocr_pdf_pages(pdf_path, ocr_engine, dpi, **preprocessing_kwargs)
    return text


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text directly from PDF (text-based PDFs only)."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text_parts: list[str] = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return '\n\n'.join(text_parts)
    except Exception as e:
        # If text extraction fails, return empty (will trigger OCR fallback)
        print(f"Text extraction failed: {e}")
        return ""


def _ocr_pdf_pages(
    pdf_path: str,
    ocr_engine: OCREngine,
    dpi: int = 300,
    **preprocessing_kwargs: Any
) -> str:
    """Convert PDF pages to images and OCR each page."""
    try:
        # Convert PDF to list of PIL Images
        pil_images = convert_from_path(pdf_path, dpi=dpi)

        page_texts: list[str] = []
        for i, pil_img in enumerate(pil_images):
            print(f"OCR processing page {i+1}/{len(pil_images)}...")
            
            # Convert PIL Image to numpy array (OpenCV format)
            img_array = np.array(pil_img)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Preprocess the image
            from .preprocessing import (
                resize_if_needed, grayscale, denoise, deskew, binarize
            )
            
            img_bgr = resize_if_needed(img_bgr)
            gray = grayscale(img_bgr)
            
            # Apply optional preprocessing steps
            apply_denoise: bool = bool(preprocessing_kwargs.get('apply_denoise', True))
            if apply_denoise:
                denoise_method: str = str(preprocessing_kwargs.get('denoise_method', 'bilateral'))
                gray = denoise(gray, method=denoise_method)

            apply_deskew: bool = bool(preprocessing_kwargs.get('apply_deskew', True))
            if apply_deskew:
                deskew_method: str = str(preprocessing_kwargs.get('deskew_method', 'hough'))
                gray = deskew(gray, method=deskew_method)

            binarize_method: str = str(preprocessing_kwargs.get('binarize_method', 'adaptive'))
            binary = binarize(gray, method=binarize_method)
            
            # OCR the page
            page_text = ocr_engine.extract_text(binary)
            page_texts.append(page_text)
        
        return '\n\n'.join(page_texts)
    
    except Exception as e:
        raise RuntimeError(f"Failed to OCR PDF pages: {e}")


def extract_from_url(
    url: str,
    timeout: int = 30,
    user_agent: Optional[str] = None
) -> str:
    """
    Extract text content from a web page URL.
    
    Focuses on extracting main article content, filtering out navigation,
    ads, and other non-content elements.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (default: 30)
        user_agent: Custom user agent string (default: None, uses default)
    
    Returns:
        Extracted article text
        
    Raises:
        requests.RequestException: If URL fetch fails
        ValueError: If URL is invalid
    """
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f"Invalid URL format: {url}")
    
    # Set headers
    headers: dict[str, str] = {}
    if user_agent:
        headers['User-Agent'] = user_agent
    else:
        headers['User-Agent'] = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )

    # Fetch the page
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'lxml')
    
    # Extract main content
    text = _extract_article_text(soup)
    return text


def _extract_article_text(soup: BeautifulSoup) -> str:
    """
    Extract main article content from parsed HTML.
    
    Strategy:
    1. Try common article tags (article, main)
    2. Remove non-content elements (nav, header, footer, ads)
    3. Extract text from paragraphs
    """
    # Remove unwanted elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                        'aside', 'iframe', 'noscript']):
        element.decompose()
    
    # Try to find main content area
    article_candidates = [
        soup.find('article'),
        soup.find('main'),
        soup.find('div', class_=['article', 'content', 'post', 'entry']),
        soup.find('div', id=['article', 'content', 'main'])
    ]
    
    content_area = None
    for candidate in article_candidates:
        if candidate:
            content_area = candidate
            break
    
    # If no main content area found, use body
    if not content_area:
        content_area = soup.find('body')
    
    if not content_area:
        return ""
    
    # Extract text from paragraphs and headings
    text_elements = content_area.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    # Get text from each element
    text_parts: list[str] = []
    for element in text_elements:
        text = element.get_text(strip=True)
        if text and len(text) > 20:  # Filter out very short snippets
            text_parts.append(text)

    return '\n\n'.join(text_parts)


def extract_from_docx(docx_path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        docx_path: Path to DOCX file

    Returns:
        Extracted text

    Raises:
        FileNotFoundError: If DOCX file doesn't exist
    """
    docx_path_obj = Path(docx_path)
    if not docx_path_obj.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    try:
        doc = Document(docx_path)

        # Extract text from all paragraphs
        text_parts: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_parts.append(text)

        return '\n\n'.join(text_parts)

    except Exception as e:
        raise RuntimeError(f"Failed to extract text from DOCX: {e}")


def extract_from_txt(txt_path: str) -> str:
    """
    Extract text from a plain text file.

    Args:
        txt_path: Path to TXT file

    Returns:
        Extracted text

    Raises:
        FileNotFoundError: If TXT file doesn't exist
    """
    txt_path_obj = Path(txt_path)
    if not txt_path_obj.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    try:
        with open(txt_path_obj, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(txt_path_obj, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from TXT: {e}")


def extract_from_markdown(md_path: str) -> str:
    """
    Extract text from a Markdown file.

    Reads markdown files with UTF-8 encoding, preserving markdown formatting.
    The markdown syntax will be included in the output, which is useful for
    LLMs that understand markdown structure.

    Args:
        md_path: Path to Markdown file

    Returns:
        Extracted text with markdown formatting preserved

    Raises:
        FileNotFoundError: If Markdown file doesn't exist
    """
    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    try:
        with open(md_path_obj, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(md_path_obj, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read markdown file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from Markdown: {e}")


def extract_text(
    source: str,
    ocr_engine: Optional[OCREngine] = None,
    source_type: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Universal text extractor - automatically detects source type and extracts text.

    Args:
        source: File path or URL
        ocr_engine: OCR engine to use (default: TesseractOCR with default config)
        source_type: Force source type ('image', 'pdf', 'url', 'docx', 'txt', 'markdown').
                    If None, auto-detect from file extension or URL format.
        **kwargs: Additional arguments passed to specific extractors

    Returns:
        Extracted text
    """
    # Default OCR engine if not provided
    if ocr_engine is None:
        ocr_engine = TesseractOCR()

    # Auto-detect source type
    if source_type is None:
        source_type = _detect_source_type(source)

    # Route to appropriate extractor
    if source_type == 'image':
        return extract_from_image(source, ocr_engine, **kwargs)
    elif source_type == 'pdf':
        return extract_from_pdf(source, ocr_engine, **kwargs)
    elif source_type == 'url':
        return extract_from_url(source, **kwargs)
    elif source_type == 'docx':
        return extract_from_docx(source)
    elif source_type == 'txt':
        return extract_from_txt(source)
    elif source_type == 'markdown':
        return extract_from_markdown(source)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def _detect_source_type(source: str) -> str:
    """Auto-detect source type from file extension or URL format."""
    # Check if URL
    if source.startswith(('http://', 'https://')):
        return 'url'

    # Check file extension
    source_lower = source.lower()
    if source_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        return 'image'
    elif source_lower.endswith('.pdf'):
        return 'pdf'
    elif source_lower.endswith('.docx'):
        return 'docx'
    elif source_lower.endswith('.txt'):
        return 'txt'
    elif source_lower.endswith(('.md', '.markdown')):
        return 'markdown'
    else:
        raise ValueError(f"Cannot detect source type from: {source}")