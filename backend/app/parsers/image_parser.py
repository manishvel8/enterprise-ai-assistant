"""
image_parser.py — Extract text from images using OCR (Optical Character Recognition).

Library: pytesseract + Pillow

OCR pipeline:
  1. Load image with Pillow
  2. Preprocess: convert to grayscale, enhance contrast (improves OCR accuracy)
  3. Run Tesseract OCR
  4. Return extracted text as a ContentBlock

Why Tesseract?
  - Open-source, free, widely used
  - Supports 100+ languages
  - Works offline (no API call required)
  - Good accuracy for printed text

Limitation:
  - Handwritten text accuracy is low (use a vision LLM for better results)
  - Requires tesseract binary installed: apt-get install tesseract-ocr
"""

import logging
from typing import List
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)


def parse_image(file_bytes: bytes, document_id: str, file_name: str = "") -> List[ContentBlock]:
    """
    Extract text from an image file using Tesseract OCR.

    Returns a single ContentBlock with the extracted text.
    If OCR fails or returns no text, returns an empty-text block.
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        from io import BytesIO
    except ImportError as e:
        logger.error(f"Image OCR deps not installed: {e}")
        logger.error("Install: pip install pytesseract Pillow")
        logger.error("Also install Tesseract binary: apt-get install tesseract-ocr")
        return _fallback(document_id, "Image OCR not available")

    try:
        # Load image
        img = Image.open(BytesIO(file_bytes))

        # Preprocess for better OCR accuracy
        img = img.convert("L")                                  # grayscale
        img = ImageEnhance.Contrast(img).enhance(2.0)           # increase contrast
        img = img.filter(ImageFilter.SHARPEN)                   # sharpen edges

        # Run OCR
        config = "--psm 3"     # PSM 3 = fully automatic page segmentation (default)
        text = pytesseract.image_to_string(img, config=config).strip()

        if not text:
            logger.warning(f"OCR returned empty text for {file_name}")
            text = "[No text detected in image]"

        logger.info(f"Image OCR complete: {len(text)} characters extracted")

        return [ContentBlock(
            block_id=f"{document_id}_block_0",
            type="text",
            text=text,
            metadata={
                "source": "image_ocr",
                "file_name": file_name,
                "image_size": f"{img.size[0]}x{img.size[1]}",
            },
        )]

    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return _fallback(document_id, f"Image OCR error: {e}")


def _fallback(document_id: str, message: str) -> List[ContentBlock]:
    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="text",
        text=message,
        metadata={"source": "error"},
    )]
