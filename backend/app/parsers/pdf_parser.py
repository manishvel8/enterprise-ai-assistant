"""
pdf_parser.py — Parse PDF files into structured content blocks.

Library: PyMuPDF (fitz) — fast, accurate, handles scanned PDFs

Why PyMuPDF over pdfplumber?
  - 10x faster than pdfplumber for text extraction
  - Better table extraction with coordinates
  - Handles encrypted and scanned PDFs
  - Returns page-level metadata (page number, dimensions)

OCR fallback:
  For scanned PDFs (no selectable text), we use Tesseract OCR.
  This is optional — if tesseract is not installed, scanned pages return empty text.
"""

import logging
from typing import List, Optional
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)


def parse_pdf(file_bytes: bytes, document_id: str) -> List[ContentBlock]:
    """
    Parse a PDF file and return a list of ContentBlocks.

    Each page becomes one or more ContentBlocks:
    - Type "text": regular paragraph/sentence text
    - Type "heading": text that appears to be a heading (large font)
    - Type "table": table data extracted as tab-separated rows

    Args:
        file_bytes: Raw PDF file content
        document_id: Used for generating block IDs

    Returns:
        List of ContentBlock objects, ordered by page number
    """
    try:
        import fitz     # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Install: pip install pymupdf")
        return _fallback_block(document_id, "PDF parser not available (install pymupdf)")

    if not file_bytes or len(file_bytes) < 5:
        return _fallback_block(document_id, "PDF file is empty or too small")
    if not file_bytes.startswith(b"%PDF"):
        return _fallback_block(
            document_id,
            f"Not a valid PDF (magic={file_bytes[:8]!r}). Check MinIO download.",
        )

    blocks: List[ContentBlock] = []
    block_index = 0

    try:
        # Prefer memory stream; fall back to temp file (more reliable for some PDFs)
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as stream_err:
            logger.warning(f"fitz.open(stream=...) failed ({stream_err}); trying tempfile")
            import tempfile
            import os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                doc = fitz.open(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        page_count = doc.page_count
        for page_num, page in enumerate(doc, start=1):
            # Extract text blocks from the page
            page_dict = page.get_text("dict")

            # --- Extract text blocks from each page ---
            page_text_parts = []

            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:    # type 0 = text block
                    for line in block.get("lines", []):
                        line_text = " ".join(
                            span.get("text", "") for span in line.get("spans", [])
                        ).strip()
                        if line_text:
                            # Detect headings: typically larger font size
                            max_font_size = max(
                                (span.get("size", 0) for span in line.get("spans", [])),
                                default=0,
                            )
                            is_heading = max_font_size >= 14     # threshold: 14pt = heading
                            page_text_parts.append((line_text, is_heading))

            # --- Group consecutive non-heading lines into paragraphs ---
            current_paragraph = []
            for text, is_heading in page_text_parts:
                if is_heading:
                    # Flush any accumulated paragraph
                    if current_paragraph:
                        blocks.append(ContentBlock(
                            block_id=f"{document_id}_block_{block_index}",
                            type="text",
                            text=" ".join(current_paragraph),
                            page_number=page_num,
                            metadata={"source": "pdf_text"},
                        ))
                        block_index += 1
                        current_paragraph = []

                    # Add heading block
                    blocks.append(ContentBlock(
                        block_id=f"{document_id}_block_{block_index}",
                        type="heading",
                        text=text,
                        page_number=page_num,
                        metadata={"source": "pdf_heading"},
                    ))
                    block_index += 1
                else:
                    current_paragraph.append(text)

            # Flush final paragraph
            if current_paragraph:
                blocks.append(ContentBlock(
                    block_id=f"{document_id}_block_{block_index}",
                    type="text",
                    text=" ".join(current_paragraph),
                    page_number=page_num,
                    metadata={"source": "pdf_text"},
                ))
                block_index += 1

            # --- Table extraction ---
            tables = page.find_tables()
            if tables and tables.tables:
                for table in tables.tables:
                    df = table.to_pandas()
                    if not df.empty:
                        # Convert DataFrame to a text representation
                        table_text = df.to_csv(sep="\t", index=False)
                        blocks.append(ContentBlock(
                            block_id=f"{document_id}_block_{block_index}",
                            type="table",
                            text=table_text,
                            page_number=page_num,
                            metadata={"source": "pdf_table", "rows": len(df), "cols": len(df.columns)},
                        ))
                        block_index += 1

            # --- OCR fallback for pages with no selectable text ---
            page_has_text = any(
                b.type in ("text", "heading", "table") and b.page_number == page_num
                for b in blocks
            )
            if not page_has_text:
                ocr_text = _try_ocr_page(page)
                if ocr_text:
                    blocks.append(ContentBlock(
                        block_id=f"{document_id}_block_{block_index}",
                        type="text",
                        text=ocr_text,
                        page_number=page_num,
                        metadata={"source": "pdf_ocr"},
                    ))
                    block_index += 1
                else:
                    logger.warning(
                        f"Page {page_num}: no text and OCR unavailable/empty "
                        "(install: sudo apt-get install tesseract-ocr && pip install pytesseract Pillow)"
                    )

        doc.close()
        logger.info(f"PDF parsed: {page_count} pages, {block_index} blocks")

        if not blocks:
            return _fallback_block(
                document_id,
                "PDF opened but no text extracted. Scanned PDFs need Tesseract OCR.",
            )

    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return _fallback_block(document_id, f"PDF parsing error: {e}")

    return blocks


def _try_ocr_page(page) -> Optional[str]:
    """
    Attempt OCR on a page using Tesseract via pytesseract.

    Returns the OCR text if successful, None otherwise.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page to an image at 150 DPI
        mat = page.get_pixmap(dpi=150)
        img_bytes = mat.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img).strip()
        return text if text else None
    except (ImportError, Exception):
        return None


def _fallback_block(document_id: str, message: str) -> List[ContentBlock]:
    """Return a single content block with an error message."""
    return [
        ContentBlock(
            block_id=f"{document_id}_block_0",
            type="text",
            text=message,
            page_number=1,
            metadata={"source": "error"},
        )
    ]
