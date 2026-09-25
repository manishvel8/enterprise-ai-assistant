"""
pipeline/normalizer.py — Convert parser output to the common NormalizedDocument schema.

Why a normalizer?
  Every parser returns ContentBlocks, but each parser may include
  different fields (page_number for PDFs, slide_number for PPTs, etc.).

  The normalizer:
  1. Routes to the correct parser based on file type
  2. Validates and cleans the ContentBlocks
  3. Returns a NormalizedDocument with guaranteed structure

  After normalization, the chunker doesn't need to know which parser
  was used — it just processes NormalizedDocument objects.

This is the "adapter" pattern: parsers produce raw output, the normalizer
adapts it to the canonical format that the rest of the pipeline expects.
"""

import logging
from typing import List
from app.models.document import NormalizedDocument, ContentBlock

logger = logging.getLogger(__name__)


def normalize(
    file_bytes: bytes,
    document_id: str,
    file_name: str,
    file_type: str,
) -> NormalizedDocument:
    """
    Parse a document and return it in the common NormalizedDocument format.

    Args:
        file_bytes: Raw file content
        document_id: Unique document identifier
        file_name: Original filename
        file_type: Extension (pdf, docx, pptx, xlsx, csv, txt, png, jpg, mp3, mp4, wav)

    Returns:
        NormalizedDocument with a list of ContentBlocks.
    """
    logger.info(f"Normalizing {file_type.upper()}: {file_name} ({len(file_bytes)} bytes)")

    # Route to the correct parser
    blocks = _route_to_parser(file_bytes, document_id, file_name, file_type)

    # Clean and validate blocks
    blocks = _clean_blocks(blocks, document_id)

    # Log summary
    total_chars = sum(len(b.text) for b in blocks)
    logger.info(
        f"Normalized: {len(blocks)} blocks, {total_chars} total characters "
        f"[file_type={file_type}]"
    )

    return NormalizedDocument(
        document_id=document_id,
        file_name=file_name,
        file_type=file_type,
        content_blocks=blocks,
    )


def _route_to_parser(
    file_bytes: bytes,
    document_id: str,
    file_name: str,
    file_type: str,
) -> List[ContentBlock]:
    """
    Route to the appropriate parser based on file type.

    Returns a list of ContentBlocks from the parser.
    """
    file_type = file_type.lower().strip(".")

    if file_type == "pdf":
        from app.parsers.pdf_parser import parse_pdf
        return parse_pdf(file_bytes, document_id)

    elif file_type == "docx":
        from app.parsers.docx_parser import parse_docx
        return parse_docx(file_bytes, document_id)

    elif file_type == "pptx":
        from app.parsers.pptx_parser import parse_pptx
        return parse_pptx(file_bytes, document_id)

    elif file_type in ("xlsx", "xls"):
        from app.parsers.excel_parser import parse_excel
        return parse_excel(file_bytes, document_id)

    elif file_type == "csv":
        from app.parsers.csv_parser import parse_csv
        return parse_csv(file_bytes, document_id)

    elif file_type == "txt":
        return _parse_text(file_bytes, document_id)

    elif file_type in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
        from app.parsers.image_parser import parse_image
        return parse_image(file_bytes, document_id, file_name)

    elif file_type in ("mp3", "wav", "m4a", "ogg", "flac", "webm"):
        from app.parsers.audio_parser import parse_audio
        return parse_audio(file_bytes, document_id, file_name)

    elif file_type in ("mp4", "avi", "mov", "mkv", "webm"):
        from app.parsers.video_parser import parse_video
        return parse_video(file_bytes, document_id, file_name)

    else:
        logger.warning(f"Unknown file type: {file_type}. Attempting text parse.")
        return _parse_text(file_bytes, document_id)


def _parse_text(file_bytes: bytes, document_id: str) -> List[ContentBlock]:
    """
    Parse a plain text file as a single ContentBlock.

    Handles encoding detection gracefully.
    """
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = file_bytes.decode(encoding)
            return [ContentBlock(
                block_id=f"{document_id}_block_0",
                type="text",
                text=text,
                page_number=1,
                metadata={"source": "txt", "encoding": encoding},
            )]
        except UnicodeDecodeError:
            continue

    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="text",
        text="[Could not decode text file]",
        page_number=1,
        metadata={"source": "error"},
    )]


def _clean_blocks(blocks: List[ContentBlock], document_id: str) -> List[ContentBlock]:
    """
    Clean and validate ContentBlocks.

    Cleaning steps:
    1. Remove blocks with empty or whitespace-only text
    2. Strip extra whitespace from text
    3. Truncate extremely long texts (> 50,000 chars)
       to prevent chunker from creating too-large chunks
    4. Re-assign block IDs with sequential numbering
    """
    cleaned = []
    MAX_BLOCK_CHARS = 50_000

    for i, block in enumerate(blocks):
        # Strip whitespace
        text = block.text.strip() if block.text else ""

        # Skip empty blocks
        if not text:
            continue

        # Truncate very long blocks
        if len(text) > MAX_BLOCK_CHARS:
            logger.warning(
                f"Block {block.block_id} truncated from {len(text)} to {MAX_BLOCK_CHARS} chars"
            )
            text = text[:MAX_BLOCK_CHARS] + "\n[... content truncated ...]"

        # Re-assign sequential block ID
        cleaned_block = ContentBlock(
            block_id=f"{document_id}_block_{i}",
            type=block.type,
            text=text,
            page_number=block.page_number,
            slide_number=block.slide_number,
            sheet_name=block.sheet_name,
            timestamp=block.timestamp,
            metadata=block.metadata or {},
        )
        cleaned.append(cleaned_block)

    return cleaned
