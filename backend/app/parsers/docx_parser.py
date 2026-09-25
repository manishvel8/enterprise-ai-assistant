"""
docx_parser.py — Parse DOCX files into ContentBlocks.

Library: python-docx

Structure of a DOCX file:
  - Paragraphs: the main text units (headings + body text)
  - Tables: grid of cells, each cell contains paragraphs
  - Each paragraph has a style (Heading 1, Heading 2, Normal, etc.)

Why not extract from the raw XML?
  python-docx handles the XML parsing and gives us a clean Python API.
  We could use zipfile to read the raw XML, but that's error-prone.
"""

import logging
from typing import List
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)


def parse_docx(file_bytes: bytes, document_id: str) -> List[ContentBlock]:
    """
    Parse a DOCX file and return ContentBlocks.

    Block types:
    - "heading": Heading 1, Heading 2, etc.
    - "text": Normal paragraph text
    - "table": Table converted to tab-separated rows

    Args:
        file_bytes: Raw .docx file content
        document_id: Used for generating block IDs

    Returns:
        List of ContentBlock objects
    """
    try:
        import docx
        from io import BytesIO
    except ImportError:
        logger.error("python-docx not installed. Install: pip install python-docx")
        return _fallback(document_id, "DOCX parser not available")

    blocks: List[ContentBlock] = []
    block_index = 0

    try:
        doc = docx.Document(BytesIO(file_bytes))

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # It's a paragraph
                para = docx.text.paragraph.Paragraph(element, doc)
                text = para.text.strip()
                if not text:
                    continue

                style_name = para.style.name.lower()
                block_type = "heading" if "heading" in style_name else "text"

                blocks.append(ContentBlock(
                    block_id=f"{document_id}_block_{block_index}",
                    type=block_type,
                    text=text,
                    metadata={"style": para.style.name, "source": "docx"},
                ))
                block_index += 1

            elif tag == "tbl":
                # It's a table
                table = docx.table.Table(element, doc)
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append("\t".join(cells))
                table_text = "\n".join(rows)
                if table_text.strip():
                    blocks.append(ContentBlock(
                        block_id=f"{document_id}_block_{block_index}",
                        type="table",
                        text=table_text,
                        metadata={"source": "docx_table", "rows": len(rows)},
                    ))
                    block_index += 1

        logger.info(f"DOCX parsed: {block_index} blocks")

    except Exception as e:
        logger.error(f"DOCX parsing failed: {e}")
        return _fallback(document_id, f"DOCX parsing error: {e}")

    return blocks


def _fallback(document_id: str, message: str) -> List[ContentBlock]:
    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="text",
        text=message,
        metadata={"source": "error"},
    )]
