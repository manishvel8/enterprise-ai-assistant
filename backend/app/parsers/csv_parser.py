"""
csv_parser.py — Parse CSV files into ContentBlocks.

Library: pandas (best for CSVs — handles encoding, quoting, mixed types)

Strategy:
  - Detect encoding automatically
  - Split large CSVs into chunks of 500 rows
    (a 10,000 row CSV shouldn't be one giant text block)
  - Include column names in each chunk for context
"""

import logging
from typing import List
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)

MAX_ROWS_PER_BLOCK = 500    # rows per ContentBlock


def parse_csv(file_bytes: bytes, document_id: str) -> List[ContentBlock]:
    """
    Parse a CSV file and return ContentBlocks.

    Large CSVs are split into blocks of MAX_ROWS_PER_BLOCK rows.
    Each block includes the header row for context.
    """
    try:
        import pandas as pd
        from io import BytesIO
    except ImportError:
        logger.error("pandas not installed. Install: pip install pandas")
        return _fallback(document_id, "CSV parser not available")

    blocks: List[ContentBlock] = []

    try:
        # Try multiple encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return _fallback(document_id, "Could not decode CSV file")

        if df.empty:
            return _fallback(document_id, "CSV file is empty")

        total_rows = len(df)
        header = list(df.columns)

        # Split into chunks
        for chunk_start in range(0, total_rows, MAX_ROWS_PER_BLOCK):
            chunk_df = df.iloc[chunk_start:chunk_start + MAX_ROWS_PER_BLOCK]
            chunk_text = chunk_df.to_csv(index=False, sep="\t")

            block_index = chunk_start // MAX_ROWS_PER_BLOCK
            row_range = f"rows {chunk_start + 1}–{min(chunk_start + MAX_ROWS_PER_BLOCK, total_rows)}"
            header_note = f"[Columns: {', '.join(str(h) for h in header)}] [{row_range} of {total_rows}]\n"

            blocks.append(ContentBlock(
                block_id=f"{document_id}_block_{block_index}",
                type="table",
                text=header_note + chunk_text,
                metadata={
                    "source": "csv",
                    "total_rows": total_rows,
                    "columns": header,
                    "chunk_start_row": chunk_start + 1,
                    "chunk_end_row": min(chunk_start + MAX_ROWS_PER_BLOCK, total_rows),
                },
            ))

        logger.info(f"CSV parsed: {total_rows} rows, {len(blocks)} blocks")

    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        return _fallback(document_id, f"CSV parsing error: {e}")

    return blocks


def _fallback(document_id: str, message: str) -> List[ContentBlock]:
    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="table",
        text=message,
        metadata={"source": "error"},
    )]
