"""
audio_parser.py — Transcribe audio files using OpenAI Whisper API.

Supported formats: MP3, WAV, M4A, OGG, FLAC, WebM

Why cloud Whisper instead of local Whisper?
  - cloud/API Whisper is the fastest and most accurate
  - Local Whisper requires CUDA GPU for acceptable speed
  - API is simple: POST the audio file, get the transcript back
  - Same API key as the rest of the OpenAI calls

Chunking strategy for long audio:
  - Whisper API has a 25MB file size limit
  - For files > 25MB, we split into 10-minute segments using ffmpeg
  - Each segment becomes a ContentBlock with timestamp metadata
"""

import logging
import asyncio
from typing import List
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)

WHISPER_MAX_SIZE_BYTES = 25 * 1024 * 1024      # 25 MB
SEGMENT_DURATION_SECONDS = 600                  # 10 minutes per segment


def parse_audio(file_bytes: bytes, document_id: str, file_name: str = "") -> List[ContentBlock]:
    """
    Transcribe an audio file and return ContentBlocks.

    For files under 25MB: single Whisper API call.
    For files over 25MB: split into 10-minute segments, transcribe each.

    Returns:
        ContentBlocks with transcript text and timestamp metadata.
    """
    if len(file_bytes) <= WHISPER_MAX_SIZE_BYTES:
        return _transcribe_single(file_bytes, document_id, file_name)
    else:
        return _transcribe_segmented(file_bytes, document_id, file_name)


def _transcribe_single(file_bytes: bytes, document_id: str, file_name: str) -> List[ContentBlock]:
    """
    Transcribe an audio file in a single Whisper API call.
    Uses asyncio.run() since Whisper calls use the async OpenAI client.
    """
    try:
        # Use asyncio to call the async OpenAI client from sync code
        transcript = asyncio.run(_async_transcribe(file_bytes, file_name))

        if not transcript:
            return _fallback(document_id, "Audio transcription returned empty result")

        logger.info(f"Audio transcribed: {len(transcript)} characters")
        return [ContentBlock(
            block_id=f"{document_id}_block_0",
            type="text",
            text=transcript,
            timestamp=0.0,
            metadata={
                "source": "audio_whisper",
                "file_name": file_name,
                "file_size_bytes": len(file_bytes),
            },
        )]

    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return _fallback(document_id, f"Audio transcription error: {e}")


async def _async_transcribe(file_bytes: bytes, file_name: str) -> str:
    """Call OpenAI Whisper API asynchronously."""
    from app.services.openai_service import get_openai_client
    from app.core.config import settings
    import io

    client = get_openai_client()
    file_tuple = (file_name or "audio.mp3", io.BytesIO(file_bytes), "audio/mpeg")

    response = await client.audio.transcriptions.create(
        model=settings.openai_whisper_model,
        file=file_tuple,
        response_format="text",
    )
    return response


def _transcribe_segmented(file_bytes: bytes, document_id: str, file_name: str) -> List[ContentBlock]:
    """
    Split a large audio file into segments and transcribe each.
    Requires ffmpeg to be installed.
    """
    try:
        import subprocess
        import tempfile
        import os

        blocks = []
        block_index = 0

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_input:
            tmp_input.write(file_bytes)
            tmp_input_path = tmp_input.name

        try:
            # Get total duration using ffprobe
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", tmp_input_path],
                capture_output=True, text=True, timeout=30
            )
            import json
            info = json.loads(result.stdout)
            total_duration = float(info["format"]["duration"])

            # Split into segments
            segment_start = 0
            while segment_start < total_duration:
                segment_end = min(segment_start + SEGMENT_DURATION_SECONDS, total_duration)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_seg:
                    tmp_seg_path = tmp_seg.name

                subprocess.run([
                    "ffmpeg", "-y", "-i", tmp_input_path,
                    "-ss", str(segment_start), "-t", str(SEGMENT_DURATION_SECONDS),
                    "-acodec", "libmp3lame", tmp_seg_path
                ], capture_output=True, timeout=120)

                with open(tmp_seg_path, "rb") as f:
                    segment_bytes = f.read()
                os.unlink(tmp_seg_path)

                if len(segment_bytes) > 0:
                    transcript = asyncio.run(_async_transcribe(segment_bytes, f"segment_{block_index}.mp3"))
                    if transcript:
                        blocks.append(ContentBlock(
                            block_id=f"{document_id}_block_{block_index}",
                            type="text",
                            text=transcript,
                            timestamp=segment_start,
                            metadata={
                                "source": "audio_whisper_segment",
                                "segment_start_seconds": segment_start,
                                "segment_end_seconds": segment_end,
                                "file_name": file_name,
                            },
                        ))
                        block_index += 1

                segment_start += SEGMENT_DURATION_SECONDS

        finally:
            os.unlink(tmp_input_path)

        logger.info(f"Audio segmented and transcribed: {block_index} segments")
        return blocks if blocks else _fallback(document_id, "No transcript segments produced")

    except FileNotFoundError:
        return _fallback(document_id, "ffmpeg not installed. Install: apt-get install ffmpeg")
    except Exception as e:
        logger.error(f"Segmented audio transcription failed: {e}")
        return _fallback(document_id, f"Audio segmentation error: {e}")


def _fallback(document_id: str, message: str) -> List[ContentBlock]:
    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="text",
        text=message,
        metadata={"source": "error"},
    )]
