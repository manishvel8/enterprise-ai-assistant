"""
video_parser.py — Extract transcript and frame text from video files.

Pipeline:
  1. Extract audio track from video using ffmpeg
  2. Transcribe audio using OpenAI Whisper → text with timestamps
  3. (Optional) Sample video frames and OCR for on-screen text

Why not use a vision model directly?
  - Video files can be 1GB+; sending frames to GPT-4V is very expensive
  - Whisper handles the audio content (usually 90%+ of video information)
  - Frame OCR is optional and only done when visual content is expected

Requires: ffmpeg binary installed (apt-get install ffmpeg)
"""

import logging
from typing import List
from app.models.document import ContentBlock

logger = logging.getLogger(__name__)


def parse_video(file_bytes: bytes, document_id: str, file_name: str = "") -> List[ContentBlock]:
    """
    Extract and transcribe audio from a video file.

    Steps:
    1. Write video bytes to a temp file
    2. Extract audio track to MP3 using ffmpeg
    3. Call parse_audio() on the extracted MP3

    Returns:
        ContentBlocks with transcript text and timestamp metadata.
    """
    try:
        import subprocess
        import tempfile
        import os
    except ImportError:
        return _fallback(document_id, "Video parsing requires subprocess module")

    try:
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video.write(file_bytes)
            tmp_video_path = tmp_video.name

        try:
            # Extract audio to MP3
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", tmp_video_path,
                    "-vn",                      # no video
                    "-acodec", "libmp3lame",
                    "-q:a", "2",                # high quality
                    tmp_audio_path,
                ],
                capture_output=True,
                timeout=300,                    # 5 minutes max
            )

            if result.returncode != 0:
                error = result.stderr.decode("utf-8", errors="replace")
                return _fallback(document_id, f"ffmpeg audio extraction failed: {error[:200]}")

            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()

            logger.info(f"Video audio extracted: {len(audio_bytes)} bytes from {len(file_bytes)} byte video")

        finally:
            os.unlink(tmp_video_path)
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

        # Transcribe the extracted audio
        from app.parsers.audio_parser import parse_audio
        blocks = parse_audio(audio_bytes, document_id, f"{file_name}_audio.mp3")

        # Update metadata to show source is video
        for block in blocks:
            block.metadata["original_source"] = "video"
            block.metadata["original_file"] = file_name

        logger.info(f"Video parsed: {len(blocks)} transcript blocks")
        return blocks

    except FileNotFoundError:
        return _fallback(document_id, "ffmpeg not installed. Install: apt-get install ffmpeg")
    except Exception as e:
        logger.error(f"Video parsing failed: {e}")
        return _fallback(document_id, f"Video parsing error: {e}")


def _fallback(document_id: str, message: str) -> List[ContentBlock]:
    return [ContentBlock(
        block_id=f"{document_id}_block_0",
        type="text",
        text=message,
        metadata={"source": "error"},
    )]
