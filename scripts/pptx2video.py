#!/usr/bin/env python3
"""
.. module:: scripts.pptx2video
   :synopsis: Convert PowerPoint presentations to video with AI-generated voiceovers.

This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images (via PDF intermediate).
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

The process is idempotent and utilizes both asynchronous and parallel processing to improve performance.

Usage:
    python pptx2video.py <presentation.pptx>

Requirements:
    - Python 3.7+
    - Install required packages:
        pip install python-pptx Pillow openai pdf2image aiohttp tenacity
    - LibreOffice must be installed and accessible in the system PATH.
    - FFmpeg must be installed and accessible in the system PATH.
    - Set the 'OPENAI_API_KEY' environment variable with your OpenAI API key.

Note:
    Ensure you have sufficient permissions and API quota for OpenAI TTS API.
"""

import argparse
import asyncio
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import aiohttp
import openai
from pdf2image import convert_from_path
from PIL import Image
from pptx import Presentation
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_CALLS = 5  # Maximum number of concurrent OpenAI API calls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, openai.OpenAIError)),
)
async def tts_async(input_text: str, model: str, voice: str, api_key: str) -> bytes:
    """
    Asynchronously convert input text to speech using OpenAI's Text-to-Speech API.

    :param input_text: The text to be converted to speech.
    :param model: The model to use for synthesis (e.g., 'tts-1', 'tts-1-hd').
    :param voice: The voice profile to use (e.g., 'alloy', 'echo', 'fable', etc.).
    :param api_key: OpenAI API key.
    :return: Audio content as bytes.
    :raises ValueError: If API key or input text is empty.
    :raises RuntimeError: If the API call fails.
    """
    if not api_key.strip() or not input_text.strip():
        raise ValueError("API key and input text are required.")

    logger.debug(f"Sending TTS request for text: {input_text[:50]}...")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model.lower(), "voice": voice.lower(), "input": input_text},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"API call failed with status {response.status}")
            audio_content = await response.read()
            logger.debug(f"Received audio content, size: {len(audio_content)} bytes")
            return audio_content


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def convert_slide_to_image(pdf_filename: str, output_dir: str, i: int) -> str:
    """
    Convert a single slide from the PDF to an image.

    :param pdf_filename: Path to the PDF file.
    :param output_dir: Directory to store output files.
    :param i: Slide index.
    :return: Path to the generated image file.
    """
    logger.debug(f"Converting slide {i+1} to image")
    image_path = os.path.join(output_dir, f"slide_{i}.png")
    images = convert_from_path(
        pdf_filename, first_page=i + 1, last_page=i + 1, dpi=300
    )
    if images:
        images[0].save(image_path, "PNG")
        logger.debug(f"Saved slide {i+1} image to {image_path}")
    else:
        logger.warning(f"Failed to convert slide {i+1} to image")
    return image_path


async def generate_audio_for_slide(
    text: str, text_hash: str, output_dir: str, i: int, api_key: str, state: dict
) -> Optional[str]:
    """
    Generate audio for a single slide using the TTS function.

    :param text: Text to convert to speech.
    :param text_hash: Hash of the text.
    :param output_dir: Directory to store output files.
    :param i: Slide index.
    :param api_key: OpenAI API key.
    :param state: State dictionary to track progress.
    :return: Path to the generated audio file, or None if generation fails.
    """
    slide_audio_filename = os.path.join(output_dir, f"voice_{i}.mp3")
    if (
        state["slide_hashes"].get(str(i)) == text_hash
        and os.path.exists(slide_audio_filename)
    ):
        logger.info(f"Audio for slide {i+1} is up to date, skipping generation")
        return slide_audio_filename
    try:
        logger.debug(
            f"Generating audio for slide {i+1} with text: {text[:50]}..."
        )
        audio_content = await tts_async(text, "tts-1-hd", "echo", api_key)
        with open(slide_audio_filename, "wb") as f:
            f.write(audio_content)
        logger.debug(f"Generated audio for slide {i+1}: {slide_audio_filename}")
        state["slide_hashes"][str(i)] = text_hash
        return slide_audio_filename
    except Exception as e:
        logger.error(f"Error generating audio for slide {i+1}: {e}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_video_for_slide(
    image_file: str, audio_file: Optional[str], output_dir: str, i: int, state: dict
) -> str:
    """
    Create a video for a single slide by combining image and audio.

    :param image_file: Path to the slide image file.
    :param audio_file: Path to the audio file, or None if no audio.
    :param output_dir: Directory to store output files.
    :param i: Slide index.
    :param state: State dictionary to track progress.
    :return: Path to the generated video file.
    """
    slide_video_filename = os.path.join(output_dir, f"video_{i}.mp4")
    if (
        state["videos_created"].get(str(i))
        and os.path.exists(slide_video_filename)
    ):
        logger.info(f"Video for slide {i+1} is up to date, skipping creation")
        return slide_video_filename

    logger.debug(f"Creating video for slide {i+1}")
    with Image.open(image_file) as img:
        width, height = img.size
    adjusted_width = width if width % 2 == 0 else width - 1

    if audio_file and os.path.exists(audio_file):
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            image_file,
            "-i",
            audio_file,
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={adjusted_width}:-2",
            "-shortest",
            slide_video_filename,
        ]
    else:
        logger.warning(f"No audio file for slide {i+1}, creating silent video")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            image_file,
            "-c:v",
            "libx264",
            "-t",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={adjusted_width}:-2",
            slide_video_filename,
        ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    logger.debug(f"Created video for slide {i+1}: {slide_video_filename}")
    state["videos_created"][str(i)] = True
    return slide_video_filename


class PPTXtoVideo:
    """
    A class to convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg.
    """

    def __init__(self, pptx_filename: str):
        """
        Initialize the PPTXtoVideo instance.

        :param pptx_filename: The path to the PowerPoint (.pptx) file.
        """
        self.pptx_filename = pptx_filename
        self.pptx_hash = self._compute_file_hash(pptx_filename)
        self.output_dir = os.path.join(os.getcwd(), "video-assets")
        os.makedirs(self.output_dir, exist_ok=True)
        self.pdf_filename = os.path.join(
            self.output_dir,
            f"{os.path.splitext(os.path.basename(pptx_filename))[0]}.pdf",
        )
        self.output_file = os.path.join(
            os.path.dirname(pptx_filename),
            f"{os.path.splitext(os.path.basename(pptx_filename))[0]}.mp4",
        )
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.slides_data = self._extract_slide_texts()
        self.video_files = []
        self.state_file = os.path.join(self.output_dir, "conversion_state.json")
        self.state = self._load_state()

    def _compute_file_hash(self, filename: str) -> str:
        """
        Compute a hash of the file contents.

        :param filename: Path to the file.
        :return: SHA256 hash of the file contents.
        """
        hasher = hashlib.sha256()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_slide_texts(self) -> List[dict]:
        """
        Extract text from slide notes and compute hashes.

        :return: List of dicts with slide index, text, and text hash.
        """
        slides_data = []
        for i, slide in enumerate(self.slides):
            text = (
                slide.notes_slide.notes_text_frame.text.strip()
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text
                else ""
            )
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            slides_data.append({"index": i, "text": text, "hash": text_hash})
            logger.debug(
                f"Slide {i+1} text hash: {text_hash} - text: {text[:50]}..."
            )
        return slides_data

    def _load_state(self) -> dict:
        """
        Load the conversion state from a file. Handles missing file gracefully.
        :return: Dictionary containing the conversion state, or an empty dict if the file is missing.
        """
        initial_state = {
            "pptx_hash": self.pptx_hash,
            "pdf_created": False,
            "slide_hashes": {},
            "videos_created": {},
        }
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                # Check for pptx hash match - if it doesn't match, reset the state.
                if state.get("pptx_hash") != self.pptx_hash:
                    state = initial_state
            return state
        except (FileNotFoundError, json.JSONDecodeError):
            return initial_state

    def _save_state(self):
        """Save the current conversion state to a file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def _convert_to_pdf(self):
        """Converts the .pptx file to a .pdf file using LibreOffice."""

        # Check if PDF already exists and if PPTX hash matches
        if (
            self.state["pdf_created"]
            and os.path.exists(self.pdf_filename)
            and self.state.get("pptx_hash") == self.pptx_hash
        ):
            logger.info(f"PDF already exists: {self.pdf_filename}")
            return

        logger.info("Converting PPTX to PDF")
        os.makedirs(os.path.dirname(self.pdf_filename), exist_ok=True)

        # Use tempfile to handle potential issues with long filenames
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name

            cmd = [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                os.path.dirname(temp_pdf_path),
                self.pptx_filename,
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"LibreOffice conversion output: {result.stdout}")
            logger.debug(f"LibreOffice conversion errors: {result.stderr}")

            # Move the temporary PDF to the desired location
            os.rename(temp_pdf_path, self.pdf_filename)

        if not os.path.exists(self.pdf_filename):
            raise RuntimeError(f"Failed to create PDF file: {self.pdf_filename}")

        self.state["pdf_created"] = True
        self.state["pptx_hash"] = self.pptx_hash  # Store the hash in the state
        self._save_state()
        logger.info(f"PDF created successfully: {self.pdf_filename}")

    async def create_videos(self):
        """Creates individual video files for each slide, combining slide images and TTS audio."""
        self._convert_to_pdf()

        # Convert PDF slides to images
        logger.info("Converting PDF slides to images")
        image_files = []
        images_to_generate = []

        # Collect slides that need image generation
        for i in range(len(self.slides)):
            image_file = os.path.join(self.output_dir, f"slide_{i}.png")
            if not os.path.exists(image_file):
                images_to_generate.append(i)
            image_files.append(image_file)

        if images_to_generate:
            with multiprocessing.Pool() as pool:
                convert_func = functools.partial(
                    convert_slide_to_image, self.pdf_filename, self.output_dir
                )
                for _ in pool.imap_unordered(convert_func, images_to_generate):
                    pass  # Images are saved in the function

        # Generate audio for slides (modified to check for changes)
        logger.info("Generating audio for slides")
        audio_files = []

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

        async def generate_all_audio():
            tasks = []
            for slide_data in self.slides_data:
                i = slide_data["index"]
                text = slide_data["text"]
                text_hash = slide_data["hash"]
                audio_file = os.path.join(self.output_dir, f"voice_{i}.mp3")
                audio_files.append(audio_file)

                # Check if audio needs to be regenerated
                if self.state["slide_hashes"].get(str(i)) == text_hash:
                    logger.info(
                        f"Audio for slide {i + 1} is up to date, skipping generation"
                    )
                    continue

                if not text.strip():
                    logger.warning(
                        f"Skipping audio generation for slide {i + 1} due to empty text"
                    )
                    continue

                async with semaphore:
                    tasks.append(
                        generate_audio_for_slide(
                            text, text_hash, self.output_dir, i, self.api_key, self.state
                        )
                    )
            if tasks:
                await asyncio.gather(*tasks)

        await generate_all_audio()

        # Create videos for each slide
        logger.info("Creating videos for each slide")
        max_workers = min(multiprocessing.cpu_count(), len(self.slides))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (image_file, audio_file) in enumerate(
                zip(image_files, audio_files)
            ):
                slide_hash = self.slides_data[i]["hash"]
                if (
                    self.state["videos_created"].get(str(i))
                    and self.state["slide_hashes"].get(str(i)) == slide_hash
                    and os.path.exists(
                        os.path.join(self.output_dir, f"video_{i}.mp4")
                    )
                ):
                    logger.info(f"Video for slide {i + 1} is up to date, skipping")
                    self.video_files.append(
                        os.path.join(self.output_dir, f"video_{i}.mp4")
                    )
                    continue
                futures.append(
                    executor.submit(
                        create_video_for_slide,
                        image_file,
                        audio_file,
                        self.output_dir,
                        i,
                        self.state,
                    )
                )

            for future in futures:
                video_file = future.result()
                if video_file:
                    self.video_files.append(video_file)
                    self._save_state()

        logger.info(f"Created {len(self.video_files)} individual slide videos")

    def combine_videos(self):
        """Concatenates individual slide videos into a final video."""
        if os.path.exists(self.output_file):
            logger.info(f"Final video already exists: {self.output_file}")
            return

        logger.info("Combining individual slide videos into final video")
        list_file = os.path.join(self.output_dir, "videos.txt")
        with open(list_file, "w") as f:
            for i in range(len(self.slides)):
                video_file = os.path.join(self.output_dir, f"video_{i}.mp4")
                if not os.path.exists(video_file):
                    raise FileNotFoundError(f"Missing video file: {video_file}")
                f.write(f"file '{video_file}'\n")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            self.output_file,
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.info(f"Final video created: {self.output_file}")

    async def convert(self):
        """Converts the PowerPoint presentation to a video."""
        logger.info(f"Starting conversion of {self.pptx_filename}")
        await self.create_videos()
        self.combine_videos()
        logger.info(f"Video created successfully: {self.output_file}")


async def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument(
        "pptx", type=str, help="The path to the PowerPoint (.pptx) file to convert."
    )
    args = parser.parse_args()

    try:
        logger.info(f"Starting conversion process for: {args.pptx}")
        converter = PPTXtoVideo(args.pptx)
        await converter.convert()
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
