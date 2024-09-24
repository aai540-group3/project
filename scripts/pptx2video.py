#!/usr/bin/env python3
"""
.. module:: pptx2video
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
import hashlib
import json
import logging
import os
import shutil
import subprocess
from typing import List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import functools

import aiohttp
import openai
from pdf2image import convert_from_path
from pptx import Presentation
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_CONCURRENT_CALLS = 5  # Maximum number of concurrent API calls

class PPTXtoVideo:
    """
    A class to convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg.
    """

    def __init__(self, pptx_filename: str):
        """
        Initialize the PPTXtoVideo instance.

        :param pptx_filename: The path to the PowerPoint (.pptx) file.
        :type pptx_filename: str
        """
        self.pptx_filename = pptx_filename
        self.pptx_hash = self._compute_file_hash(pptx_filename)
        self.output_dir = os.path.dirname(os.path.abspath(pptx_filename))
        self.assets_dir = os.path.join(self.output_dir, "video-assets")
        self.pdf_filename = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(pptx_filename))[0]}.pdf")
        self.output_file = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(pptx_filename))[0]}.mp4")
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.temp_dir = os.path.join(self.assets_dir, f"temp_{self.pptx_hash[:8]}")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.voiceover_texts = self._extract_slide_texts()
        self.video_files = []
        self.state_file = os.path.join(self.assets_dir, f"conversion_state_{self.pptx_hash[:8]}.json")
        self.state = self._load_state()

    def _compute_file_hash(self, filename: str) -> str:
        """
        Compute a hash of the file contents.

        :param filename: The path to the file.
        :type filename: str
        :return: The hash of the file contents.
        :rtype: str
        """
        hasher = hashlib.sha256()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_slide_texts(self) -> List[str]:
        """
        Extract text from slide notes.

        :return: A list of texts extracted from slide notes.
        :rtype: List[str]
        """
        texts = []
        for i, slide in enumerate(self.slides):
            text = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
            texts.append(text)
            logger.debug(f"Slide {i+1} text: {text[:50]}...")  # Log first 50 chars of each slide's text
        return texts

    def _load_state(self) -> dict:
        """
        Load the conversion state from a file.

        :return: The conversion state.
        :rtype: dict
        """
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"pdf_created": False, "images_created": [], "audio_created": [], "videos_created": []}

    def _save_state(self):
        """Save the current conversion state to a file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def _convert_to_pdf(self):
        """
        Convert the .pptx file to a .pdf file using LibreOffice.
        """
        if self.state["pdf_created"] and os.path.exists(self.pdf_filename):
            logger.info(f"PDF already exists: {self.pdf_filename}")
            return

        logger.info("Converting PPTX to PDF")
        os.makedirs(os.path.dirname(self.pdf_filename), exist_ok=True)

        cmd = [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", os.path.dirname(self.pdf_filename), self.pptx_filename
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"LibreOffice conversion output: {result.stdout}")
        logger.debug(f"LibreOffice conversion errors: {result.stderr}")

        if not os.path.exists(self.pdf_filename):
            raise RuntimeError(f"Failed to create PDF file: {self.pdf_filename}")

        self.state["pdf_created"] = True
        self._save_state()
        logger.info(f"PDF created successfully: {self.pdf_filename}")

    async def create_videos(self):
        """
        Create individual video files for each slide, combining slide images and TTS audio.
        """
        self._convert_to_pdf()

        # Convert PDF slides to images
        logger.info("Converting PDF slides to images")
        image_files = []
        with multiprocessing.Pool() as pool:
            convert_func = functools.partial(convert_slide_to_image, self.pdf_filename, self.temp_dir)
            for i, image_path in enumerate(pool.imap(convert_func, range(len(self.slides)))):
                if i not in self.state["images_created"]:
                    image_files.append(image_path)
                    self.state["images_created"].append(i)
                    self._save_state()
                else:
                    image_files.append(os.path.join(self.temp_dir, f"slide_{i}.png"))
        logger.info(f"Created {len(image_files)} slide images")

        # Generate audio for slides
        logger.info("Generating audio for slides")
        audio_files = []
        async def generate_all_audio():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
            async def bounded_generate(text, i):
                audio_file = os.path.join(self.temp_dir, f"voice_{i}.mp3")
                if i in self.state["audio_created"] and os.path.exists(audio_file):
                    return audio_file
                if not text.strip():
                    logger.warning(f"Skipping audio generation for slide {i+1} due to empty text")
                    return None
                async with semaphore:
                    audio_file = await generate_audio_for_slide(text, self.temp_dir, i, self.api_key)
                    if audio_file:
                        self.state["audio_created"].append(i)
                        self._save_state()
                    return audio_file
            return await asyncio.gather(*[bounded_generate(text, i) for i, text in enumerate(self.voiceover_texts)])

        audio_files = await generate_all_audio()
        logger.info(f"Generated {len([a for a in audio_files if a])} audio files")

        # Create videos for each slide
        logger.info("Creating videos for each slide")
        max_workers = min(multiprocessing.cpu_count(), len(self.slides))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (image_file, audio_file) in enumerate(zip(image_files, audio_files)):
                video_file = os.path.join(self.temp_dir, f"video_{i}.mp4")
                if i not in self.state["videos_created"] or not os.path.exists(video_file):
                    futures.append(executor.submit(create_video_for_slide, image_file, audio_file, self.temp_dir, i))
                else:
                    self.video_files.append(video_file)

            for i, future in enumerate(futures):
                video_file = future.result()
                if video_file:
                    self.video_files.append(video_file)
                    self.state["videos_created"].append(i)
                    self._save_state()

        logger.info(f"Created {len(self.video_files)} individual slide videos")

    def combine_videos(self):
        """
        Concatenate individual slide videos into a final video.
        """
        if os.path.exists(self.output_file):
            logger.info(f"Final video already exists: {self.output_file}")
            return

        logger.info("Combining individual slide videos into final video")
        list_file = os.path.join(self.temp_dir, "videos.txt")
        with open(list_file, "w") as f:
            for video_file in self.video_files:
                f.write(f"file '{video_file}'\n")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", self.output_file,
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logger.info(f"Final video created: {self.output_file}")

    async def convert(self):
        """
        Convert the PowerPoint presentation to a video.
        """
        logger.info(f"Starting conversion of {self.pptx_filename}")
        await self.create_videos()
        self.combine_videos()
        logger.info(f"Video created successfully: {self.output_file}")

    def cleanup(self):
        """
        Clean up temporary files but keep the final output and assets.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")

# ... (rest of the code, including helper functions like convert_slide_to_image, generate_audio_for_slide, create_video_for_slide, etc.)

async def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument("pptx", type=str, help="The path to the PowerPoint (.pptx) file to convert.")
    args = parser.parse_args()

    try:
        logger.info(f"Starting conversion process for: {args.pptx}")
        converter = PPTXtoVideo(args.pptx)
        await converter.convert()
        converter.cleanup()
        logger.info("Conversion process completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())