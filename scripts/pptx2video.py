#!/usr/bin/env python3
"""
.. module:: pptx2video
   :synopsis: Convert PowerPoint presentations to video with AI-generated voiceovers.

This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images (via PDF intermediate).
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

Usage:
    python pptx2video.py <presentation.pptx>

Requirements:
    - Python 3.7+
    - Required packages: python-pptx, Pillow, openai, pdf2image, aiohttp, tenacity
    - LibreOffice and FFmpeg must be installed and accessible in the system PATH.
    - Set the 'OPENAI_API_KEY' environment variable with your OpenAI API key.
"""

import argparse
import asyncio
import functools
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_CALLS = 5  # Maximum number of concurrent API calls


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, openai.OpenAIError)),
)
async def tts_async(input_text: str, model: str, voice: str, api_key: str) -> bytes:
    if not api_key.strip() or not input_text.strip():
        raise ValueError("API key and input text are required.")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model.lower(), "voice": voice.lower(), "input": input_text},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"API call failed with status {response.status}")
            return await response.read()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def convert_slide_to_image(pdf_filename: str, temp_dir: str, i: int) -> str:
    image_path = os.path.join(temp_dir, f"slide_{i}.png")
    images = convert_from_path(pdf_filename, first_page=i + 1, last_page=i + 1, dpi=300)
    if images:
        images[0].save(image_path, "PNG")
    return image_path


async def generate_audio_for_slide(
    text: str, temp_dir: str, i: int, api_key: str
) -> Optional[str]:
    slide_audio_filename = os.path.join(temp_dir, f"voice_{i}.mp3")
    try:
        audio_content = await tts_async(text, "tts-1-hd", "echo", api_key)
        with open(slide_audio_filename, "wb") as f:
            f.write(audio_content)
        return slide_audio_filename
    except Exception as e:
        logger.error(f"Error generating audio for slide {i+1}: {e}")
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_video_for_slide(
    image_file: str, audio_file: Optional[str], temp_dir: str, i: int
) -> str:
    slide_video_filename = os.path.join(temp_dir, f"video_{i}.mp4")
    with Image.open(image_file) as img:
        width, height = img.size
    adjusted_width = width if width % 2 == 0 else width - 1

    if audio_file:
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
    return slide_video_filename


class PPTXtoVideo:
    def __init__(self, pptx_filename: str):
        self.pptx_filename = pptx_filename
        self.pdf_filename = os.path.splitext(pptx_filename)[0] + ".pdf"
        self.output_file = os.path.splitext(pptx_filename)[0] + ".mp4"
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.voiceover_texts = [
            (
                slide.notes_slide.notes_text_frame.text.strip()
                if slide.has_notes_slide
                else ""
            )
            for slide in self.slides
        ]
        self.video_files = []

    def __del__(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _convert_to_pdf(self):
        project_dir = os.path.dirname(os.path.abspath(self.pptx_filename))
        pdf_path = os.path.join(project_dir, os.path.basename(self.pdf_filename))
        os.makedirs(project_dir, exist_ok=True)

        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            project_dir,
            self.pptx_filename,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if not os.path.exists(pdf_path):
            raise RuntimeError(f"Failed to create PDF file: {pdf_path}")
        self.pdf_filename = pdf_path

    async def create_videos(self):
        self._convert_to_pdf()

        with multiprocessing.Pool() as pool:
            convert_func = functools.partial(
                convert_slide_to_image, self.pdf_filename, self.temp_dir
            )
            image_files = pool.map(convert_func, range(len(self.slides)))

        async def generate_all_audio():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

            async def bounded_generate(text, i):
                if not text.strip():
                    return None
                async with semaphore:
                    return await generate_audio_for_slide(
                        text, self.temp_dir, i, self.api_key
                    )

            return await asyncio.gather(
                *[
                    bounded_generate(text, i)
                    for i, text in enumerate(self.voiceover_texts)
                ]
            )

        audio_files = await generate_all_audio()

        with ThreadPoolExecutor() as executor:
            self.video_files = list(
                executor.map(
                    create_video_for_slide,
                    image_files,
                    audio_files,
                    [self.temp_dir] * len(image_files),
                    range(len(image_files)),
                )
            )

    def combine_videos(self):
        list_file = os.path.join(self.temp_dir, "videos.txt")
        with open(list_file, "w") as f:
            for video_file in self.video_files:
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

    async def convert(self):
        await self.create_videos()
        self.combine_videos()
        logger.info(f"Video created successfully: {self.output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument(
        "pptx", type=str, help="The path to the PowerPoint (.pptx) file to convert."
    )
    args = parser.parse_args()

    try:
        converter = PPTXtoVideo(args.pptx)
        await converter.convert()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
