#!/usr/bin/env python3
"""
.. module:: pptx2video
   :synopsis: Convert PowerPoint presentations to video with AI-generated voiceovers.

This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images (via PDF intermediate).
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

The process utilizes both asynchronous and parallel processing to improve performance,
especially for presentations with many slides. It uses multiprocessing for CPU-bound tasks,
asyncio for I/O-bound tasks, and incorporates retry logic for resilience.

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
import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, openai.OpenAIError))
)
async def tts_async(
    input_text: str,
    model: str,
    voice: str,
    api_key: str,
    response_format: str = "mp3",
    speed: float = 1.0,
) -> bytes:
    """
    Asynchronously convert input text to speech using OpenAI's Text-to-Speech API.

    :param input_text: The text to be converted to speech.
    :param model: The model to use for synthesis (e.g., 'tts-1', 'tts-1-hd').
    :param voice: The voice profile to use (e.g., 'alloy', 'echo', 'fable', etc.).
    :param api_key: OpenAI API key.
    :param response_format: The audio format of the output file, defaults to 'mp3'.
    :param speed: The speed of the synthesized speech (0.25 to 4.0), defaults to 1.0.
    :return: Audio content as bytes.
    :raises ValueError: If input parameters are invalid.
    :raises RuntimeError: If API call fails.
    """
    if not api_key.strip():
        raise ValueError("API key is required.")
    if not input_text.strip():
        raise ValueError("Input text cannot be empty.")

    logger.debug(f"Sending TTS request for text: {input_text[:50]}...")  # Log first 50 chars of input text

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model.lower(),
                "voice": voice.lower(),
                "input": input_text,
                "response_format": response_format,
                "speed": speed,
            },
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"API call failed with status {response.status}")
            return await response.read()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def convert_slide_to_image(pdf_filename: str, temp_dir: str, i: int) -> str:
    """
    Convert a single slide from the PDF to an image.

    :param pdf_filename: Path to the PDF file.
    :param temp_dir: Directory to store temporary files.
    :param i: Slide index.
    :return: Path to the generated image file.
    """
    image_path = os.path.join(temp_dir, f"slide_{i}.png")
    images = convert_from_path(pdf_filename, first_page=i+1, last_page=i+1, dpi=300)
    if images:
        images[0].save(image_path, "PNG")
        logger.debug(f"Converted slide {i+1} to image: {image_path}")
    else:
        logger.warning(f"Failed to convert slide {i+1} to image")
    return image_path

async def generate_audio_for_slide(text: str, temp_dir: str, i: int, api_key: str) -> Optional[str]:
    """
    Generate audio for a single slide using the TTS function.

    :param text: Text to convert to speech.
    :param temp_dir: Directory to store temporary files.
    :param i: Slide index.
    :param api_key: OpenAI API key.
    :return: Path to the generated audio file or None if generation fails.
    """
    slide_audio_filename = os.path.join(temp_dir, f"voice_{i}.mp3")
    try:
        logger.debug(f"Generating audio for slide {i+1} with text: {text[:50]}...")  # Log first 50 chars of text
        audio_content = await tts_async(text, "tts-1-hd", "echo", api_key)
        with open(slide_audio_filename, "wb") as f:
            f.write(audio_content)
        logger.debug(f"Generated audio for slide {i+1}: {slide_audio_filename}")
        return slide_audio_filename
    except Exception as e:
        logger.error(f"Error generating audio for slide {i+1}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_video_for_slide(image_file: str, audio_file: str, temp_dir: str, i: int) -> Optional[str]:
    """
    Create a video for a single slide by combining image and audio.

    :param image_file: Path to the slide image file.
    :param audio_file: Path to the audio file.
    :param temp_dir: Directory to store temporary files.
    :param i: Slide index.
    :return: Path to the generated video file, or None if creation fails.
    """
    if audio_file is None:
        logger.warning(f"Skipping video creation for slide {i+1} due to missing audio")
        return None

    slide_video_filename = os.path.join(temp_dir, f"video_{i}.mp4")

    with Image.open(image_file) as img:
        width, height = img.size

    adjusted_width = width if width % 2 == 0 else width - 1

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", image_file,
        "-i", audio_file,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={adjusted_width}:-2",
        "-shortest",
        slide_video_filename,
    ]

    run_ffmpeg_command(ffmpeg_cmd)
    logger.info(f"Created video for slide {i+1}: {slide_video_filename}")
    return slide_video_filename

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_ffmpeg_command(cmd: List[str]):
    """
    Run an FFmpeg command and handle errors.

    :param cmd: FFmpeg command as a list of strings.
    :raises RuntimeError: If the FFmpeg command fails.
    """
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

class PPTXtoVideo:
    """
    A class to convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg.
    """

    def __init__(self, pptx_filename: str, keep_temp: bool = False):
        """
        Initialize the PPTXtoVideo instance.

        :param pptx_filename: The path to the PowerPoint (.pptx) file.
        :param keep_temp: Whether to keep the temporary directory after conversion.
        """
        self.pptx_filename = pptx_filename
        self.pdf_filename = os.path.splitext(pptx_filename)[0] + ".pdf"
        self.output_file = os.path.splitext(pptx_filename)[0] + ".mp4"
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.keep_temp = keep_temp
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.voiceover_texts = []
        for i, slide in enumerate(self.slides):
            text = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
            self.voiceover_texts.append(text)
            logger.debug(f"Slide {i+1} text: {text[:50]}...")  # Log first 50 chars of each slide's text
        self.video_files = []

    def __del__(self):
        """Cleans up the temporary directory upon deletion of the instance."""
        if not self.keep_temp:
            try:
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _convert_to_pdf(self):
        """
        Converts the .pptx file to a .pdf file using LibreOffice.
        Saves the PDF file in the project directory (top level).
        """
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
            self.pptx_filename
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("PDF conversion command output: " + result.stdout)
        logger.info("PDF conversion command error: " + result.stderr)

        if not os.path.exists(pdf_path):
            raise RuntimeError(f"Failed to create PDF file: {pdf_path}")

        self.pdf_filename = pdf_path

    async def create_videos(self):
        """
        Creates individual video files for each slide, combining slide images and TTS audio.
        """
        self._convert_to_pdf()

        with multiprocessing.Pool() as pool:
            convert_func = functools.partial(convert_slide_to_image, self.pdf_filename, self.temp_dir)
            image_files = pool.map(convert_func, range(len(self.slides)))

        async def generate_all_audio():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
            async def bounded_generate(text, i):
                if not text.strip():
                    logger.warning(f"Skipping audio generation for slide {i+1} due to empty text.")
                    return None
                async with semaphore:
                    return await generate_audio_for_slide(text, self.temp_dir, i, self.api_key)
            return await asyncio.gather(*[bounded_generate(text, i) for i, text in enumerate(self.voiceover_texts)])

        audio_files = await generate_all_audio()

        with ThreadPoolExecutor() as executor:
            create_video_func = functools.partial(create_video_for_slide, temp_dir=self.temp_dir)
            self.video_files = list(executor.map(create_video_func, image_files, audio_files, range(len(image_files))))

        self.video_files = [v for v in self.video_files if v]
        if not self.video_files:
            raise RuntimeError("No video files were created successfully.")

    def combine_videos(self):
        """
        Concatenates individual slide videos into a final video.
        """
        list_file = os.path.join(self.temp_dir, "videos.txt")
        with open(list_file, "w") as f:
            for video_file in self.video_files:
                f.write(f"file '{video_file}'\n")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            self.output_file,
        ]
        run_ffmpeg_command(ffmpeg_cmd)

    async def convert(self):
        """
        Converts the PowerPoint presentation to a video.
        """
        try:
            await self.create_videos()
            self.combine_videos()
            logger.info(f"Video created successfully: {self.output_file}")
        except Exception as e:
            logger.error(f"An error occurred during conversion: {e}")
            if not self.keep_temp:
                shutil.rmtree(self.temp_dir)
            raise

async def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument("pptx", type=str, help="The path to the PowerPoint (.pptx) file to convert.")
    parser.add_argument("--keep_temp", action="store_true", help="Keep the temporary directory after conversion (for debugging).")
    args = parser.parse_args()

    try:
        converter = PPTXtoVideo(args.pptx, keep_temp=args.keep_temp)
        await converter.convert()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())