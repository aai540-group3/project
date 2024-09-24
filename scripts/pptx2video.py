#!/usr/bin/env python3
"""
.. module:: pptx2video
   :synopsis: Convert PowerPoint presentations to video with AI-generated voiceovers.

This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images (via PDF intermediate).
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

The process utilizes parallel processing to improve performance, especially for presentations
with many slides. It uses multiprocessing for CPU-bound tasks and ThreadPoolExecutor for I/O-bound tasks.

Usage:
    python pptx2video.py <presentation.pptx>

Requirements:
    - Python 3.x
    - Install required packages:
        pip install python-pptx Pillow numpy openai pdf2image aiohttp tenacity
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
import re
import shutil
import subprocess
import tempfile
from typing import List
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

import aiohttp
import openai
from pdf2image import convert_from_path
from pptx import Presentation
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAX_CHARS = 4096  # Maximum characters allowed for OpenAI TTS API per request
MAX_CONCURRENT_CALLS = 5  # Maximum number of concurrent API calls

def tts(
    input_text: str,
    model: str,
    voice: str,
    api_key: str,
    response_format: str = "mp3",
    speed: float = 1.0,
) -> str:
    """
    Convert input text to speech using OpenAI's Text-to-Speech API.

    :param input_text: The text to be converted to speech.
    :type input_text: str
    :param model: The model to use for synthesis (e.g., 'tts-1', 'tts-1-hd').
    :type model: str
    :param voice: The voice profile to use (e.g., 'alloy', 'echo', 'fable', etc.).
    :type voice: str
    :param api_key: OpenAI API key.
    :type api_key: str
    :param response_format: The audio format of the output file, defaults to 'mp3'.
    :type response_format: str, optional
    :param speed: The speed of the synthesized speech (0.25 to 4.0), defaults to 1.0.
    :type speed: float, optional
    :return: File path to the generated audio file.
    :rtype: str
    :raises RuntimeError: If input parameters are invalid or API call fails.
    """
    if not api_key.strip():
        raise RuntimeError(
            "API key is required. Get an API key at: https://platform.openai.com/account/api-keys"
        )

    if not input_text.strip():
        raise RuntimeError("Input text cannot be empty.")

    openai.api_key = api_key

    try:
        # Create the audio speech object
        speech_file = openai.audio.speech.create(
            model=model.lower(),
            voice=voice.lower(),
            input=input_text,
            response_format=response_format,
            speed=speed,
        )
        # Save the audio content to a temporary file
        file_extension = f".{response_format}"
        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False, mode="wb"
        ) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(speech_file.content)

    except openai.OpenAIError as e:
        # Catch OpenAI exceptions
        raise RuntimeError(f"An OpenAI error occurred: {e}")
    except Exception as e:
        # Catch any other exceptions
        raise RuntimeError(f"An unexpected error occurred: {e}")

    return temp_file_path

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
    return image_path

def generate_audio_for_slide(text: str, temp_dir: str, i: int, api_key: str) -> str:
    """
    Generate audio for a single slide using the TTS function.

    :param text: Text to convert to speech.
    :param temp_dir: Directory to store temporary files.
    :param i: Slide index.
    :param api_key: OpenAI API key.
    :return: Path to the generated audio file.
    """
    slide_audio_filename = os.path.join(temp_dir, f"voice_{i}.mp3")
    audio_file = tts(text, "tts-1-hd", "echo", api_key)
    shutil.move(audio_file, slide_audio_filename)
    return slide_audio_filename

def create_video_for_slide(image_file: str, audio_file: str, temp_dir: str, i: int) -> str:
    """
    Create a video for a single slide by combining image and audio.

    :param image_file: Path to the slide image file.
    :param audio_file: Path to the audio file.
    :param temp_dir: Directory to store temporary files.
    :param i: Slide index.
    :return: Path to the generated video file, or None if creation fails.
    """
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

    try:
        run_ffmpeg_command(ffmpeg_cmd)
    except RuntimeError as e:
        logger.error(f"Error creating video for slide {i+1}: {e}")
        return None

    logger.info(f"Created video for slide {i+1}: {slide_video_filename}")
    return slide_video_filename

def run_ffmpeg_command(cmd: List[str]):
    """
    Run an FFmpeg command and handle errors.

    :param cmd: FFmpeg command as a list of strings.
    :raises RuntimeError: If the FFmpeg command fails.
    """
    result = subprocess.run(
        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
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

        # Extract voiceover texts from slide notes
        self.voiceover_texts = [
            slide.notes_slide.notes_text_frame.text.strip()
            if slide.has_notes_slide
            else ""
            for slide in self.slides
        ]

        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()

        # Ensure OpenAI API key is available in environment variables
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it before running the script."
            )
        self.api_key = os.environ["OPENAI_API_KEY"]

        # Initialize list to store generated video file paths
        self.video_files = []

    def __del__(self):
        """Cleans up the temporary directory upon deletion of the instance."""
        if not self.keep_temp:
            try:
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")

    def _convert_to_pdf(self):
        """
        Converts the .pptx file to a .pdf file using LibreOffice.
        Saves the PDF file in the project directory (top level).
        """
        project_dir = os.path.dirname(os.path.abspath(self.pptx_filename))
        pdf_path = os.path.join(project_dir, os.path.basename(self.pdf_filename))

        # Ensure the output directory exists
        os.makedirs(project_dir, exist_ok=True)

        # Construct the command as a list of arguments
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            project_dir,
            self.pptx_filename
        ]

        try:
            # Run the command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("PDF conversion command output: " + result.stdout)
            logger.info("PDF conversion command error: " + result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error("Command output: " + e.output)
            logger.error("Command error: " + e.stderr)
            raise RuntimeError(f"Failed to convert PPTX to PDF: {e}")

        # Verify if the PDF file was created successfully
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"Failed to create PDF file: {pdf_path}")

        self.pdf_filename = pdf_path  # Update the pdf_filename attribute

    def create_videos(self):
        """
        Creates individual video files for each slide, combining slide images and TTS audio.
        """
        # Convert PPTX to PDF
        self._convert_to_pdf()

        # Convert PDF to images in parallel
        with multiprocessing.Pool() as pool:
            convert_func = functools.partial(convert_slide_to_image, self.pdf_filename, self.temp_dir)
            image_files = pool.map(convert_func, range(len(self.slides)))

        # Generate TTS audio for all slides in parallel
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
            generate_audio_func = functools.partial(generate_audio_for_slide, temp_dir=self.temp_dir, api_key=self.api_key)
            audio_futures = {
                executor.submit(generate_audio_func, text, i): i
                for i, text in enumerate(self.voiceover_texts)
            }

            audio_files = [None] * len(self.voiceover_texts)
            for future in as_completed(audio_futures):
                i = audio_futures[future]
                audio_files[i] = future.result()

        # Create videos in parallel
        with multiprocessing.Pool() as pool:
            create_video_func = functools.partial(create_video_for_slide, temp_dir=self.temp_dir)
            self.video_files = pool.starmap(
                create_video_func,
                zip(image_files, audio_files, range(len(image_files)))
            )

        # Remove None values (failed video creations)
        self.video_files = [v for v in self.video_files if v]

        if not self.video_files:
            raise RuntimeError("No video files were created successfully.")

    def combine_videos(self):
        """
        Concatenates individual slide videos into a final video.
        """
        # Create a text file listing the video files to concatenate
        list_file = os.path.join(self.temp_dir, "videos.txt")
        with open(list_file, "w") as f:
            for video_file in self.video_files:
                f.write(f"file '{video_file}'\n")

        # Use FFmpeg to concatenate videos
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
        run_ffmpeg_command(ffmpeg_cmd)

    def convert(self):
        """
        Converts the PowerPoint presentation to a video.
        """
        try:
            self.create_videos()
            self.combine_videos()
            logger.info(f"Video created successfully: {self.output_file}")
        except Exception as e:
            logger.error(f"An error occurred during conversion: {e}")
            # Clean up temporary files on error
            if not self.keep_temp:
                shutil.rmtree(self.temp_dir)
            raise

def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument(
        "pptx", type=str, help="The path to the PowerPoint (.pptx) file to convert."
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep the temporary directory after conversion (for debugging).",
    )
    args = parser.parse_args()

    try:
        converter = PPTXtoVideo(args.pptx, keep_temp=args.keep_temp)
        converter.convert()
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error(
            "Please set the OPENAI_API_KEY environment variable before running the script."
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
