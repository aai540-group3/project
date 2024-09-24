#!/usr/bin/env python3
"""
.. module:: scripts.pptx2video
   :synopsis: Convert PowerPoint presentations to video with AI-generated voiceovers.

This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images (via PDF intermediate).
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

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

def convert_slide_to_image(pdf_filename: str, temp_dir: str, i: int) -> str:
    """
    Converts a single slide from the PDF to an image.

    :param pdf_filename: Path to the PDF file.
    :param temp_dir: Directory to save the temporary image file.
    :param i: Slide index.
    :return: Path to the saved image file.
    """
    image_path = os.path.join(temp_dir, f"slide_{i}.png")
    images = convert_from_path(pdf_filename, first_page=i+1, last_page=i+1, dpi=300)
    if images:
        images[0].save(image_path, "PNG")
    return image_path

def generate_audio_for_slide(text: str, temp_dir: str, i: int, api_key: str) -> str:
    """
    Generates audio for a single slide using text-to-speech.

    :param text: Text to convert to speech.
    :param temp_dir: Directory to save the temporary audio file.
    :param i: Slide index.
    :param api_key: OpenAI API key.
    :return: Path to the generated audio file.
    """
    slide_audio_filename = os.path.join(temp_dir, f"voice_{i}.mp3")
    text_to_speech(text, slide_audio_filename, api_key)
    return slide_audio_filename

def create_video_for_slide(image_file: str, audio_file: str, temp_dir: str, i: int) -> str:
    """
    Creates a video for a single slide by combining image and audio.

    :param image_file: Path to the slide image file.
    :param audio_file: Path to the slide audio file.
    :param temp_dir: Directory to save the temporary video file.
    :param i: Slide index.
    :return: Path to the created video file, or None if creation fails.
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
    Runs an FFmpeg command and handles errors.

    :param cmd: The FFmpeg command to run as a list of strings.
    :raises RuntimeError: If the FFmpeg command fails.
    """
    result = subprocess.run(
        cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")

def split_text(text: str) -> List[str]:
    """
    Splits text into chunks suitable for the OpenAI TTS API.

    :param text: The text to split.
    :return: A list of text chunks within the character limit.
    """
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    logger.debug(f"Original text: {text}")

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= MAX_CHARS:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                logger.debug(f"Adding chunk: {current_chunk.strip()}")
                current_chunk = ""
            # Split long sentences if needed
            if len(sentence) > MAX_CHARS:
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) <= MAX_CHARS:
                        current_chunk += word + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            logger.debug(f"Adding chunk: {current_chunk.strip()}")
                            current_chunk = ""
                        current_chunk = word + " "
            else:
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())
        logger.debug(f"Adding chunk: {current_chunk.strip()}")

    logger.info(f"Split text into {len(chunks)} chunks.")
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _make_tts_api_call(session, chunk: str, i: int, api_key: str) -> str:
    """
    Make an asynchronous API call to OpenAI TTS API with retry logic.

    :param session: aiohttp ClientSession object.
    :param chunk: Text chunk to convert to speech.
    :param i: Chunk index.
    :param api_key: OpenAI API key.
    :return: Path to the temporary audio file.
    """
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1-hd",
        "voice": "echo",
        "input": chunk
    }

    async with session.post(url, json=payload, headers=headers) as response:
        if response.status != 200:
            raise Exception(f"API call failed with status {response.status}")
        content = await response.read()

    temp_file = os.path.join(tempfile.gettempdir(), f"temp_audio_{i}.mp3")
    with open(temp_file, "wb") as f:
        f.write(content)
    return temp_file

async def _process_chunks(text_chunks: List[str], api_key: str) -> List[str]:
    """
    Process text chunks asynchronously, limiting concurrent API calls.

    :param text_chunks: List of text chunks to process.
    :param api_key: OpenAI API key.
    :return: List of paths to temporary audio files.
    """
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        async def bounded_api_call(chunk, i):
            async with semaphore:
                return await _make_tts_api_call(session, chunk, i, api_key)

        tasks = [bounded_api_call(chunk, i) for i, chunk in enumerate(text_chunks)]
        return await asyncio.gather(*tasks)

def text_to_speech(text: str, filename: str, api_key: str):
    """
    Converts text to speech using OpenAI TTS API and saves it as an audio file.

    :param text: The text to convert to speech.
    :param filename: The output audio file path.
    :param api_key: OpenAI API key.
    """
    try:
        text_chunks = split_text(text)
        logger.info(f"Converting text to speech for file: {filename}")
        logger.info(f"Number of chunks: {len(text_chunks)}")

        # Use asyncio to run the API calls
        temp_audio_files = asyncio.run(_process_chunks(text_chunks, api_key))

        # Combine audio chunks using FFmpeg
        if len(temp_audio_files) == 1:
            shutil.move(temp_audio_files[0], filename)
        else:
            concat_file = os.path.join(tempfile.gettempdir(), "concat.txt")
            with open(concat_file, "w") as f:
                for temp_file in temp_audio_files:
                    f.write(f"file '{temp_file}'\n")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                filename,
            ]
            run_ffmpeg_command(ffmpeg_cmd)

            # Clean up temporary audio chunk files
            for temp_file in temp_audio_files:
                os.remove(temp_file)

    except Exception as e:
        raise RuntimeError(f"Error in text-to-speech conversion: {e}")

    # Check if audio file was created successfully
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        raise RuntimeError(f"Failed to create audio file: {filename}")

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

    def get_audio_duration(self, audio_file: str) -> float:
        """
        Retrieves the duration of an audio file in seconds.

        :param audio_file: The path to the audio file.
        :return: The duration of the audio file in seconds.
        """
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_file,
        ]
        result = subprocess.run(
            ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(result.stdout.strip())

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

        # Convert PPTX to images in parallel
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
