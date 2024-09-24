#!/usr/bin/env python3
"""
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


class PPTXtoVideo:
    """
    A class to convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg.
    """

    def __init__(self, pptx_filename: str, keep_temp: bool = False):
        """
        Initialize the PPTXtoVideo instance.

        :param pptx_filename: The path to the PowerPoint (.pptx) file.
        :type pptx_filename: str
        :param keep_temp: Whether to keep the temporary directory after conversion.
        :type keep_temp: bool
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
        openai.api_key = os.environ["OPENAI_API_KEY"]

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

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks suitable for the OpenAI TTS API.

        :param text: The text to split.
        :type text: str
        :return: A list of text chunks within the character limit.
        :rtype: List[str]
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
    async def _make_tts_api_call(self, session, chunk, i):
        """
        Make an asynchronous API call to OpenAI TTS API with retry logic.
        """
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
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

        temp_file = os.path.join(self.temp_dir, f"temp_audio_{i}.mp3")
        with open(temp_file, "wb") as f:
            f.write(content)
        return temp_file

    async def _process_chunks(self, text_chunks):
        """
        Process text chunks asynchronously, limiting concurrent API calls.
        """
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
            async def bounded_api_call(chunk, i):
                async with semaphore:
                    return await self._make_tts_api_call(session, chunk, i)

            tasks = [bounded_api_call(chunk, i) for i, chunk in enumerate(text_chunks)]
            return await asyncio.gather(*tasks)

    def text_to_speech(self, text: str, filename: str):
        """
        Converts text to speech using OpenAI TTS API and saves it as an audio file.

        :param text: The text to convert to speech.
        :type text: str
        :param filename: The output audio file path.
        :type filename: str
        """
        try:
            text_chunks = self.split_text(text)
            logger.info(f"Converting text to speech for file: {filename}")
            logger.info(f"Number of chunks: {len(text_chunks)}")

            # Use asyncio to run the API calls
            temp_audio_files = asyncio.run(self._process_chunks(text_chunks))

            # Combine audio chunks using FFmpeg
            if len(temp_audio_files) == 1:
                shutil.move(temp_audio_files[0], filename)
            else:
                concat_file = os.path.join(self.temp_dir, "concat.txt")
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
                self._run_ffmpeg_command(ffmpeg_cmd)

                # Clean up temporary audio chunk files
                for temp_file in temp_audio_files:
                    os.remove(temp_file)

        except Exception as e:
            raise RuntimeError(f"Error in text-to-speech conversion: {e}")

        # Check if audio file was created successfully
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            raise RuntimeError(f"Failed to create audio file: {filename}")

    def get_audio_duration(self, audio_file: str) -> float:
        """
        Retrieves the duration of an audio file in seconds.

        :param audio_file: The path to the audio file.
        :type audio_file: str
        :return: The duration of the audio file in seconds.
        :rtype: float
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

    def convert_pptx_to_images(self):
        """
        Converts the PowerPoint presentation to images using LibreOffice (via PDF).

        :return: A list of image file paths corresponding to the slides.
        :rtype: List[str]
        """
        # Convert PPTX to PDF
        self._convert_to_pdf()

        # Extract images from PDF
        images = convert_from_path(self.pdf_filename, dpi=300)

        image_files = []
        for i, image in enumerate(images):
            image_path = os.path.join(self.temp_dir, f"slide_{i}.png")
            image.save(image_path, "PNG")
            image_files.append(image_path)

        logger.info(f"Extracted {len(image_files)} images from PPTX (via PDF).")
        return image_files

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
        # Convert PPTX to images
        image_files = self.convert_pptx_to_images()

        for i, image_file in enumerate(image_files):
            text = self.voiceover_texts[i]
            if len(text) > MAX_CHARS:
                logger.warning(
                    f"Text for slide {i+1} exceeds {MAX_CHARS} characters. "
                    "It will be split into multiple audio files."
                )

            # Generate TTS audio for the slide
            slide_audio_filename = os.path.join(self.temp_dir, f"voice_{i}.mp3")
            self.text_to_speech(text, slide_audio_filename)

            # Get audio duration
            duration = self.get_audio_duration(slide_audio_filename)

            # Create video file combining image and audio
            slide_video_filename = os.path.join(self.temp_dir, f"video_{i}.mp4")

            # Get image dimensions
            with Image.open(image_file) as img:
                width, height = img.size

            # Adjust width to be divisible by 2
            adjusted_width = width if width % 2 == 0 else width - 1

            # FFmpeg command to create video from image and audio
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-loop", "1",
                "-i", image_file,
                "-i", slide_audio_filename,
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-vf", f"scale={adjusted_width}:-2",  # Adjust width and maintain aspect ratio
                "-shortest",
                slide_video_filename,
            ]

            try:
                self._run_ffmpeg_command(ffmpeg_cmd)
            except RuntimeError as e:
                logger.error(f"Error creating video for slide {i+1}: {e}")
                continue  # Skip to the next slide if there's an error

            # Append video file to the list
            self.video_files.append(slide_video_filename)

            logger.info(f"Created video for slide {i+1}: {slide_video_filename}")

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
        self._run_ffmpeg_command(ffmpeg_cmd)

    def _run_ffmpeg_command(self, cmd):
        """
        Runs an FFmpeg command and handles errors.

        :param cmd: The FFmpeg command to run.
        :type cmd: list
        """
        result = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

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
