#!/usr/bin/env python3
"""
This script converts a PowerPoint presentation (.pptx) into a video (.mp4) by:
- Extracting slides as images.
- Generating voiceover audio using OpenAI TTS API based on slide notes.
- Combining images and audio into video segments per slide.
- Concatenating the slide videos into a final video.

Usage:
    python pptx2video.py <presentation.pptx>

Requirements:
    - Python 3.x
    - Install required packages:
        pip install python-pptx Pillow numpy openai
    - LibreOffice must be installed and accessible in the system PATH.
    - FFmpeg must be installed and accessible in the system PATH.
    - Set the 'OPENAI_API_KEY' environment variable with your OpenAI API key.

Note:
    Ensure you have sufficient permissions and API quota for OpenAI TTS API.
"""

import argparse
import os
import shutil
import tempfile
import subprocess
from typing import List
import re

from pptx import Presentation
import openai

MAX_CHARS = 4096  # Maximum characters allowed for OpenAI TTS API per request

class PPTXtoVideo:
    """
    A class to convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg.
    """

    def __init__(self, pptx_filename: str):
        """
        Initializes the PPTXtoVideo instance.

        Args:
            pptx_filename (str): The path to the PowerPoint (.pptx) file.
        """
        self.pptx_filename = pptx_filename
        self.output_file = os.path.splitext(pptx_filename)[0] + ".mp4"
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides

        # Extract voiceover texts from slide notes
        self.voiceover_texts = [
            slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
            for slide in self.slides
        ]

        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp()

        # Ensure OpenAI API key is available in environment variables
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables. "
                             "Please set it before running the script.")
        openai.api_key = os.environ['OPENAI_API_KEY']

        # Initialize list to store generated video file paths
        self.video_files = []

    def __del__(self):
        """Cleans up the temporary directory upon deletion of the instance."""
        shutil.rmtree(self.temp_dir)

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks suitable for the OpenAI TTS API.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of text chunks within the character limit.
        """
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= MAX_CHARS:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
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
                                current_chunk = ""
                            current_chunk = word + " "
                else:
                    current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def text_to_speech(self, text: str, filename: str):
        """
        Converts text to speech using OpenAI TTS API and saves it as an audio file.

        Args:
            text (str): The text to convert to speech.
            filename (str): The output audio file path.
        """
        try:
            text_chunks = self.split_text(text)
            temp_audio_files = []

            for i, chunk in enumerate(text_chunks):
                # Make the API call to OpenAI TTS API
                response = openai.audio.speech.create(
                    model="tts-1-hd",
                    voice="echo",
                    input=chunk
                )

                # Save the audio content to a temporary file
                temp_file = os.path.join(self.temp_dir, f"temp_audio_{i}.mp3")
                with open(temp_file, 'wb') as f:
                    f.write(response.content)  # Assuming 'response.content' contains the audio bytes
                temp_audio_files.append(temp_file)

            # Combine audio chunks using FFmpeg
            if len(temp_audio_files) == 1:
                shutil.move(temp_audio_files[0], filename)
            else:
                concat_file = os.path.join(self.temp_dir, "concat.txt")
                with open(concat_file, 'w') as f:
                    for temp_file in temp_audio_files:
                        f.write(f"file '{temp_file}'\n")
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
                    '-c', 'copy', filename
                ]
                subprocess.run(ffmpeg_cmd, check=True)

                # Clean up temporary audio chunk files
                for temp_file in temp_audio_files:
                    os.remove(temp_file)

        except Exception as e:
            raise RuntimeError(f"Error in text-to-speech conversion: {e}")

    def get_audio_duration(self, audio_file: str) -> float:
        """
        Retrieves the duration of an audio file in seconds.

        Args:
            audio_file (str): The path to the audio file.

        Returns:
            float: The duration of the audio file in seconds.
        """
        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    def convert_pptx_to_images(self):
        """
        Converts the PowerPoint presentation to images using LibreOffice.

        Returns:
            List[str]: A list of image file paths corresponding to the slides.
        """
        # Use LibreOffice to convert PPTX to PNG images
        cmd = [
            'libreoffice',
            '--headless',
            '--convert-to',
            'png',
            '--outdir',
            self.temp_dir,
            self.pptx_filename
        ]
        subprocess.run(cmd, check=True)

        # Collect the generated image files
        image_files = sorted(
            [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.endswith('.png')],
            key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1))
        )
        return image_files

    def create_videos(self):
        """
        Creates individual video files for each slide, combining slide images and TTS audio.
        """
        # Convert PPTX to images
        image_files = self.convert_pptx_to_images()

        for i, image_file in enumerate(image_files):
            text = self.voiceover_texts[i]
            if len(text) > MAX_CHARS:
                print(f"Warning: Text for slide {i+1} exceeds {MAX_CHARS} characters. "
                      f"It will be split into multiple audio files.")

            # Generate TTS audio for the slide
            voice_filename = os.path.join(self.temp_dir, f'voice_{i}.mp3')
            self.text_to_speech(text, voice_filename)

            # Get audio duration
            duration = self.get_audio_duration(voice_filename)

            # Create video file combining image and audio
            video_filename = os.path.join(self.temp_dir, f'video_{i}.mp4')
            # FFmpeg command to create video from image and audio
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', image_file, '-i', voice_filename,
                '-c:v', 'libx264', '-tune', 'stillimage', '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p', '-shortest', video_filename
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # Append video file to the list
            self.video_files.append(video_filename)

    def combine_videos(self):
        """
        Concatenates individual slide videos into a final video.
        """
        # Create a text file listing the video files to concatenate
        list_file = os.path.join(self.temp_dir, 'videos.txt')
        with open(list_file, 'w') as f:
            for video_file in self.video_files:
                f.write(f"file '{video_file}'\n")

        # Use FFmpeg to concatenate videos
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-c', 'copy', self.output_file
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    def convert(self):
        """
        Converts the PowerPoint presentation to a video.
        """
        self.create_videos()
        self.combine_videos()
        print(f"Video created successfully: {self.output_file}")

def main():
    """
    Main function to parse arguments and execute conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS and FFmpeg."
    )
    parser.add_argument(
        "pptx",
        type=str,
        help="The path to the PowerPoint (.pptx) file to convert.",
    )
    args = parser.parse_args()

    try:
        converter = PPTXtoVideo(args.pptx)
        converter.convert()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the OPENAI_API_KEY environment variable before running the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()