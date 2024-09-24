#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
import subprocess
from typing import List
import re

import numpy as np
from pptx import Presentation
from PIL import Image
import io
import openai

MAX_CHARS = 4096  # Maximum characters allowed for OpenAI TTS API

class PPTXtoVideo:
    def __init__(self, pptx_filename: str):
        self.pptx_filename = pptx_filename
        self.output_file = pptx_filename.replace(".pptx", ".mp4")
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.voiceover_texts = [
            slide.notes_slide.notes_text_frame.text for slide in self.slides
        ]
        self.temp_dir = tempfile.mkdtemp()

        # Ensure OpenAI API key is available
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it before running the script.")
        openai.api_key = os.environ['OPENAI_API_KEY']

    def __del__(self):
        shutil.rmtree(self.temp_dir)

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks, preferring to split at sentence boundaries."""
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
                
                # If a single sentence is longer than MAX_CHARS, split it
                if len(sentence) > MAX_CHARS:
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk) + len(word) <= MAX_CHARS:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def text_to_speech(self, text: str, filename: str):
        try:
            text_chunks = self.split_text(text)
            temp_audio_files = []

            for i, chunk in enumerate(text_chunks):
                response = openai.audio.speech.create(
                    model="tts-1-hd",
                    voice="echo",
                    input=chunk
                )
                temp_file = os.path.join(self.temp_dir, f"temp_audio_{i}.mp3")
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                temp_audio_files.append(temp_file)

            # Combine audio chunks using FFmpeg
            input_files = '|'.join(temp_audio_files)
            ffmpeg_cmd = [
                'ffmpeg', '-i', f"concat:{input_files}", '-acodec', 'copy', filename
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # Clean up temporary files
            for file in temp_audio_files:
                os.remove(file)

        except Exception as e:
            raise RuntimeError(f"Error in text-to-speech conversion: {e}")

    def extract_slide_image(self, slide, dpi=300):
        image_stream = io.BytesIO()
        slide.save(image_stream, format='PNG')
        image_stream.seek(0)
        image = Image.open(image_stream)
        image = image.resize((int(image.width * dpi / 72), int(image.height * dpi / 72)), Image.LANCZOS)
        return image

    def create_videos(self):
        for i, slide in enumerate(self.slides):
            text = self.voiceover_texts[i]
            if len(text) > MAX_CHARS:
                print(f"Warning: Text for slide {i+1} exceeds {MAX_CHARS} characters. It will be split into multiple audio files.")
            
            image = self.extract_slide_image(slide)
            image_filename = os.path.join(self.temp_dir, f"slide_{i}.png")
            image.save(image_filename, "PNG")
            
            voice_filename = os.path.join(self.temp_dir, f"voice_{i}.mp3")
            self.text_to_speech(text, voice_filename)

    def combine_videos(self):
        # Create a text file with input files for FFmpeg
        input_file = os.path.join(self.temp_dir, 'input.txt')
        with open(input_file, 'w') as f:
            for i in range(len(self.slides)):
                f.write(f"file '{os.path.join(self.temp_dir, f'slide_{i}.png')}'\n")
                f.write(f"duration {self.get_audio_duration(os.path.join(self.temp_dir, f'voice_{i}.mp3'))}\n")

        # Use FFmpeg to combine images and audio into a video
        ffmpeg_cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', input_file, '-i',
            os.path.join(self.temp_dir, 'voice_*.mp3'), '-filter_complex',
            '[1:a]concat=n=' + str(len(self.slides)) + ':v=0:a=1[aout]',
            '-map', '0:v', '-map', '[aout]', '-c:v', 'libx264', '-r', '24',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
            '-shortest', self.output_file
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    def get_audio_duration(self, audio_file):
        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
        ]
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout)

    def convert(self):
        self.create_videos()
        self.combine_videos()

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video."
    )
    parser.add_argument(
        "pptx",
        type=str,
        help="The name of the PowerPoint file to convert.",
    )
    args = parser.parse_args()

    try:
        PPTXtoVideo(args.pptx).convert()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the OPENAI_API_KEY environment variable before running the script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
