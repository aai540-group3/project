#!/usr/bin/env python3
import argparse
import os
import shutil
import tempfile
from typing import List
import re

import numpy as np
from pptx import Presentation
from PIL import Image
import io
import openai
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

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
            audio_clips = []

            for chunk in text_chunks:
                response = openai.audio.speech.create(
                    model="tts-1-hd",
                    voice="echo",
                    input=chunk
                )
                temp_file = os.path.join(self.temp_dir, f"temp_audio_{len(audio_clips)}.mp3")
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                audio_clips.append(AudioFileClip(temp_file))

            # Concatenate audio clips
            final_audio = concatenate_videoclips(audio_clips)
            final_audio.write_audiofile(filename)

            # Clean up temporary files
            for clip in audio_clips:
                clip.close()
                os.remove(clip.filename)

        except Exception as e:
            raise RuntimeError(f"Error in text-to-speech conversion: {e}")

    def extract_slide_image(self, slide, dpi=300):
        width = int(self.presentation.slide_width * dpi / 72)
        height = int(self.presentation.slide_height * dpi / 72)
        image = Image.new('RGB', (width, height), 'white')
        for shape in slide.shapes:
            if hasattr(shape, 'image'):
                image_stream = io.BytesIO(shape.image.blob)
                img = Image.open(image_stream)
                img = img.resize((int(shape.width * dpi / 72), int(shape.height * dpi / 72)), Image.LANCZOS)
                image.paste(img, (int(shape.left * dpi / 72), int(shape.top * dpi / 72)))
        return image

    def create_video(self):
        video_clips = []
        for i, slide in enumerate(self.slides):
            text = self.voiceover_texts[i]
            if len(text) > MAX_CHARS:
                print(f"Warning: Text for slide {i+1} exceeds {MAX_CHARS} characters. It will be split into multiple audio files.")

            image = self.extract_slide_image(slide)
            image_filename = os.path.join(self.temp_dir, f"slide_{i}.png")
            image.save(image_filename, "PNG")

            voice_filename = os.path.join(self.temp_dir, f"voice_{i}.mp3")
            self.text_to_speech(text, voice_filename)

            audio_clip = AudioFileClip(voice_filename)
            image_clip = ImageClip(image_filename).set_duration(audio_clip.duration)
            video_clip = image_clip.set_audio(audio_clip)
            video_clips.append(video_clip)

        final_video = concatenate_videoclips(video_clips)
        final_video.write_videofile(self.output_file, fps=24)

        # Close all clips
        for clip in video_clips:
            clip.close()
        final_video.close()

    def convert(self):
        self.create_video()

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video using OpenAI TTS (Echo voice, tts-1-hd model) and MoviePy."
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