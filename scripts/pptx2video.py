#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from typing import List

import numpy as np
from google.cloud import texttospeech as tts
from gtts import gTTS
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from moviepy.video.compositing.transitions import crossfadein, crossfadeout
from pdf2image import convert_from_path
from pptx import Presentation


class PPTXtoVideo:
    """
    A class to automate the creation of a video presentation from a PowerPoint file.
    """

    def __init__(self, pptx_filename: str):
        self.pptx_filename = pptx_filename
        self.pdf_filename = pptx_filename.replace(".pptx", ".pdf")
        self.output_file = pptx_filename.replace(".pptx", ".mp4")
        self.presentation = Presentation(pptx_filename)
        self.slides = self.presentation.slides
        self.voiceover_texts = [
            slide.notes_slide.notes_text_frame.text for slide in self.slides
        ]

    def text_to_wav(
        self, text: str, filename: str, voice_name: str = "en-US-Standard-J"
    ):
        """
        Converts the given text to speech and saves it as a .wav file.

        If the GOOGLE_APPLICATION_CREDENTIALS environment variable is set, this method uses
        Google Cloud Text-to-Speech to generate the speech. Otherwise, it uses gTTS.

        Args:
            text (str): The text to convert to speech.
            filename (str): The name of the .wav file to save the speech to.
            voice_name (str, optional): The name of the voice to use for speech generation.
                This should be a voice name from Google Cloud Text-to-Speech (e.g., "en-US-Standard-J").
                Defaults to "en-US-Standard-J".
        """
        # USE PROFESSIONAL VOICES FROM GOOGLE CLOUD
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            language_code = "-".join(voice_name.split("-")[:2])
            text_input = tts.SynthesisInput(text=text)
            voice_params = tts.VoiceSelectionParams(
                language_code=language_code, name=voice_name
            )
            audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)
            client = tts.TextToSpeechClient()

            response = client.synthesize_speech(
                input=text_input,
                voice=voice_params,
                audio_config=audio_config,
            )

            with open(filename, "wb") as out:
                out.write(response.audio_content)
        # USE FREE NON-PROFESSIONAL VOICES FROM GTTS
        else:
            voice = gTTS(text=text, lang="en", slow=False)
            voice.save(filename)

    def format_duration(self, duration: int) -> str:
        """
        Formats a duration in seconds into a string in the format 'mm:ss'.

        Args:
            duration (int): Duration in seconds.

        Returns:
            str: Formatted duration string.
        """
        minutes, seconds = divmod(int(duration), 60)
        return f"{minutes}:{seconds:02}"

    def write_metadata(self, videos: List[AudioFileClip]):
        """
        Writes metadata to a text file.

        Args:
            videos (List[AudioFileClip]): List of video clips.
        """
        total_duration = sum(video.duration for video in videos)
        with open(self.pptx_filename.replace(".pptx", ".txt"), "w") as f:
            f.write(f"Total duration: {self.format_duration(total_duration)}\n")
            for i, video in enumerate(videos):
                f.write(f"\nSlide {i+1}:\n")
                f.write(f"Duration: {self.format_duration(video.duration)}\n")
                f.write(f"Voiceover: {self.voiceover_texts[i]}\n")

    def convert_to_pdf(self):
        """
        Converts the .pptx file to a .pdf file using LibreOffice.
        """
        cmd = f"libreoffice --headless --convert-to pdf {self.pptx_filename}"
        subprocess.run(cmd, shell=True, check=True)

    def create_videos(self) -> List[AudioFileClip]:
        """
        Creates a video for each slide with a voiceover.

        Returns:
            List[AudioFileClip]: List of video clips.
        """
        videos = []
        assets_dir = "assets"
        if os.path.exists(assets_dir):
            shutil.rmtree(assets_dir)
        os.makedirs(assets_dir, exist_ok=True)
        for i, _ in enumerate(self.slides):
            text = self.voiceover_texts[i]
            images = convert_from_path(self.pdf_filename, dpi=300)
            images[i].save(f"{assets_dir}/slide_{i}.png", "PNG")
            voice_filename = f"{assets_dir}/voice_{i}.wav"
            self.text_to_wav(text, voice_filename)
            audio = AudioFileClip(voice_filename)
            # Create a silent audio clip of 0.5 seconds
            silence = AudioArrayClip(np.array([[0], [0]]), fps=44100).set_duration(0.5)
            # Add silence to the beginning and end of the audio
            audio = concatenate_audioclips([silence, audio, silence])
            img_clip = ImageClip(f"{assets_dir}/slide_{i}.png", duration=audio.duration)
            video = img_clip.set_audio(audio)
            videos.append(video)
        return videos

    def combine_videos(self, videos: List[AudioFileClip]):
        """
        Combines all the videos into one video.

        Args:
            videos (List[AudioFileClip]): List of video clips.
        """
        intro_clip = VideoFileClip("stock/intro.mp4")
        final_clip = concatenate_videoclips(videos, method="compose")
        final_clip.write_videofile(self.output_file, fps=24)

    def convert(self):
        """
        Converts the PowerPoint presentation to a video.
        """
        self.convert_to_pdf()
        videos = self.create_videos()
        self.write_metadata(videos)
        self.combine_videos(videos)


def main():
    """
    Main function to test the PPTXtoVideo class.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PowerPoint presentation to a video."
    )

    parser.add_argument(
        "pptx",
        type=str,
        help="The name of the PowerPoint file to convert.",
    )

    parser.add_argument(
        "--keyfile",
        type=str,
        help="The path to the Google service account JSON file.",
        required=False,
    )

    args = parser.parse_args()

    if args.keyfile:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.keyfile

    PPTXtoVideo(args.pptx).convert()


if __name__ == "__main__":
    main()
