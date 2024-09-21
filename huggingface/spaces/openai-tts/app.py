"""
This script implements a Gradio interface for text-to-speech conversion using OpenAI's API.
Users can input text, select a model and voice, and receive an audio output of the synthesized speech.

Dependencies:
    - gradio
    - openai

Usage:
    Run the script to launch a web interface for text-to-speech conversion.

Note:
    - Ensure that you have installed the required packages:
        pip install gradio openai
    - Obtain a valid OpenAI API key with access to the necessary services.
"""

import gradio as gr
import tempfile
import openai
from typing import Tuple


def tts(input_text: str, model: str, voice: str, api_key: str) -> str:
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
    :return: File path to the generated audio file.
    :rtype: str
    :raises ValueError: If input parameters are invalid.
    :raises openai.error.OpenAIError: If API call fails.
    """
    if not input_text.strip():
        raise ValueError("Input text cannot be empty.")

    if not api_key.strip():
        raise ValueError("API key is required.")

    openai.api_key = api_key

    try:
        response = openai.audio.speech.create(
            input=input_text,
            voice=voice,
            model=model
        )
    except openai.error.OpenAIError as e:
        raise e

    if not hasattr(response, 'content'):
        raise Exception("Invalid response from OpenAI API. The response does not contain audio content.")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    return temp_file_path


def on_convert_click(input_text: str, model: str, voice: str, api_key: str) -> Tuple[str, str]:
    """
    Callback function to handle the click event for text-to-speech conversion.

    :param input_text: Text input from the user.
    :type input_text: str
    :param model: Selected model.
    :type model: str
    :param voice: Selected voice.
    :type voice: str
    :param api_key: User's OpenAI API key.
    :type api_key: str
    :return: Tuple containing the file path to the generated audio file and an error message.
    :rtype: Tuple[str, str]
    """
    try:
        file_path = tts(input_text, model, voice, api_key)
        return file_path, ""
    except Exception as e:
        return None, str(e)


def main():
    """
    Main function to create and launch the Gradio interface.
    """
    # Define model and voice options
    MODEL_OPTIONS = ["tts-1", "tts-1-hd"]
    VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="API Key", type="password", placeholder="Enter your OpenAI API Key"
                )
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS, label="Model", value="tts-1"
                )
                voice_dropdown = gr.Dropdown(
                    choices=VOICE_OPTIONS, label="Voice Options", value="echo"
                )
            with gr.Column(scale=2):
                input_textbox = gr.Textbox(
                    label="Input Text",
                    lines=10,
                    placeholder="Type your text here..."
                )
                submit_button = gr.Button("Convert Text to Speech", variant="primary")
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Output Audio")
                error_output = gr.Textbox(
                    label="Error Message", interactive=False, visible=False
                )

        # Define the event handler for the submit button
        submit_button.click(
            fn=on_convert_click,
            inputs=[input_textbox, model_dropdown, voice_dropdown, api_key_input],
            outputs=[output_audio, error_output]
        )

        # Allow pressing Enter in the input textbox to trigger the conversion
        input_textbox.submit(
            fn=on_convert_click,
            inputs=[input_textbox, model_dropdown, voice_dropdown, api_key_input],
            outputs=[output_audio, error_output]
        )

    demo.launch()


if __name__ == "__main__":
    main()