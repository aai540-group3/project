import tempfile
from functools import partial

import gradio as gr
import openai


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
    :raises gr.Error: If input parameters are invalid or API call fails.
    """
    if not api_key.strip():
        raise gr.Error(
            "API key is required. Get an API key at: https://platform.openai.com/account/api-keys"
        )

    if not input_text.strip():
        raise gr.Error("Input text cannot be empty.")

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
        raise gr.Error(f"An OpenAI error occurred: {e}")
    except Exception as e:
        # Catch any other exceptions
        raise gr.Error(f"An unexpected error occurred: {e}")

    return temp_file_path

def main():
    """
    Main function to create and launch the Gradio interface.
    """
    MODEL_OPTIONS = ["tts-1", "tts-1-hd"]
    VOICE_OPTIONS = ["Alloy", "Echo", "Fable", "Onyx", "Nova", "Shimmer"]
    RESPONSE_FORMAT_OPTIONS = ["mp3", "opus", "aac", "flac", "wav"]
    VOICE_PREVIEW_FILES = {voice: f"{voice}.wav" for voice in VOICE_OPTIONS}

    with gr.Blocks(title="OpenAI - Text to Speech") as demo:
        with gr.Row():
            with gr.Column(scale=1):

                def play_voice_sample(voice: str):
                    """
                    Play the preview audio sample for the selected voice.

                    :param voice: The name of the voice to preview.
                    :type voice: str
                    :return: Updated Gradio Audio component with the selected voice sample.
                    :rtype: gr.Audio
                    """
                    return gr.update(
                        value=VOICE_PREVIEW_FILES[voice],
                        label=f"Preview Voice: {voice.capitalize()}",
                    )

                with gr.Group():
                    # Set the default preview audio to one of the voices
                    default_voice = "Echo"
                    preview_audio = gr.Audio(
                        interactive=False,
                        label=f"Preview Voice: {default_voice.capitalize()}",
                        value=VOICE_PREVIEW_FILES[default_voice],
                        visible=True,
                        show_download_button=False,
                        show_share_button=False,
                        autoplay=False,
                    )

                    # Create buttons for each voice
                    for voice in VOICE_OPTIONS:
                        voice_button = gr.Button(
                            value=f"{voice.capitalize()}",
                            variant="secondary",
                            size="sm",
                        )
                        voice_button.click(
                            fn=partial(play_voice_sample, voice=voice),
                            outputs=preview_audio,
                        )

            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    info="https://platform.openai.com/account/api-keys",
                    type="password",
                    placeholder="Enter your OpenAI API Key",
                )
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    label="Model",
                    value="tts-1",
                    info="Select tts-1 for speed or tts-1-hd for quality",
                )
                voice_dropdown = gr.Dropdown(
                    choices=VOICE_OPTIONS,
                    label="Voice",
                    value="Echo",
                )
                response_format_dropdown = gr.Dropdown(
                    choices=RESPONSE_FORMAT_OPTIONS,
                    label="Response Format",
                    value="mp3",
                )
                speed_slider = gr.Slider(
                    minimum=0.25,
                    maximum=4.0,
                    step=0.05,
                    label="Voice Speed",
                    value=1.0,
                )

            with gr.Column(scale=2):
                input_textbox = gr.Textbox(
                    label="Input Text (0000 / 4096 chars)",
                    lines=10,
                    placeholder="Type your text here...",
                )

                def update_label(input_text: str):
                    """
                    Update the label of the input textbox with the current character count.

                    :param input_text: The current text in the input textbox.
                    :type input_text: str
                    :return: Updated Gradio component with new label.
                    :rtype: gr.update
                    """
                    char_count = len(input_text)
                    new_label = f"Input Text ({char_count:04d} / 4096 chars)"
                    return gr.update(label=new_label)

                # Update the label when the text changes, with progress hidden
                input_textbox.change(
                    fn=update_label,
                    inputs=input_textbox,
                    outputs=input_textbox,
                    show_progress="hidden",  # Hide the progress indicator
                )

                # Initialize the submit button as non-interactive
                submit_button = gr.Button(
                    "Enter OpenAI API Key",
                    variant="primary",
                    interactive=False,
                )

                # Function to update the submit button based on API Key input
                def update_button(api_key):
                    """
                    Update the submit button's label and interactivity based on the API key input.

                    :param api_key: The current text in the API key input.
                    :type api_key: str
                    :return: Updated Gradio component for the submit button.
                    :rtype: gr.update
                    """
                    if api_key.strip():
                        # There is an API key, enable the submit button
                        return gr.update(
                            value="Convert Text to Speech", interactive=True
                        )
                    else:
                        # No API key, disable the submit button
                        return gr.update(
                            value="Enter OpenAI API Key", interactive=False
                        )

                # Update the submit button whenever the API Key input changes
                api_key_input.input(
                    fn=update_button,
                    inputs=api_key_input,
                    outputs=submit_button,
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Output Audio",
                    show_download_button=True,
                    show_share_button=False,
                )

        def on_submit(
            input_text: str,
            model: str,
            voice: str,
            api_key: str,
            response_format: str,
            speed: float,
        ) -> str:
            """
            Event handler for the submit button; converts text to speech using the tts function.

            :param input_text: The text to convert to speech.
            :type input_text: str
            :param model: The TTS model to use (e.g., 'tts-1', 'tts-1-hd').
            :type model: str
            :param voice: The voice profile to use (e.g., 'alloy', 'echo', etc.).
            :type voice: str
            :param api_key: OpenAI API key.
            :type api_key: str
            :param response_format: The audio format of the output file.
            :type response_format: str
            :param speed: The speed of the synthesized speech.
            :type speed: float
            :return: File path to the generated audio file.
            :rtype: str
            """
            audio_file = tts(
                input_text, model.lower(), voice.lower(), api_key, response_format, speed
            )
            return audio_file

        # Trigger the conversion when the submit button is clicked
        submit_button.click(
            fn=on_submit,
            inputs=[
                input_textbox,
                model_dropdown,
                voice_dropdown,
                api_key_input,
                response_format_dropdown,
                speed_slider,
            ],
            outputs=output_audio,
        )

    # Launch the Gradio app with error display enabled
    demo.launch(show_error=True)

if __name__ == "__main__":
    main()
