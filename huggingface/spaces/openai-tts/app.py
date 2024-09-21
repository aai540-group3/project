import gradio as gr
import tempfile
import openai
import requests
import os
from functools import partial

def tts(
    input_text: str,
    model: str,
    voice: str,
    api_key: str,
    response_format: str = "mp3",
    speed: float = 1.0,
) -> str:
    """
    [Function remains unchanged]
    """
    # [Function body remains unchanged]
    # ...

def main():
    """
    Main function to create and launch the Gradio interface.
    """
    MODEL_OPTIONS = ["tts-1", "tts-1-hd"]
    VOICE_OPTIONS = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    RESPONSE_FORMAT_OPTIONS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    # Predefine voice previews URLs
    VOICE_PREVIEW_URLS = {
        voice: f"https://cdn.openai.com/API/docs/audio/{voice}.wav"
        for voice in VOICE_OPTIONS
    }

    # Download audio previews to disk before initiating the interface
    PREVIEW_DIR = "voice_previews"
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    VOICE_PREVIEW_FILES = {}
    for voice, url in VOICE_PREVIEW_URLS.items():
        local_file_path = os.path.join(PREVIEW_DIR, f"{voice}.wav")
        if not os.path.exists(local_file_path):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(local_file_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {voice} preview: {e}")
        VOICE_PREVIEW_FILES[voice] = local_file_path

    # Set static paths for Gradio to serve
    gr.set_static_paths(paths=[PREVIEW_DIR])

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

                    preview_audio = gr.Audio(
                        interactive=False,
                        label="Preview Voice: Echo",
                        value=VOICE_PREVIEW_FILES['echo'],
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
                    label="Voice Options",
                    value="echo",
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
                    show_progress='hidden',  # Hide the progress indicator
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
                        return gr.update(value="Convert Text to Speech", interactive=True)
                    else:
                        # No API key, disable the submit button
                        return gr.update(value="Enter OpenAI API Key", interactive=False)

                # Update the submit button whenever the API Key input changes
                api_key_input.input(
                    fn=update_button,
                    inputs=api_key_input,
                    outputs=submit_button,
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Output Audio")

        def on_submit(
            input_text: str, model: str, voice: str, api_key: str, response_format: str, speed: float
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
                input_text, model, voice, api_key, response_format, speed
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