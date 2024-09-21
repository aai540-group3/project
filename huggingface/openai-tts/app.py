import gradio as gr
import tempfile
import openai

def tts(input_text, model, voice, api_key):
    openai.api_key = api_key
    response = openai.audio.speech.create(
        input=input_text,
        voice=voice,
        model=model
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    return temp_file_path

model_options = ["tts-1", "tts-1-hd"]
voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(label="API Key", type="password")
            model_dropdown = gr.Dropdown(choices=model_options, label="Model", value="tts-1-hd")
            voice_dropdown = gr.Dropdown(choices=voice_options, label="Voice Options", value="echo")
        with gr.Column(scale=2):
            input_textbox = gr.Textbox(
                label="Input Text",
                lines=10,
                placeholder="Type your text here..."
            )
            submit_button = gr.Button("Text-to-Speech", variant="primary")
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Output Audio")

    def on_convert_click(input_text, model, voice, api_key):
        return tts(input_text, model, voice, api_key)

    submit_button.click(
        fn=on_convert_click,
        inputs=[input_textbox, model_dropdown, voice_dropdown, api_key_input],
        outputs=output_audio
    )

    input_textbox.submit(
        fn=on_convert_click,
        inputs=[input_textbox, model_dropdown, voice_dropdown, api_key_input],
        outputs=output_audio
    )

demo.launch(debug=True, share=True)
