import gradio as gr
import openai
import os

def transcribe_audio(audio_file, api_key):
    # Set the OpenAI API key
    openai.api_key = api_key
    
    try:
        # Open the audio file
        with open(audio_file, "rb") as audio:
            # Transcribe using Whisper large v3 (called "whisper-1" in the API)
            transcript = openai.Audio.transcribe(
                model="whisper-v3-large",
                file=audio,
                response_format="text"
            )
        return transcript
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File"),
        gr.Textbox(label="OpenAI API Key", type="password")
    ],
    outputs=gr.Textbox(label="Transcription"),
    title="Audio Transcription with Whisper v3",
    description="Upload an audio file (mp3, wav, etc.) to transcribe it using OpenAI's Whisper large v3 model."
)

# Run the app
if __name__ == "__main__":
    interface.launch()