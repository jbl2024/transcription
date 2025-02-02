import gradio as gr
from openai import OpenAI
client = OpenAI()
import os
from transcription_service import transcript

def transcribe_audio_dinum(audio_file):
    try:
        result = transcript(audio_file)
        if result.get("text"):
            return result.get("text")
        return result
    except Exception as e:
        return f"Error: {str(e)}"    


def transcribe_audio_openai(audio_file):
    try:
        # Open the audio file
        with open(audio_file, "rb") as audio:
            # Transcribe using Whisper v3
            transcript = client.audio.transcriptions.create(
                model="whisper-1",  # This is the API name for the latest Whisper model
                file=audio,
                response_format="text"
            )
            return transcript
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=transcribe_audio_dinum,
    inputs=[
        gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File"),
    ],
    outputs=gr.Textbox(label="Transcription"),
    title="Audio Transcription with Whisper",
    description="Upload an audio file (mp3, wav, etc.) to transcribe it using OpenAI's Whisper model."
)

# Run the app
if __name__ == "__main__":
    interface.launch()