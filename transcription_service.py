
import mimetypes
import os
import tempfile
from typing import Dict, List

import requests
from django.conf import settings
from pydub import AudioSegment

CHUNK_DURATION_MINUTES = 10

def get_mime_type(file_path: str) -> str:
    """
    Guess the MIME type based on the file extension.
    Args:
        file_path (str): The path to the file for which to guess the MIME type.
    Returns:
        str: The guessed MIME type of the file.
    Raises:
        ValueError: If the MIME type cannot be determined or is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Unsupported file type: {file_path}")

    return mime_type


def split_audio(
    audio_path: str, chunk_length_ms: int = CHUNK_DURATION_MINUTES * 60 * 1000
) -> List[AudioSegment]:
    """
    Split an audio file into chunks of specified length.
    Args:
        audio_path (str): Path to the audio file
        chunk_length_ms (int): Length of each chunk in milliseconds (default: 10 minutes)
    Returns:
        List[AudioSegment]: List of audio chunks
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i : i + chunk_length_ms]
        chunks.append(chunk)

    return chunks


def transcribe_chunk(chunk: AudioSegment, previous_text: str = "") -> Dict:
    """
    Transcribe a single audio chunk.
    Args:
        chunk (AudioSegment): Audio chunk to transcribe
        previous_text (str): Text from previous chunks to provide context
    Returns:
        dict: Transcription result
    """
    base_url = settings.CASSANDRE_API_BASE
    api_key = settings.CASSANDRE_API_KEY
    model = settings.WHISPER_MODEL

    # Export chunk to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        chunk.export(temp_file.name, format="wav")
        mime_type = get_mime_type(temp_file.name)

        # Create prompt with context from previous chunks
        prompt = (
            "Bonjour, Voici un fichier audio d'une conference sur l'IA "
            "par quelqu'un qui travaille au rectorat de l'académie de Lyon. "
        )
        if previous_text:
            prompt += f"Contexte précédent: {previous_text[-500:]}"  # Last 500 chars of context

        with open(temp_file.name, "rb") as audio_file:
            files = {"file": (temp_file.name, audio_file, mime_type)}
            data = {
                "model": model,
                "language": "fr",
                "prompt": prompt,
                "response_format": "json",
                "temperature": 0.2,
                "timestamp_granularities[]": ["segment"],
            }

            response = requests.post(
                f"{base_url}/audio/transcriptions",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300000,
            )

            if response.status_code != 200:
                raise requests.exceptions.RequestException(
                    f"Failed to transcribe audio chunk: {response.status_code} - {response.text}"
                )

            return response.json()


def transcript(audio_file_path: str) -> Dict:
    """
    Transcribes a long audio file by splitting it into chunks and transcribing each chunk.
    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
    Returns:
        dict: The combined transcription result as a dictionary.
    Raises:
        FileNotFoundError: If the audio file does not exist.
        PermissionError: If the audio file cannot be accessed due to permissions.
        IOError: If there's an error reading the audio file.
        requests.exceptions.RequestException: If the transcription request fails.
    """
    try:
        # Validate file exists and has valid mime type
        get_mime_type(audio_file_path)

        # Split audio into chunks
        chunks = split_audio(audio_file_path)

        # Initialize results
        combined_result = {"text": "", "chunks": [], "language": "fr"}
        previous_text = ""

        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")

            # Transcribe chunk with context from previous chunks
            chunk_result = transcribe_chunk(chunk, previous_text)

            # Adjust timestamps for chunks based on chunk position
            time_offset = i * (CHUNK_DURATION_MINUTES * 60)  # 10 minutes in seconds
            for chunk_data in chunk_result["chunks"]:
                chunk_data["timestamp"][0] += time_offset
                chunk_data["timestamp"][1] += time_offset
                combined_result["chunks"].append(chunk_data)

            # Update combined text and context for next chunk
            combined_result["text"] += " " + chunk_result["text"].strip()
            previous_text = combined_result["text"]

        return combined_result

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}") from exc
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied when accessing file: {audio_file_path}"
        ) from exc
    except IOError as exc:
        raise IOError(
            f"Error reading audio file {audio_file_path}: {str(exc)}"
        ) from exc
