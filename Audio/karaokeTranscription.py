import os
import streamlit as st
import whisper
import tempfile
import subprocess
import json

# Streamlit app
st.title("Karaoke-style Transcript for Dubbing Artists")
st.write(
    "Upload an audio file to generate the karaoke-style transcript video for dubbing."
)

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with st.spinner("Transcribing audio..."):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_path = temp_audio_file.name

        # Load Whisper model and transcribe the audio
        model = whisper.load_model("medium")
        result = model.transcribe(temp_audio_path)

        st.success("Transcription complete!")
        st.write("Transcript:", result["text"])

        # Save the transcription result to a JSON file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json"
        ) as temp_transcript_file:
            temp_transcript_file.write(json.dumps(result).encode())
            temp_transcript_path = temp_transcript_file.name

        # Generate the karaoke video using the Pygame script
        pygame_command = (
            f"python generate_karaoke.py {temp_audio_path} {temp_transcript_path}"
        )
        subprocess.run(pygame_command, shell=True)

        # Inform the user
        st.success(
            "Karaoke video generated successfully! The video will be displayed in a separate Pygame window."
        )
