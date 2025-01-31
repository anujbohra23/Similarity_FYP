import streamlit as st
from pathlib import Path
import torch
from pyannote.audio import Pipeline
import whisper
import json
from pydub import AudioSegment
import re
import subprocess

# Title of the Streamlit app
st.title("Audio Diarization and Transcription")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "mp4"])

# Hugging Face token
access_token = st.text_input("Enter your Hugging Face token", type="password")


# Function to convert audio file to WAV format using ffmpeg
def convert_to_wav(input_file, output_file):
    command = f'ffmpeg -i "{input_file}" -ar 16000 -ac 1 "{output_file}"'
    subprocess.run(command, shell=True, check=True)


# If a file is uploaded and token is provided
if uploaded_file is not None and access_token:
    try:
        # Save the uploaded file
        input_path = Path("input") / uploaded_file.name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert the uploaded file to WAV format if it's not already in WAV
        wav_path = Path("input_prep.wav")
        if input_path.suffix != ".wav":
            convert_to_wav(input_path, wav_path)
        else:
            # Add silence to the beginning
            spacer = AudioSegment.silent(duration=2000)
            audio = AudioSegment.from_wav(input_path)
            audio = spacer.append(audio, crossfade=0)
            audio.export(wav_path, format="wav")

        # Diarization
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=access_token
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        DEMO_FILE = {"uri": "blabla", "audio": str(wav_path)}
        dz = pipeline(DEMO_FILE)

        with open("diarization.txt", "w") as text_file:
            text_file.write(str(dz))

        # Parse diarization result
        def millisec(timeStr):
            spl = timeStr.split(":")
            s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
            return s

        dzs = open("diarization.txt").read().splitlines()

        groups = []
        g = []
        lastend = 0

        for d in dzs:
            if g and (g[0].split()[-1] != d.split()[-1]):  # different speaker
                groups.append(g)
                g = []

            g.append(d)

            end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
            end = millisec(end)
            if lastend > end:  # segment engulfed by a previous segment
                groups.append(g)
                g = []
            else:
                lastend = end

        if g:
            groups.append(g)

        audio = AudioSegment.from_wav(str(wav_path))
        gidx = -1
        for g in groups:
            start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
            end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
            start = millisec(start)
            end = millisec(end)
            gidx += 1
            audio[start:end].export(str(gidx) + ".wav", format="wav")

        # Transcription with language detection
        model = whisper.load_model("large", device=device)

        transcriptions = []
        for i in range(len(groups)):
            audiof = str(i) + ".wav"
            result = model.transcribe(audio=audiof, word_timestamps=True)
            transcriptions.append(result)
            with open(str(i) + ".json", "w") as outfile:
                json.dump(result, outfile, indent=4)

        # Display results
        st.write("Diarization and Transcription Results")
        for idx, transcription in enumerate(transcriptions):
            st.write(f"Speaker {idx}: {transcription['text']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
