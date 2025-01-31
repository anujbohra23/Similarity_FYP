import streamlit as st
import numpy as np
import torch
from transformers import (
    pipeline,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)
import librosa
import io
import srt
from datetime import timedelta
import json

# Load the Whisper model from OpenAI
model_name_or_path = "openai/whisper-medium"
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

# Initialize the ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    processor=processor,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Streamlit app title
st.title("Voice Recognition with Whisper and Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Read the file as bytes
    file_bytes = uploaded_file.read()

    # Use librosa to process the audio file
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    audio_array = np.array(y, dtype=np.float32)

    # Prepare the input features
    input_features = processor(
        audio_array, sampling_rate=sr, return_tensors="pt"
    ).input_features

    # Perform speech recognition using the pipeline
    with torch.no_grad():
        generated_ids = model.generate(input_features)
        transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

    # Display the transcription
    st.write("Transcription:")
    st.write(transcription)

    # Download options
    st.download_button("Download TXT", transcription, file_name="transcription.txt")

    # Generate SRT format
    subtitles = [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=len(y) / sr),
            content=transcription,
        )
    ]
    srt_output = srt.compose(subtitles)
    st.download_button("Download SRT", srt_output, file_name="transcription.srt")

    # Generate VTT format
    vtt_output = srt_output.replace(" --> ", " --> ").replace(",", ".")
    st.download_button("Download VTT", vtt_output, file_name="transcription.vtt")

    # Generate TSV format
    tsv_output = f"start\tend\ttext\n0\t{len(y) / sr}\t{transcription}"
    st.download_button("Download TSV", tsv_output, file_name="transcription.tsv")

    # Generate JSON format
    json_output = json.dumps({"transcription": transcription}, indent=4)
    st.download_button("Download JSON", json_output, file_name="transcription.json")
