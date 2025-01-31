import streamlit as st
import os
import subprocess

os.system("pip install insanely-fast-whisper")


def transcribe_audio(file_path):

    result = subprocess.run(
        f'insanely-fast-whisper --file-name "{file_path}" --device cpu', shell=True
    )
    if result.returncode != 0:
        st.error("Error during transcription. Please check the logs.")


# Streamlit application
st.title("Audio Transcription with Insanely Fast Whisper")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save uploaded file to disk
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"File uploaded: {uploaded_file.name}")

    # Transcribe audio
    st.write("Transcribing audio...")
    transcribe_audio(file_path)
    st.write("Transcription complete.")

    # List and provide download links for all generated files
    result_files = [
        file for file in os.listdir(".") if file.endswith((".srt", ".txt", ".json"))
    ]
    if result_files:
        for result_file in result_files:
            with open(result_file, "rb") as f:
                st.download_button(
                    label=f"Download {result_file}",
                    data=f,
                    file_name=result_file,
                    mime="text/plain",
                )
    else:
        st.write("No transcription result files found.")
else:
    st.write("Please upload an audio file.")
