import whisper
import gradio as gr

model = whisper.load_model("large")


def transcribe(audio):
    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


gr.Interface(
    title="OpenAI Whisper ASR Gradio Web UI",
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Textbox(),
    live=True,
).launch()
