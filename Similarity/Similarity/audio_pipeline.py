import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
from similarity_utils import compute_cosine_similarity


# Load your fine-tuned Whisper from HF Hub
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained("bohraanuj23/whisper-small-hindi")
model = WhisperForConditionalGeneration.from_pretrained(
    "bohraanuj23/whisper-small-hindi"
).to(DEVICE)
# WHISPER_MODEL = "YOUR_HF_USERNAME/YOUR_FINE_TUNED_WHISPER"

# processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
# model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

# Embedding model (same used in notebook)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def transcribe_and_embed(audio_file):
    # Read audio bytes
    audio_bytes = audio_file.read()

    # Load audio (assumes WAV)
    import librosa
    import io
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        audio, sr = librosa.load(f, sr=16000)

    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    embedding = embedding_model.encode(transcription)

    return transcription, embedding


def compute_similarity(new_embedding, existing_embeddings):
    similarities = []
    for ref_embedding in existing_embeddings:
        sim = compute_cosine_similarity(new_embedding, ref_embedding)
        similarities.append(sim)
    return similarities
