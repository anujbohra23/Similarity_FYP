# Install faster-whisper
# !pip install faster-whisper
from faster_whisper import WhisperModel

# Specify model size
model_size = "large-v3"

# Load model
# Adjust the device according to your setup (use "cpu" if you don't have a compatible GPU)
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# File path to the audio file
file_path = r"C:\Users\Anuj Bohra\Desktop\PostStudio\audioFiles\TOEFL Listening Practice Test.mp3"

# Transcribe the audio file
segments, info = model.transcribe(file_path, beam_size=5)

# Print detected language and probability
print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

# Print each segment with timestamps
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
