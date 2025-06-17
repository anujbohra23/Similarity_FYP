import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PIL import Image
from similarity_utils import compute_cosine_similarity

# Load BLIP captioning model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Embedding model (same as audio pipeline)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def caption_and_embed(image_file):
    # Open image
    image = Image.open(image_file).convert("RGB")

    # Generate caption
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Embed caption
    embedding = embedding_model.encode(caption)

    return caption, embedding


def compute_similarity(new_embedding, existing_embeddings):
    similarities = []
    for ref_embedding in existing_embeddings:
        sim = compute_cosine_similarity(new_embedding, ref_embedding)
        similarities.append(sim)
    return similarities
