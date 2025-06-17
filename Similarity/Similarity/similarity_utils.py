# similarity_utils.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_cosine_similarity(embedding1, embedding2):
    # Ensure numpy arrays
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)
    return float(similarity[0][0])
