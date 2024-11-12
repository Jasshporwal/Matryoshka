
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector."""
    return embedding / np.linalg.norm(embedding)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]


def create_evaluation_pairs(texts: List[str]) -> List[Tuple[str, str, float]]:
    """Create evaluation pairs from a list of texts with meaningful similarity scores."""
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Calculate semantic similarity
            same_topic = i // 2 == j // 2
            partial_match = any(
                word in texts[j].lower() for word in texts[i].lower().split()
            )

            if same_topic:
                similarity = 0.9  # High similarity for same topic
            elif partial_match:
                similarity = 0.5  # Medium similarity for partial matches
            else:
                similarity = 0.1  # Low similarity for different topics

            pairs.append((texts[i], texts[j], similarity))
    return pairs
