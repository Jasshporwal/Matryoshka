from sentence_transformers import SentenceTransformer
from typing import  Union
import numpy as np


class MatryoshkaEmbedder:
    def __init__(
        self,
        model_name: str = "tomaarsen/mpnet-base-nli-matryoshka",
        default_dim: int = 768,
    ):
        """Initialize the Matryoshka Embedder with a specific model and dimension."""
        self.model = SentenceTransformer(model_name)
        self.default_dim = default_dim
        self.available_dims = [768, 512, 256, 128, 64]

    def encode(self, texts: Union[str, list[str]], dimension: int = None) -> np.ndarray:
        """
        Encode texts using the Matryoshka model with specified dimension.

        Args:
            texts: Single text or list of texts to encode
            dimension: Target embedding dimension (must be one of available_dims)

        Returns:
            numpy array of embeddings
        """
        if dimension and dimension not in self.available_dims:
            raise ValueError(f"Dimension must be one of {self.available_dims}")

        dimension = dimension or self.default_dim
        self.model.truncate_dim = dimension

        return self.model.encode(texts)
