from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np


class MatryoshkaEmbedder:
    def __init__(
        self,
        model: Union[str, SentenceTransformer] = "tomaarsen/mpnet-base-nli-matryoshka",
        default_dim: int = 768,
    ):
        """
        Initialize the Matryoshka Embedder with a specific model and dimension.

        Args:
            model: Either a model path/name (str) or a pre-loaded SentenceTransformer model
            default_dim: Default dimension for embeddings
        """
        if isinstance(model, str):
            self.model = SentenceTransformer(model)
        elif isinstance(model, SentenceTransformer):
            self.model = model
        else:
            raise ValueError(
                "model must be either a string path/name or a SentenceTransformer instance"
            )

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
        # Validate dimension
        dimension = dimension or self.default_dim
        if dimension not in self.available_dims:
            raise ValueError(f"Dimension must be one of {self.available_dims}")

        # Generate embeddings
        embeddings = self.model.encode([texts] if isinstance(texts, str) else texts)
        
        # Truncate to the specified dimension
        truncated_embeddings = np.array([emb[:dimension] for emb in embeddings])

        # If a single string was provided, return a 1D array instead of a 2D array
        if isinstance(texts, str):
            return truncated_embeddings[0]
        return truncated_embeddings

