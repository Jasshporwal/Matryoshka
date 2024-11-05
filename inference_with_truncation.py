from sentence_transformers import SentenceTransformer
import numpy as np

# Load the trained model
model_path = "trained-matryoshka-model"
model = SentenceTransformer(model_path)


def get_truncated_embeddings(model, texts, dim):
    return [model.encode(text)[:dim] for text in texts]


# Example texts
texts = [
    "The weather is beautiful today",
    "It's a sunny and pleasant day",
    "This is a completely different topic",
]


# Set desired dimension to truncate embeddings
desired_dimension = 64

# Get truncated embeddings
truncated_embeddings = get_truncated_embeddings(model, texts, desired_dimension)

# Display truncated embeddings
for i, emb in enumerate(truncated_embeddings):
    print(f"Embedding for text '{texts[i]}' (dimension {desired_dimension}):\n{emb}\n")
