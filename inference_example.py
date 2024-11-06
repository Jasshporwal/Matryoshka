from model import MatryoshkaEmbedder
from utils import compute_similarity
from sentence_transformers import SentenceTransformer


def test_matryoshka_model():
    """Test the base MatryoshkaEmbedder implementation"""
    # Initialize the model
    embedder = MatryoshkaEmbedder()

    # Example texts
    texts = [
        "The weather is beautiful today",
        "It's a sunny and pleasant day",
        "This is a completely different topic",
    ]

    # Compare embeddings at different dimensions
    dimensions = [768, 512, 256, 128, 64]

    for dim in dimensions:
        print(f"\nTesting MatryoshkaEmbedder with dimension: {dim}")
        embeddings = embedder.encode(texts, dimension=dim)

        # Compare similarities
        sim1 = compute_similarity(embeddings[0], embeddings[1])
        sim2 = compute_similarity(embeddings[0], embeddings[2])

        print(f"Similarity between similar sentences: {sim1:.4f}")
        print(f"Similarity between different sentences: {sim2:.4f}")


def test_trained_model():
    """Test the custom trained model"""
    print("\nTesting trained model performance:")

    # Load the trained model
    model = SentenceTransformer("trained-matryoshka-model")

    # Test texts
    texts = ["The weather is nice", "It's a beautiful day"]

    # Test different dimensions
    dimensions = [768, 512, 256, 128, 64]

    for dim in dimensions:
        print(f"\nTesting trained model with dimension: {dim}")
        # Set the desired dimension
        model.truncate_dim = dim

        # Get embeddings
        embeddings = model.encode(texts)

        # Compare similarity
        sim = compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between sentences: {sim:.4f}")


def main():
    # Test original MatryoshkaEmbedder
    print("Testing original MatryoshkaEmbedder implementation:")
    test_matryoshka_model()

    # Test trained model
    print("\nTesting trained model:")
    test_trained_model()


if __name__ == "__main__":
    main()
