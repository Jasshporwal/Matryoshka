from model import MatryoshkaEmbedder
from evaluator import MatryoshkaEvaluator
from utils import create_evaluation_pairs
import matplotlib.pyplot as plt


def main():
    # Initialize the model
    embedder = MatryoshkaEmbedder()
    evaluator = MatryoshkaEvaluator(embedder.model)

    # Sample evaluation data
    texts = [
        "The weather is beautiful today",
        "It's a sunny and pleasant day",
        "The cat is sleeping on the couch",
        "A feline is resting on the sofa",
        "Python is a great programming language",
        "Programming in Python is wonderful",
    ]

    eval_pairs = create_evaluation_pairs(texts)

    # Evaluate different dimensions
    results = evaluator.evaluate_dimensions(eval_pairs)

    # Extract the values from the results dictionary
    dimensions = list(results.keys())
    scores = [
        list(value.values())[0] for value in results.values()
    ]  # Extract the first value from each dictionary

    # Plot the results
    plt.plot(dimensions, scores, marker="o")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Performance Score")
    plt.title("Matryoshka Embedding Performance vs Dimension")
    plt.grid(True)
    plt.savefig("dimension_comparison.png")
    plt.close()

    # Print the results
    for dim, value in results.items():
        print(f"Dimension {dim}: Score = {list(value.values())[0]:.4f}")


if __name__ == "__main__":
    main()
