from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np


class MatryoshkaEvaluator:
    def __init__(self, model):
        """Initialize the Matryoshka evaluator."""
        self.model = model

    def evaluate_dimensions(
        self,
        eval_pairs: List[Tuple[str, str, float]],
        dimensions: List[int] = [768, 512, 256, 128, 64],
    ) -> dict:
        """
        Evaluate model performance across different dimensions.

        Args:
            eval_pairs: List of (text1, text2, similarity_score) tuples
            dimensions: List of dimensions to evaluate

        Returns:
            Dictionary mapping dimensions to performance metrics
        """
        results = {}

        for dim in dimensions:
            self.model.truncate_dim = dim
            evaluator = EmbeddingSimilarityEvaluator(*zip(*eval_pairs))
            score = evaluator(self.model)
            results[dim] = score

        return results


# Create a list of evaluation pairs (text1, text2, similarity_score)
eval_pairs = [
    ("The weather is beautiful today", "This is a completely different topic", 0.1),
    ("text3", "text4", 0.9),
]

# You need to define the 'model' variable, for example:
model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
evaluator = MatryoshkaEvaluator(model)
# Call the evaluate_dimensions method
results = evaluator.evaluate_dimensions(eval_pairs)
# Print the results
print(results)
