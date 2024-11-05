# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from typing import List, Tuple
# import numpy as np

# class MatryoshkaEvaluator:
#     def __init__(self, model):
#         """Initialize the Matryoshka evaluator."""
#         self.model = model

#     def evaluate_dimensions(
#         self,
#         eval_pairs: List[Tuple[str, str, float]],
#         dimensions: List[int] = [768, 512, 256, 128, 64]
#     ) -> dict:
#         """
#         Evaluate model performance across different dimensions.

#         Args:
#             eval_pairs: List of (text1, text2, similarity_score) tuples
#             dimensions: List of dimensions to evaluate

#         Returns:
#             Dictionary mapping dimensions to performance metrics
#         """
#         results = {}

#         for dim in dimensions:
#             self.model.truncate_dim = dim
#             evaluator = EmbeddingSimilarityEvaluator(*zip(*eval_pairs))
#             score = evaluator(self.model)
#             results[dim] = score

#         return results

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
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
    ("The weather is beautiful toda", "This is a completely different topic", 0.8),
    ("text3", "text4", 0.9),
]

# Create an instance of the MatryoshkaEvaluator class
# You need to define the 'model' variable, for example:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

evaluator = MatryoshkaEvaluator(model)

# Call the evaluate_dimensions method
results = evaluator.evaluate_dimensions(eval_pairs)

# Print the results
print(results)
