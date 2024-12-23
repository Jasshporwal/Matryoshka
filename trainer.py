from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from typing import List
import logging


class MatryoshkaTrainer:
    def __init__(
        self,
        base_model: str = "tomaarsen/mpnet-base-nli-matryoshka",
        matryoshka_dims: List[int] = [768, 512, 256, 128, 64],
        weights: List[float] = None,
    ):
        """Initialize the Matryoshka trainer."""
        # Create the base model
        word_embedding_model = models.Transformer(base_model)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.matryoshka_dims = matryoshka_dims
        self.weights = weights or [1.0] * len(matryoshka_dims)

    def prepare_training_data(self, texts_pairs: List[tuple]) -> List[InputExample]:
        """Convert text pairs to InputExample format."""
        examples = []
        for text1, text2, score in texts_pairs:
            examples.append(InputExample(texts=[text1, text2], label=float(score)))
        return examples

    def train(
        self,
        train_data: List[tuple],
        batch_size: int = 16,
        epochs: int = 5,
        output_path: str = "matryoshka-model",
        warmup_steps: int = 100,
    ):
        """Train the Matryoshka model for each dimension."""
        # Convert training data to InputExample format
        train_examples = self.prepare_training_data(train_data)
        train_dataloader = DataLoader(
            train_examples, batch_size=batch_size, shuffle=True
        )

        # Loop through each dimension in matryoshka_dims
        for dim in self.matryoshka_dims:
            logging.info(f"Training with dimension: {dim}")

            # Create a loss function with the current dimension
            base_loss = MultipleNegativesRankingLoss(self.model)
            # Optionally, you could use other loss functions here
            loss = MatryoshkaLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=[dim],  # Only use the current dimension
            )

            # Train the model for the current dimension
            logging.info(
                f"Starting training with {len(train_examples)} examples for dimension {dim}"
            )
            self.model.fit(
                train_objectives=[(train_dataloader, loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=f"{output_path}_{dim}",
                show_progress_bar=True,
            )
            logging.info(f"Training for dimension {dim} completed!")
