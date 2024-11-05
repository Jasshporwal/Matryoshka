# trainer.py
from sentence_transformers import SentenceTransformer, InputExample, models
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from torch.utils.data import DataLoader
from typing import List
import logging


class MatryoshkaTrainer:
    def __init__(
        self,
        base_model: str = "microsoft/mpnet-base",
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
        """Train the Matryoshka model."""
        # Convert training data to InputExample format
        train_examples = self.prepare_training_data(train_data)
        train_dataloader = DataLoader(
            train_examples, batch_size=batch_size, shuffle=True
        )

        # Create base loss - using MultipleNegativesRankingLoss instead of CoSENTLoss
        base_loss = MultipleNegativesRankingLoss(self.model)

        # Wrap with MatryoshkaLoss
        loss = MatryoshkaLoss(
            model=self.model,
            loss=base_loss,
            matryoshka_dims=self.matryoshka_dims,
        )

        # Train the model
        logging.info(f"Starting training with {len(train_examples)} examples")
        self.model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
        )
