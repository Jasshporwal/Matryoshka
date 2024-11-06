
from trainer import MatryoshkaTrainer
from utils import create_evaluation_pairs
import logging
import os
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create output directory if it doesn't exist
    output_dir = "trained-matryoshka-model"
    os.makedirs(output_dir, exist_ok=True)

    # Sample training data with more examples and clear semantic relationships
    texts = [
        # Weather-related pairs
        "The weather is beautiful today",
        "It's a sunny and pleasant day",
        "The sky is clear and bright",
        "The temperature is perfect outside",
        # Animal-related pairs
        "The cat is sleeping on the couch",
        "A feline is resting on the sofa",
        "The dog is playing in the yard",
        "The puppy is running around outside",
        # Programming-related pairs
        "Python is a great programming language",
        "Programming in Python is wonderful",
        "Java is used for enterprise development",
        "Software development using Java is common",
        # Additional varied examples
        "The book was very interesting",
        "This novel is quite engaging",
        "The movie was fantastic",
        "The film was excellent",
        "I love cooking Italian food",
        "Making pasta dishes is my passion",
        "Playing guitar is fun",
        "Learning music is enjoyable",
    ]

    logger.info("Creating training pairs...")
    train_data = create_evaluation_pairs(texts)

    logger.info("Initializing trainer...")
    trainer = MatryoshkaTrainer(
        matryoshka_dims=[768, 512, 256, 128, 64], weights=[1.0, 0.8, 0.6, 0.4, 0.2]
    )

    logger.info("Starting training...")
    trainer.train(
        train_data=train_data,
        batch_size=16,
        epochs=5,
        output_path=output_dir,
        warmup_steps=100,
    )
    logger.info("Training completed!")


if __name__ == "__main__":
    # Set environment variable to handle tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
