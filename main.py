
import os
import logging
from comparison_system import EmbeddingComparisonSystem
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize the embedding comparison system
        logger.info("Initializing embedding comparison system...")

        matryoshka_model_paths = {
            768: "trained-matryoshka-model_768",
            512: "trained-matryoshka-model_512",
            256: "trained-matryoshka-model_256",
            128: "trained-matryoshka-model_128",
        }  # Specify base path for the models

        # Check if each model path exists
        for dim, path in matryoshka_model_paths.items():
            if not os.path.exists(path):
                logger.error(
                    f"Model base path '{path}' for dimension {dim} does not exist. Exiting."
                )
                return

        # Initialize the system with the valid model paths
        system = EmbeddingComparisonSystem(
            matryoshka_model_paths=matryoshka_model_paths
        )

        # Read documents from the specified folder
        folder_path = "/Users/jassh.porwal/Desktop/matryoshka/sample_data"
        logger.info(f"Reading documents from {folder_path}")
        documents = system.read_documents_from_folder(folder_path)

        if not documents:
            logger.warning(f"No documents found in the folder: {folder_path}")
            return

        # Process documents
        logger.info(f"Processing {len(documents)} documents...")
        system.process_documents(documents)

        # Define questions to query
        questions = [
            "What is contentstack?",
            "What is automation in contentstack?",
            "What is CMS in content stack?",
            # Add more questions as needed
        ]

        # Query and compare results for all dimensions
        matryoshka_dimensions = [
            768,
            512,
            256,
            128,
        ]  # Specify the model dimensions to test
        logger.info("Running queries and comparing results...")

        # Run queries for all dimensions and compare the results
        results_per_dimension = system.query_and_compare(
            questions=questions, dimensions=matryoshka_dimensions
        )

        # Save all results to separate CSV files for each dimension
        system.save_all_results_to_csv(results_per_dimension)

        # Print summary statistics
        logger.info("Comparison completed successfully!")
        logger.info("\nSummary Statistics:")

        # Example statistics (you can adapt based on your result structure)
        for dim in matryoshka_dimensions:
            logger.info(f"\nFor Dimension {dim}:")
            dimension_results = results_per_dimension.get(dim, {})
            if not dimension_results.empty:
                logger.info(
                    f"Average Matryoshka Score: {dimension_results['Matryoshka_Score'].mean():.3f}"
                )
                logger.info(
                    f"Average OpenSource Score: {dimension_results['OpenSource_Score'].mean():.3f}"
                )
                logger.info(
                    f"Average Matryoshka Relevancy: {dimension_results['Matryoshka_Relevancy'].mean():.3f}"
                )
                logger.info(
                    f"Average OpenSource Relevancy: {dimension_results['OpenSource_Relevancy'].mean():.3f}"
                )
            else:
                logger.warning(f"No results found for dimension {dim}.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()