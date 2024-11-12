
# Matryoshka Embedding 

This project implements a custom multi-resolution embedding model, called the Matryoshka model, which allows text embeddings to be generated at different dimensions. This flexibility is useful for tasks such as similarity measurement, clustering, and search, where embeddings of various granularities are beneficial.


## Setup

1. Clone the repository:
   ```
   git clone https://github.com/Jasshporwal/Matryoshka.git
   cd Matryoshka
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Training the Model

To train the Matryoshka model, use train_model.py. This script loads sample training data, prepares it, and trains the model to generate multi-dimensional embeddings.

1.	Prepare Training Data: Define text pairs with similarity scores (0.0 to 1.0), indicating the degree of relatedness. You can edit train_model.py to add your data or load from a file.

2.	Run Training:
    ```
    python train_model.py
    ```
3.	Training Parameters:
   
    - batch_size: Number of examples per training batch.
    - epochs: Number of training epochs.
    - matryoshka_dims: List of dimensions to train the model on (e.g., [768, 512, 256, 128, 64]).
    - weights: Weighting for each dimension during training.


The model will be saved at the specified output directory after training.

## Evaluating the Model

You can use evaluator.py to assess model performance on text similarity tasks by generating pairs of text embeddings and comparing them with cosine similarity.

Steps:

1.	Modify evaluator.py to include or load evaluation data if needed.
2.	Run the evaluator script:
    ```
    python evaluator.py
    ```
3.	Evaluation metrics such as cosine similarity will be calculated for the provided pairs.

## Generating Truncated Embeddings

Use inference_with_truncation.py to load the model, generate embeddings, and truncate them to a desired dimension.

1.	Load the Model: The script loads the trained model from the output directory (trained-matryoshka-model by default).
2.	Generate and Truncate Embeddings: Specify desired_dimension to truncate the embeddings to that size (e.g., 256).
3.	Run the Script:
    ```
    python inference_with_truncation.py
    ```

# Example Output 
    
Embedding for text 'The weather is nice' (dimension 256):
[array of truncated embedding values]

Embedding for text 'It's a beautiful day' (dimension 256):
[array of truncated embedding values]

## Inference

To generate embeddings for new texts without truncation, you can use inference_example.py. This script loads the trained model and creates embeddings for each input text at full dimensionality.

Steps:

1.	Add or modify text data in inference_example.py.
2.	Run the script:
    ```
    python inference_example.py
    ```
3.	The embeddings will be printed out for each input text.

## Embedding Comparison System
 
 The comparison_system.py handles the core logic for comparing embeddings between the Matryoshka Embedding model and the Opensource model

 Steps:
1. Model Loading: Loads the Matryoshka and OpenSource models and initializes their Pinecone indexes.
2. Document Processing: Encodes documents and stores their embeddings in Pinecone.
3. Query Comparison: Compares the query results from the Matryoshka and OpenSource models, calculating relevancy scores.
4. Results Saving: Saves the results for each dimension in CSV files.

## Main Script

The main.py file is the entry point for running the embedding comparison system.

Steps:
1. Model Initialization: Loads Matryoshka models for various dimensions and checks model paths.
2. Document Processing: Reads and processes documents from a specified folder.
3. Query Execution: Executes predefined queries and compares results from both Matryoshka and OpenSource models.
4. Results Output: Saves comparison results per dimension in CSV files and prints summary statistics.
5. Run the Script:
   ```
   python main.py
   ```

## Contributing

Contributions are welcome! Please submit a pull request with your changes.
