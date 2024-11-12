import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from model import MatryoshkaEmbedder
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class EmbeddingComparisonSystem:
    def __init__(self, matryoshka_model_paths: Dict[int, str] = None):
        # Initialize Pinecone
        self.pc = self._initialize_pinecone()
        self.opensource_embedder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Default model paths if none provided
        if matryoshka_model_paths is None:
            matryoshka_model_paths = {
                768: "trained-matryoshka-model-768",
                512: "trained-matryoshka-model-512",
                256: "trained-matryoshka-model-256",
                128: "trained-matryoshka-model-128",
            }
        # Load Matryoshka models for each dimension
        self.matryoshka_models = {}
        for dim, path in matryoshka_model_paths.items():
            if not os.path.exists(path):
                raise ValueError(
                    f"Model path {path} for dimension {dim} does not exist"
                )
            self.matryoshka_models[dim] = SentenceTransformer(path)

        # Create embedder instances for each dimension
        self.matryoshka_embedders = {
            dim: MatryoshkaEmbedder(model)
            for dim, model in self.matryoshka_models.items()
        }

        # Initialize indexes for each dimension
        self.dimensions = [768, 512, 256, 128]
        self.opensource_index = self._create_index("opensource-index", dimension=768)
        self.matryoshka_indexes = {
            dim: self._create_index(f"matryoshka-index-{dim}", dimension=dim)
            for dim in self.dimensions
        }

    def _initialize_pinecone(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key not found in environment variables")

        # Instantiate Pinecone with the API key
        pc = Pinecone(api_key=api_key)
        return pc

    def _create_index(self, index_name: str, dimension: int = 768):
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index '{index_name}' created successfully.")
        return self.pc.Index(index_name)

    def process_documents(self, documents: List[Dict[str, str]]):
        texts = [doc["text"] for doc in documents]

        # Process and store OpenSource embeddings
        opensource_embeddings = self.opensource_embedder.encode(texts)
        for idx, (doc, emb) in enumerate(zip(documents, opensource_embeddings)):
            self.opensource_index.upsert(
                vectors=[
                    {
                        "id": f"os_{idx}",
                        "values": emb.tolist(),
                        "metadata": {"text": doc["text"], "filename": doc["filename"]},
                    }
                ]
            )

        # Process and store Matryoshka embeddings for each dimension
        for dim in self.dimensions:
            matryoshka_embedder = self.matryoshka_embedders[dim]
            matryoshka_embeddings = matryoshka_embedder.encode(texts, dimension=dim)
            for idx, (doc, emb) in enumerate(zip(documents, matryoshka_embeddings)):
                self.matryoshka_indexes[dim].upsert(
                    vectors=[
                        {
                            "id": f"mat_{dim}_{idx}",
                            "values": emb.tolist(),
                            "metadata": {
                                "text": doc["text"],
                                "filename": doc["filename"],
                                "dimension": dim,
                            },
                        }
                    ]
                )

    def calculate_context_relevancy(
        self, question_embedding: np.ndarray, answer_embeddings: np.ndarray
    ) -> float:
        """
        Calculate the context relevancy between the question embedding and the answer embeddings.

        Args:
            question_embedding (np.ndarray): Embedding of the question.
            answer_embeddings (np.ndarray): Embeddings of the retrieved answers.

        Returns:
            float: The average context relevancy.
        """
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1), answer_embeddings
        ).squeeze()
        return float(np.mean(similarities))

    def query_and_compare(
        self, questions: List[str], dimensions: List[int], top_k: int = 3
    ) -> Dict[int, pd.DataFrame]:
        all_results = {}

        for dim in dimensions:
            results_data = []
            mat_relevancy_values = []
            os_relevancy_values = []

            for question in questions:
                os_query_embedding = self.opensource_embedder.encode([question])[0]
                os_results = self.opensource_index.query(
                    vector=os_query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                )

                mat_query_embedding = self.matryoshka_embedders[dim].encode(
                    [question], dimension=dim
                )[0]
                mat_results = self.matryoshka_indexes[dim].query(
                    vector=mat_query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                )

                os_answer_embeddings = np.array(
                    [
                        self.opensource_embedder.encode([match.metadata["text"]])[0]
                        for match in os_results.matches
                    ]
                )
                mat_answer_embeddings = np.array(
                    [
                        self.matryoshka_embedders[dim].encode(
                            [match.metadata["text"]], dimension=dim
                        )[0]
                        for match in mat_results.matches
                    ]
                )

                os_relevancy = self.calculate_context_relevancy(
                    os_query_embedding, os_answer_embeddings
                )
                mat_relevancy = self.calculate_context_relevancy(
                    mat_query_embedding, mat_answer_embeddings
                )

                mat_relevancy_values.append(mat_relevancy)
                os_relevancy_values.append(os_relevancy)

                results_data.append(
                    {
                        "Question": question,
                        "Matryoshka_Answer": mat_results.matches[0].metadata["text"],
                        "OpenSource_Answer": os_results.matches[0].metadata["text"],
                        "Matryoshka_Relevancy": mat_relevancy,
                        "OpenSource_Relevancy": os_relevancy,
                    }
                )

            mat_score = (
                sum(mat_relevancy_values) / len(mat_relevancy_values)
                if mat_relevancy_values
                else 0
            )
            os_score = (
                sum(os_relevancy_values) / len(os_relevancy_values)
                if os_relevancy_values
                else 0
            )

            for result in results_data:
                result["Matryoshka_Score"] = mat_score
                result["OpenSource_Score"] = os_score

            all_results[dim] = pd.DataFrame(results_data)

        return all_results

    def save_all_results_to_csv(
        self, results: Dict[int, pd.DataFrame], output_dir: str = "results"
    ):
        os.makedirs(output_dir, exist_ok=True)
        for dim, df in results.items():
            output_path = os.path.join(
                output_dir, f"embedding_comparison_results_{dim}.csv"
            )
            df.to_csv(output_path, index=False)
            print(f"Results for dimension {dim} saved to {output_path}")

    def read_documents_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r") as file:
                    text = file.read()
                    summary = text[:200]  # get the first 200 words
                    documents.append({"text": summary, "filename": filename})
        return documents
