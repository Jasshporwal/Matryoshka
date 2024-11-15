
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from model import MatryoshkaEmbedder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

class EmbeddingComparisonSystem:
    def __init__(self, matryoshka_model_paths: Dict[int, str] = None):
        # Initialize Qdrant
        self.client = self._initialize_qdrant()
        self.opensource_embedder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Default model paths if none provided
        if matryoshka_model_paths is None:
            matryoshka_model_paths = {
                768: "trained-matryoshka-model_768",
                512: "trained-matryoshka-model_512",
                256: "trained-matryoshka-model_256",
                128: "trained-matryoshka-model_128",
                64: "trained-matryoshka-model_64",
            }
        
        # Load Matryoshka models for each dimension
        self.matryoshka_models = {}
        for dim, path in matryoshka_model_paths.items():
            if not os.path.exists(path):
                raise ValueError(f"Model path {path} for dimension {dim} does not exist")
            self.matryoshka_models[dim] = SentenceTransformer(path)

        # Create embedder instances for each dimension
        self.matryoshka_embedders = {
            dim: MatryoshkaEmbedder(model)
            for dim, model in self.matryoshka_models.items()
        }

        # Initialize collections for each dimension
        self.dimensions = [768, 512,256, 128, 64]
        self.collection_names = {
            'opensource': 'opensource_collection',
            **{dim: f'matryoshka_collection_{dim}' for dim in self.dimensions}
        }
        self._create_collections()

    def _initialize_qdrant(self):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        # Increase the timeout settings
        timeout =60.0  # 30s general timeout, 60s write timeout
        
        # Initialize the Qdrant client with the increased timeout
        return QdrantClient(url=url, api_key=api_key, timeout=timeout)


    def _create_collections(self):
        # Create collection for opensource embeddings
        self.client.recreate_collection(
            collection_name=self.collection_names['opensource'],
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )

        # Create collections for each Matryoshka dimension
        for dim in self.dimensions:
            self.client.recreate_collection(
                collection_name=self.collection_names[dim],
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )

    def create_clusters(self, embeddings: np.ndarray, n_clusters: int = 6):
        """Create clusters from embeddings using KMeans"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels, kmeans.cluster_centers_

    def process_documents(self, documents: List[Dict[str, str]]):
        texts = [doc["text"] for doc in documents]

        # Process and store OpenSource embeddings
        opensource_embeddings = self.opensource_embedder.encode(texts)
        
        # Create clusters for opensource embeddings
        cluster_labels, _ = self.create_clusters(opensource_embeddings)
        
        # Store opensource embeddings with cluster information
        points = [
            models.PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "cluster": int(cluster_label)
                }
            )
            for idx, (doc, emb, cluster_label) in enumerate(zip(documents, opensource_embeddings, cluster_labels))
        ]
        self.client.upsert(
            collection_name=self.collection_names['opensource'],
            points=points
        )

        # Process and store Matryoshka embeddings for each dimension
        for dim in self.dimensions:
            matryoshka_embedder = self.matryoshka_embedders[dim]
            matryoshka_embeddings = matryoshka_embedder.encode(texts, dimension=dim)
            
            # Create clusters for this dimension
            cluster_labels, _ = self.create_clusters(matryoshka_embeddings)
            
            points = [
                models.PointStruct(
                    id=idx,
                    vector=emb.tolist(),
                    payload={
                        "text": doc["text"],
                        "filename": doc["filename"],
                        "dimension": dim,
                        "cluster": int(cluster_label)
                    }
                )
                for idx, (doc, emb, cluster_label) in enumerate(zip(documents, matryoshka_embeddings, cluster_labels))
            ]
            self.client.upsert(
                collection_name=self.collection_names[dim],
                points=points
            )

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
                os_results = self.client.search(
                    collection_name=self.collection_names['opensource'],
                    query_vector=os_query_embedding.tolist(),
                    limit=top_k
                )

                mat_query_embedding = self.matryoshka_embedders[dim].encode(
                    [question], dimension=dim
                )[0]
                mat_results = self.client.search(
                    collection_name=self.collection_names[dim],
                    query_vector=mat_query_embedding.tolist(),
                    limit=top_k
                )

                os_answer_embeddings = np.array([
                    self.opensource_embedder.encode([match.payload["text"]])[0]
                    for match in os_results
                ])
                mat_answer_embeddings = np.array([
                    self.matryoshka_embedders[dim].encode(
                        [match.payload["text"]], dimension=dim
                    )[0]
                    for match in mat_results
                ])

                os_relevancy = self.calculate_context_relevancy(
                    os_query_embedding, os_answer_embeddings
                )
                mat_relevancy = self.calculate_context_relevancy(
                    mat_query_embedding, mat_answer_embeddings
                )

                mat_relevancy_values.append(mat_relevancy)
                os_relevancy_values.append(os_relevancy)

                results_data.append({
                    "Question": question,
                    "Matryoshka_Answer": mat_results[0].payload["text"],
                    "OpenSource_Answer": os_results[0].payload["text"],
                    "Matryoshka_Relevancy": mat_relevancy,
                    "OpenSource_Relevancy": os_relevancy,
                    "Matryoshka_Cluster": mat_results[0].payload["cluster"],
                    "OpenSource_Cluster": os_results[0].payload["cluster"]
                })

            mat_score = sum(mat_relevancy_values) / len(mat_relevancy_values) if mat_relevancy_values else 0
            os_score = sum(os_relevancy_values) / len(os_relevancy_values) if os_relevancy_values else 0

            for result in results_data:
                result["Matryoshka_Score"] = mat_score
                result["OpenSource_Score"] = os_score

            all_results[dim] = pd.DataFrame(results_data)

        return all_results

    def calculate_context_relevancy(
        self, question_embedding: np.ndarray, answer_embeddings: np.ndarray
    ) -> float:
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1), answer_embeddings
        ).squeeze()
        return float(np.mean(similarities))

    def save_all_results_to_csv(
        self, results: Dict[int, pd.DataFrame], output_dir: str = "results"
    ):
        os.makedirs(output_dir, exist_ok=True)
        for dim, df in results.items():
            output_path = os.path.join(output_dir, f"embedding_comparison_results_{dim}.csv")
            df.to_csv(output_path, index=False)
            print(f"Results for dimension {dim} saved to {output_path}")

    def read_documents_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r") as file:
                    text = file.read()
                    documents.append({"text": text, "filename": filename})
        return documents
