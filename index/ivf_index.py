import numpy as np
from typing import List, Dict
from sklearn.cluster import MiniBatchKMeans
from abstract_index import AbstractIndex

class IVFIndex(AbstractIndex):

    def __init__(
        self,
        table_name: str,
        dimension: int,
        ids: list,
        embeddings: np.array,
        n_clusters: int,
        metadatas: List[Dict] = None,
    ):
        super().__init__(table_name, "IVF", dimension, ids, embeddings)

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (
            embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings)
        )

        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, self.dimension)

        if self.embeddings.ndim > 2:
            raise ValueError("Embeddings should be a 2D array.")

        if self.embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings dimension does not match index dimension i.e {self.dimension}"
            )

        self.ids = ids
        self.metadatas = metadatas if metadatas is not None else []
        self.vector_count = self.embeddings.shape[0]
        self.n_clusters = n_clusters

        # Create an Inverted Index
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(self.embeddings)
        self.clusters_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_

        # Create an inverted index mapping cluster labels to sentence indices. 3 clusters and 6 sentences -> {0: [1, 3 , 5], 1: [2, 4], 2: [6]}
        self.inverted_index = {}
        for item, label in enumerate(self.clusters_labels):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(item)

    def add(self, id: int, vector: np.array, metadata: Dict = None):
        
        if id in self.ids:
            raise ValueError(f"ID {id} already exists in the index.")

        if vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if vector.size != self.dimension:
            raise ValueError(
                f"Vector dimension ({vector.size}) does not match index dimension ({self.dimension})."
            )

        # TODO: Add a check for unique ID

        self.ids.append(id)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.append(metadata) if metadata else None
        self._update_vector_count()

    def search(self, query_vector: np.array, top_k: int, filter_param: Dict = None):

        if top_k <= 0:
            raise ValueError("Top K must be greater than 0")

        if query_vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if query_vector.size != self.dimension:
            raise ValueError(
                f"Vector dimension ({query_vector.size}) does not match index dimension ({self.dimension})."
            )

        # TODO: Implement  Metadata Filtering to narrow search space
        # - Post Filtering
        # - Pre Filtering

        # Calculate distances from query_embedding to each cluster center to find closes cluster
        distances = np.linalg.norm(query_vector - self.cluster_centers, axis=1)
        closest_cluster_index = np.argmin(distances)

        # Retrieve the top_k Result
        score = {}  # -> {1: 0.8, 2: 0.1, 3: 0.5, 4: 0.2, 5: 0.9}
        for doc_id in self.inverted_index[closest_cluster_index]:
            cosine_similarity = np.dot(query_vector, self.embeddings[doc_id]) / (
                np.linalg.norm(query_vector)
                * np.linalg.norm(self.embeddings[doc_id])
                + 1e-10
            )
            score[doc_id] = cosine_similarity
        
        # Sort the indices by similarity scores in descending order
        sorted_top_k = sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_k] # -> [(5, 0.9), (1, 0.8), (3, 0.5)] if top_k = 3
        result = [
            {
                "id": self.ids[sorted_top_k[i][0]],
                "embedding": self.embeddings[sorted_top_k[i][0]],
                "metadata": (
                    self.metadatas[sorted_top_k[i][0]] if self.metadatas else {}
                ),
                "score": sorted_top_k[i][1],
            }
            for i in range(len(sorted_top_k))
        ]
        return result
