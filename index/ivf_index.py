import numpy as np
from typing import List, Dict, Union
from sklearn.cluster import MiniBatchKMeans, KMeans
from .abstract_index import AbstractIndex

# TODO: Trigger to Rebuild Index after deletion threshold
class IVFIndex(AbstractIndex):
    """
    An efficient embedding-based search index using Inverted File (IVF) structure.

    Suitable for medium to large-scale datasets, it clusters embeddings to speed up search operations. Supports adding embeddings, updating the index, and performing similarity searches with optional metadata filtering.

    Attributes:
        table_name (str): Name of the associated table.
        dimension (int): Dimensionality of embeddings.
        ids (list): Unique IDs for each embedding.
        embeddings (np.ndarray): Stored embeddings matrix.
        n_clusters (int, optional): Number of clusters for partitioning.
        metadatas (List[Dict], optional): Associated metadata for each embedding.
    """
    def __init__(self, table_name: str, dimension: int, ids: list, embeddings: np.array, n_clusters: int = None, metadatas: List[Dict] = None):
        super().__init__(table_name, "IVF", dimension, ids, embeddings)

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings))

        self.dimension = dimension

        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, self.dimension)

        if self.embeddings.ndim > 2:
            raise ValueError("Embeddings should be a 2D array.")

        if self.embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings dimension does not match index dimension i.e {self.dimension}")

        self.table_name = table_name
        self.ids = ids
        self.metadatas = metadatas if metadatas is not None else []
        self.vector_count = self.embeddings.shape[0]
        self.n_clusters = n_clusters if n_clusters else int(np.sqrt(self.vector_count))

        # Selecting the appropriate KMeans algorithm based on the size of the dataset.
        if self.vector_count <= 10000:
            self.kmeans_method = KMeans(n_clusters=self.n_clusters, random_state=0)
        else:
            self.kmeans_method = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0)

        self._build_index()

    def _build_index(self):

        self.kmeans_method.fit(self.embeddings)
        self.clusters_labels = self.kmeans_method.labels_
        self.cluster_centers = self.kmeans_method.cluster_centers_

        # Create an inverted index mapping cluster labels to sentence indices. 
        # 3 clusters and 6 sentences -> {0: [1, 3 , 5], 1: [2, 4], 2: [6]}
        self.inverted_index = {}
        for item, label in enumerate(self.clusters_labels):
            if label not in self.inverted_index:
                self.inverted_index[label] = []
            self.inverted_index[label].append(item)

    def add(self, idx: List[int], vector: Union[List, np.array], metadata: Dict = None):
        
        if any(i in self.ids for i in idx):
            raise ValueError("One or more IDs already exist in the index.")
        
        # Handle input validation
        if not isinstance(vector, (np.ndarray, list)):
            raise ValueError("Vector should be a NumPy array or a list.")
        
        vector = vector if isinstance(vector, np.ndarray) else np.array(vector)
        
        if len(idx) != vector.shape[0]:
            raise ValueError("Each ID must correspond to a single vector.")

        if any(v.shape[0] != self.dimension for v in vector):
            raise ValueError("One or more vectors' dimensions do not match the index dimension.")

        self.ids.extend(idx)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.extend(metadata) if metadata else None

        for _ in range(len(idx)):
            self._update_vector_count()

        self._build_index()

    def search(self, query_vector: np.array, top_k: int, filter_param: Dict = None):

        if top_k <= 0:
            raise ValueError("Top K must be greater than 0")
        
        # Handle input validation
        if not isinstance(query_vector, (np.ndarray, list)):
            raise ValueError("Vector should be a NumPy array or a list.")

        query_vector = query_vector if isinstance(query_vector, np.ndarray) else np.array(query_vector)

        if query_vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if query_vector.size != self.dimension:
            raise ValueError(f"Vector dimension ({query_vector.size}) does not match index dimension ({self.dimension}).")

        # TODO: Implement  Metadata Filtering to narrow search space
        # - Post Filtering
        # - Pre Filtering

        # Calculate distances from query_embedding to each cluster center to find closes cluster
        distances = np.linalg.norm(query_vector - self.cluster_centers, axis=1)

        # Find the closest cluster(s) to the query vector
        closest_cluster_indices = np.argsort(distances)
        self.indices_to_consider = []

        # if top_k is more than the number of documents in the closest cluster, select docs from the next closest cluster
        for cluster_index in closest_cluster_indices:
            self.indices_to_consider.extend(self.inverted_index[cluster_index])
            if len(self.indices_to_consider) >= top_k:
                break
        
        # Retrieve the top_k Result
        score = {}  # -> {1: 0.8, 2: 0.1, 3: 0.5, 4: 0.2, 5: 0.9}
        for doc_id in self.indices_to_consider[:top_k]:

            cosine_similarity = np.dot(query_vector, self.embeddings[doc_id]) / (
                np.linalg.norm(query_vector) * np.linalg.norm(self.embeddings[doc_id])+ 1e-10)
            score[doc_id] = cosine_similarity

        # Sort the indices by similarity scores in descending order
        sorted_top_k = sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_k]  # -> [(5, 0.9), (1, 0.8), (3, 0.5)] if top_k = 3

        result = [
            {
                "id": self.ids[sorted_top_k[i][0]],
                "embedding": self.embeddings[sorted_top_k[i][0]],
                "metadata": (self.metadatas[sorted_top_k[i][0]] if self.metadatas else {}),
                "score": sorted_top_k[i][1],
            }
            for i in range(len(sorted_top_k))
        ]
        return result


    def delete_vector(self, idx: List[int]):

        if any(i not in self.ids for i in idx):
            raise ValueError("One or more IDs not in the index.")
        
                # Handle input validation
        if not isinstance(idx, List):
            raise ValueError("ID should be a List.")
        
        # Impleement Lazy Deletion
        self.ids = [_id if _id not in idx else None for _id in self.ids ]
        self.embeddings = [item if index not in idx else None for index, item in enumerate(self.embeddings) ]
        self.metadatas = [item if index not in idx else None for index, item in enumerate(self.metadatas)]
        
        self.indices_to_consider = [i for i in self.indices_to_consider if i not in idx]
        self.inverted_index = {label: [v for v in vectors if v not in idx] for label, vectors in self.inverted_index.items()}
        
        return self.ids, self.embeddings, self.metadatas, self.inverted_index
