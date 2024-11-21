import hnswlib
import numpy as np
from typing import List, Dict
from index.abstract_index import AbstractIndex

# TODO: Include Metadata filtering (post and pre filtering)
# TODO: Add support for Update and Delete

class HNSWIndex(AbstractIndex):
    '''
    A class for implementing the Hierarchical Navigable Small World (HNSW) index for efficient vector search.

    This class supports adding data and performing similarity search using the HNSW algorithm. It is designed
    to be used with a large number of embeddings and supports incremental updates to the index.

    Attributes:
        table_name (str): The name of the table associated with this index.
        dimension (int): The dimensionality of the embeddings.
        ids (list): A list of unique IDs for each embedding.
        embeddings (np.ndarray): The matrix of stored embeddings.
        metadatas (List[Dict], optional): Metadata associated with each embedding.
        M (int, optional): The number of links in the HNSW graph. Default is 16.
        ef_construction (int, optional): The number of nearest neighbors to consider during the graph construction. Default is 100.
        metric (str, optional): The distance metric to use for the HNSW algorithm. Default is "cosine". Other possible options are 'l2' and 'ip'
    '''
    def __init__(self, table_name: str, dimension: int, ids: list, embeddings: np.array, metadatas: List[Dict] = None, M: int = 16, ef_construction: int = 100, metric: str = "cosine"):
        super().__init__(table_name, "HNSW", dimension, ids, embeddings)

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings))
        self.dimension = dimension

        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, self.dimension)

        elif self.embeddings.ndim > 2:
            raise ValueError("Embeddings should be a 1D or 2D array.")

        if self.embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embeddings dimension does not match index dimension i.e {self.dimension}")

        self.table_name = table_name
        self.ids = ids
        self.metadatas = metadatas if metadatas is not None else []
        self.vector_count = self.embeddings.shape[0]
        self.metric = metric
        self.ef_construction = ef_construction
        self.M = M

        self._build_index()

    def _build_index(self):
        index = hnswlib.Index(space=self.metric, dim=self.dimension)
        index.init_index(max_elements=self.vector_count+100, ef_construction=self.ef_construction, M=self.M)
        index.add_items(self.embeddings, self.ids)
        self.index = index
    
    def add(self, id: int, vector: np.array, metadata: Dict = None):
        
        # Handle input validation
        if not isinstance(vector, (np.ndarray, list)):
            raise ValueError("Vector should be a NumPy array or a list.")
        
        vector = vector if isinstance(vector, np.ndarray) else np.array(vector)

        # By default the Embedder module returns a 2D Array, so we need to handle that case
        if vector.ndim != 1 and len(vector[0]) == self.dimension:
            vector = vector[0]

        if id in self.ids:
            raise ValueError(f"ID {id} already exists in the index.")

        if vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if vector.size != self.dimension:
            raise ValueError(f"Vector dimension ({vector.size}) does not match index dimension ({self.dimension}).")

        self.index.add_items(vector, id)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.append(metadata) if metadata else None
        self._update_vector_count()

    def search(self, query_vector: np.ndarray, top_k: int, filter_param: Dict = None):
        if top_k <= 0:
            raise ValueError("Top K must be greater than 0")

        if query_vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if query_vector.size != self.dimension:
            raise ValueError(f"Vector dimension ({query_vector.size}) does not match index dimension ({self.dimension}).")

        labels, distances = self.index.knn_query(query_vector, k=top_k)
        top_k_indices = labels[0]
        score = distances[0]

        results = [
            {
                "id": item,
                "embedding": self.embeddings[item],
                "metadata": (self.metadatas[item] if self.metadatas else {}),
                "score": score[i],
            }
            for i, item in enumerate(top_k_indices)
        ]
        return results

        