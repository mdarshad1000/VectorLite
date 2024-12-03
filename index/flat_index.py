import numpy as np
from typing import List, Dict, Union
from .abstract_index import AbstractIndex


class FlatIndex(AbstractIndex):
    """
    A simple, brute-force indexing class for embedding-based search.

    This class supports adding embeddings and performing similarity search
    using cosine similarity. It is intended for small-scale datasets due to
    its linear time complexity. Guarantees best accuracy

    Attributes:
        table_name (str): The name of the table associated with this index.
        ids (list): A list of unique IDs for each embedding.
        dimension (int): The dimensionality of the embeddings.
        embeddings (np.ndarray): The matrix of stored embeddings.
        metadatas (List[Dict], optional): Metadata associated with each embedding.
    """

    def __init__(self,table_name: str,dimension: int,ids: List,embeddings: np.array,metadatas: List[Dict] = None,):
        super().__init__(table_name, "FlatIndex (Brute Force)", dimension, ids, embeddings)

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings))
        self.dimension = dimension

        # if 1D array, then convert to 2D
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

    def add(self, idx: List[int], vector: Union[List, np.array], metadata: List[Dict] = None):

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


        for _ in range(len(idx)):
            self._update_vector_count()

        self.ids.extend(idx)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.extend(metadata) if metadata else None

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

        # Calculate Cosine similarity -> cosine_similarity(A, B) = cos(Θ) = (A.B) / (|A|*|B|)
        cosine_similarities = np.dot(self.embeddings, query_vector) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)+ 1e-10)

        # Use argpartition to quickly get indices of the top_k highest similarity scores
        unsorted_top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]

        # Sort these indices by similarity scores in descending order
        sorted_top_k_indices = unsorted_top_k_indices[np.argsort(-cosine_similarities[unsorted_top_k_indices])]

        # Retrieve the top_k Result
        result = [
            {
                "id": self.ids[i],
                "embedding": self.embeddings[i],
                "metadata": self.metadatas[i] if self.metadatas else {},
                "score": cosine_similarities[i],
            }
            for i in sorted_top_k_indices
        ]
        return result

    
    def delete_vector(self, idx: List[int]):

        if any(i not in self.ids for i in idx):
            raise ValueError("One or more IDs not in the index.")
        
        # Handle input validation
        if not isinstance(idx, List):
            raise ValueError("ID should be a List.")
        
        # Remove the IDs from self.ids
        self.ids = [_id for _id in self.ids if _id not in idx]
        self.embeddings = [item for index, item in enumerate(self.embeddings) if index not in idx]
        self.metadatas = [item for index, item in enumerate(self.metadatas) if index not in idx]

        return self.ids, self.embeddings, self.metadatas