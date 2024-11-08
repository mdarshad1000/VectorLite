import numpy as np
from typing import List, Dict
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

    def __init__(
        self,
        table_name: str,
        dimension: int,
        ids: List,
        embeddings: np.array,
        metadatas: List[Dict] = None,
    ):
        super().__init__(
            table_name, "FlatIndex (Brute Force)", dimension, ids, embeddings
        )

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (
            embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings)
        )
        self.dimension = dimension

        # if 1D array, then convert to 2D
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, self.dimension)

        elif self.embeddings.ndim > 2:
            raise ValueError("Embeddings should be a 1D or 2D array.")

        if self.embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings dimension does not match index dimension i.e {self.dimension}"
            )

        self.table_name = table_name
        self.ids = ids
        self.metadatas = metadatas if metadatas is not None else []
        self.vector_count = self.embeddings.shape[0]

    def add(self, id: int, vector: np.array, metadata: Dict = None):
        if id in self.ids:
            raise ValueError(f"ID {id} already exists in the index.")

        if vector.ndim != 1:
            raise ValueError("Input vector must be 1-dimensional.")

        if vector.size != self.dimension:
            raise ValueError(
                f"Vector dimension ({vector.size}) does not match index dimension ({self.dimension})."
            )

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

        # Calculate Cosine similarity -> cosine_similarity(A, B) = cos(Θ) = (A.B) / (|A|*|B|)
        cosine_similarities = np.dot(self.embeddings, query_vector) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
            + 1e-10
        )

        # Use argpartition to quickly get indices of the top_k highest similarity scores
        unsorted_top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]

        # Sort these indices by similarity scores in descending order
        sorted_top_k_indices = unsorted_top_k_indices[
            np.argsort(-cosine_similarities[unsorted_top_k_indices])
        ]

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
