import hnswlib
import numpy as np
from typing import List, Dict
from abstract_index import AbstractIndex


class HNSWIndex(AbstractIndex):
    '''
    A class for implementing the Hierarchical Navigable Small World (HNSW) index for efficient vector search.

    This class supports adding embeddings and performing similarity search using the HNSW algorithm. It is designed
    to be used with a large number of embeddings and supports incremental updates to the index.

    Attributes:
        table_name (str): The name of the table associated with this index.
        dimension (int): The dimensionality of the embeddings.
        ids (list): A list of unique IDs for each embedding.
        embeddings (np.ndarray): The matrix of stored embeddings.
        metadatas (List[Dict], optional): Metadata associated with each embedding.
        M (int, optional): The number of links in the HNSW graph. Default is 16.
        ef_construction (int, optional): The number of nearest neighbors to consider during the graph construction. Default is 100.
        metric (str, optional): The distance metric to use for the HNSW algorithm. Default is "cosine".
    '''
    def __init__(
        self,
        table_name: str,
        dimension: int,
        ids: list,
        embeddings: np.array,
        metadatas: List[Dict] = None,
        M: int = 16,
        ef_construction: int = 100,
        metric: str = "cosine",
    ):
        super().__init__(table_name, "HNSW", dimension, ids, embeddings)

        if not isinstance(embeddings, (np.ndarray, list)):
            raise ValueError("Embeddings should be a NumPy array or a list.")

        self.embeddings = (
            embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings)
        )
        self.dimension = dimension

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

        index = hnswlib.Index(space=metric, dim=self.dimension)
        index.init_index(max_elements=self.vector_count, ef_construction=ef_construction, M=16)

