from abstract_index import AbstractIndex
from typing import List, Dict
import numpy as np


class FlatIndex(AbstractIndex):

    def __init__(
        self,
        table_name: str,
        ids: list,
        dimension: int,
        embeddings: np.array,
        metadatas: List[Dict] = None,
    ):
        super().__init__(
            table_name, "FlatIndex (Brute Force)", ids, dimension, embeddings
        )

        self.ids = ids
        self.embeddings = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings # if embeddings in not a 2D array, convert it into a 2D array (List of vectors)
        self.metadatas = metadatas if metadatas else []

    def add(self, id: int, vector: np.array, metadata: Dict = None):
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension does not match index dimension i.e {self.dimension}"
            )

        self.ids.append(id)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.append(metadata) if metadata else None

    def search(self, query_vector: np.array, top_k: int, filter_param: Dict = None):
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Vector dimension does not match index dimension i.e {self.dimension}"
            )

        # TODO: Implement Similarity Search and Metadata Filtering
        pass

        