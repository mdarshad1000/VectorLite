import hnswlib
import numpy as np
from typing import List, Dict, Union
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
    def __init__(self, table_name: str, dimension: int, ids: list, embeddings: np.ndarray, metadatas: List[Dict] = None, M: int = 16, ef_construction: int = 100, metric: str = "cosine", max_elements: int = None):
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
        self.max_elements = max_elements if max_elements is not None else self.vector_count

        self._build_index(max_element=self.max_elements)

    def add(self, idx: Union[int, List[int]], vector : Union[List, np.array], metadata: Union[Dict, List[Dict], None] = None):
        
        new_size = self.vector_count + len(idx) if isinstance(idx, list) else self.vector_count + 1

        if new_size > self.max_elements:
            self._build_index(max_element=new_size)

        # Handle input validation
        if not isinstance(vector, (np.ndarray, list)):
            raise ValueError("Vector should be a NumPy array or a list.") 
        
        vector = vector if isinstance(vector, np.ndarray) else np.array(vector) 
        
        if isinstance(idx, int):
            idx = [idx]
            vector = vector[0] 
            # Validate the single vector's dimensions
            if vector.shape[0] != self.dimension:
                raise ValueError("Vector dimension does not match index dimension.")
        else:
            if len(idx) != vector.shape[0]:
                raise ValueError("Each ID must correspond to a single vector.")
            
            # Validate each vector's dimensions
            if any(v.size != self.dimension for v in vector):
                raise ValueError("One or more vectors' dimensions do not match the index dimension.")

        # Check for existing IDs
        if any(i in self.ids for i in idx):
            raise ValueError(f"One or more IDs already exist in the index.")

        for _ in range(len(idx)):
            self._update_vector_count()

        self.embeddings = np.vstack([self.embeddings, vector])
        self.metadatas.extend(metadata) if metadata else None
        self.index.add_items(vector, idx)

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

    def _build_index(self, max_element:int):
        index = hnswlib.Index(space=self.metric, dim=self.dimension)
        index.init_index(max_elements=max_element, ef_construction=self.ef_construction, M=self.M)
        index.add_items(self.embeddings, self.ids)
        self.index = index
    