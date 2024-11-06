import numpy as np
from typing import List
from abc import ABC, abstractmethod

class AbstractIndex(ABC):

    def __init__(self, table_name: str, index_type: str, dimension: int, ids: List, embeddings: np.array):
        self.table_name = table_name
        self.index_type = index_type
        self.dimension = dimension
        self.ids = ids
        self.embeddings = embeddings
        self.vector_count = self.embeddings.shape[0]
        
    @abstractmethod
    def add(self):
        """
        Add vectors to the index.
        """
        pass

    @abstractmethod
    def search(self):
        """
        Search for vectors in the index
        """
        pass

    def _update_vector_count(self):
        """
        Update vector count in the index.
        """
        self.vector_count += 1

    def __str__(self):
        """
        Get a string representation of the index.

        Returns:
            str: A string describing the index.
        """
        return f"Index Name: {self.index_type} with Dimensions: {self.dimension}"
    
    def __repr__(self):
        """
        Get a string representation of the index for debugging.

        Returns:
            str: A string describing the index, including its name and dimension.
        """
        return f"{self.__class__.__name__}(index_type={self.index_type}, dimension={self.dimension})"

    def __len__(self):
        """
        Get the number of vectors in the index.

        Returns:
            int: The number of vectors in the index.
        """
        return self.vector_count