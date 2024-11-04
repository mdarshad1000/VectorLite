from abc import ABC, abstractmethod

class AbstractIndex(ABC):

    def __init__(self, table_name: str, index_type: str, ids: list, dimension: int, vector_count: int):
        self.table_name = table_name
        self.index_type = index_type
        self.ids = ids
        self.dimension = dimension
        self.vector_count = vector_count

        
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

    @classmethod
    def _update_vector_count(cls):
        """
        Update vector count in the index.
        """
        pass

    def __str__(self):
        """
        Get a string representation of the index.

        Returns:
            str: A string describing the index.
        """
        return f"Index Name: {self.index_name} with Dimensions: {self.dimension}"
    
    def __repr__(self):
        """
        Get a string representation of the index for debugging.

        Returns:
            str: A string describing the index, including its name and dimension.
        """
        return f"{self.__class__.__name__}(index_name={self.index_name}, dimension={self.dimension})"

    def __len__(self):
        """
        Get the number of vectors in the index.

        Returns:
            int: The number of vectors in the index.
        """
        return len(self.vector_count)