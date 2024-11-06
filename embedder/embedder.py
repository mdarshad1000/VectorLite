import os
from typing import List
from dotenv import load_dotenv
from abc import ABC, abstractmethod


# Abstract Embedder Class
class Embedder(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Abstract method to fetch embeddings for a list of texts using a specified model.
        """
        pass


# OpenAI Embedder
class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-ada-002"):
        from openai import OpenAI

        load_dotenv()

        # Set the API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found in environment variables.")

        # Initialize the client and store model name
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch embeddings for a list of texts from OpenAI's embedding model.
        """
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [i.embedding for i in response.data]
        return embeddings


# Cohere Embedder
class CohereEmbedder(Embedder):
    def __init__(self, api_key: str = None, model_name: str = "embed-english-v3.0"):
        import cohere

        load_dotenv()

        # Set the API key
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API Key not found in environment variables.")

        # Initialize the client and store model name
        self.client = cohere.Client(self.api_key)
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch embeddings for a list of texts from Cohere's embedding model.
        """
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document",
            embedding_types=["float"]
        )
        embeddings = [i for i in response.embeddings.float]
        return embeddings


# SentenceTransformer Embedder
class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        # Initialize the client and store model name
        self.client = SentenceTransformer(model_name)
        self.model_name = model_name

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch embeddings for a list of texts from SentenceTransformers' model.
        """
        embeddings = self.client.encode(texts)
        return embeddings


# Factory function to select the correct embedder
def get_embedder(model_provider: str, model_name: str, api_key: str = None) -> Embedder:
    """
    Factory function to return the appropriate Embedder class based on the model name.
    """
    if model_provider.lower() == "openai":
        return OpenAIEmbedder(api_key=api_key)
    elif model_provider.lower() == "cohere":
        return CohereEmbedder(api_key=api_key)
    elif model_provider.lower() == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


if __name__ == "__main__":

    # Example: Get embeddings using SentenceTransformer
    embedder = get_embedder("sentence-transformer", model_name="all-MiniLM-L6-v2")
    sentences = ["Hello, world!", "How are you?"]

    embeddings = embedder.get_embeddings(sentences)
    print(f"SentenceTransformer Embeddings: {embeddings} with dimensions {len(embeddings[0])}")

    # Example: Get embeddings using OpenAI (make sure your OPENAI_API_KEY is set)
    embedder = get_embedder("openai", model_name="text-embedding-ada-002")
    embeddings = embedder.get_embeddings(sentences)
    print(f"OpenAI Embeddings: {embeddings} with dimensions {len(embeddings[0])}")

    # Example: Get embeddings using Cohere (make sure your COHERE_API_KEY is set)
    embedder = get_embedder("cohere", model_name="embed-english-v3.0")
    embeddings = embedder.get_embeddings(sentences)
    print(f"Cohere Embeddings: {embeddings} with dimensions {len(embeddings[0])}")
