from embedder.embedder import get_embedder
from index.flat_index import FlatIndex
from index.ivf_index import IVFIndex
import numpy as np

docs = [
    "He scored the winning goal in the final seconds of the game.",
    "The concert last night was absolutely electrifying!", 
    "She spent hours reviewing her notes for the biology exam.", 
    "The new action movie just released in theaters.", 
    "He attended a weekend seminar on advanced calculus.",  
    "She completed her physics homework before dinner.", 
    "The tennis match went into an intense tiebreaker.",  
    "The comedy show had everyone laughing nonstop.", 
    "He joined a local soccer league to stay active.", 
    "They are practicing their lines for the upcoming play.",  
    "She’s researching for her history paper due next week.", 
    "The boxing match was thrilling and kept everyone on edge.", 
    "He is learning guitar to perform at the talent show.", 
    "The chess club meets every Wednesday to improve strategies.", 
    "She trained hard for the marathon happening next month.", 
    "The award ceremony was a glamorous event with many celebrities.",  
    "He studied late into the night for his chemistry exam.", 
    "The football team celebrated after a hard-fought victory.", 
    "The new TV series has everyone talking about the plot twists."
]

ids = [i for i in range(len(docs))]

metadatas = [
    {
        "id": i,
        "text": docs[i],
    }
    for i in range(len(docs))
]

# Get embeddings for the documents
embedder = get_embedder("openai", model_name="text-embedding-ada-002")
embeddings = embedder.get_embeddings(docs)

# Get the Index
index = IVFIndex(
    "Experiment_table",
    dimension=1536,
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
)


# search for the top 5 most similar documents
query_embedding = np.array(embedder.get_embeddings(["I love playing football"])[0])
top_k = 5

results = index.search(query_embedding, top_k)

for result in results:
    print(f"Document ID: {result['id']}, Score: {result['score']}, metadata: {result['metadata']}")