> This is a work in progress.

# VectorLite
VectorLite is a lightweight, easy-to-use, and scalable vector database. Primarily built to study how vector DBs work and how they can be used in practice.

## Components of VectorLite 

1. Database 
2. Table 
3. Index
  A. Flat
  B. IVF
  C. HNSW
  D. PQ
  E. SQ

4. Embedder
  A. Sentence Transformer
  B. OpenAI
  C. Cohere


### Methods in Index Class
1. Initialise the Index
2. Add Vectors to the Index
3. Search Vectors in the Index


### Methods in Table Class
1. Initialise the Table
2. Add Vectors to the Table
3. Search Vectors in the Table
4. Query the Index
5. Delete Vectors from the Index

### TODO:
-> Persist to Disk using basic Serialisation and Deserialisation
-> Add MetaData Filtering
-> Search across multiple tables (Join Functionality?)
-> Figure a way to add a WHERE clause (it's super handy) -- a lot of databases don't have this.
