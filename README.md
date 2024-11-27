> This is a work in progress.

# VectorLite
VectorLite is a lightweight, easy-to-use, and scalable vector database. It's primarily built to explore the workings of vector databases and their practical applications.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=11wRls96Y5Fr8_81JQVd6iUw7fRIfqR5g" alt="Alt Text" width="25%">
</div>

## Components of VectorLite 

1. **Database**: The core container for tables and indexes.
2. **Table**: Stores and manages vectors and their metadata.
3. **Index**: Facilitates efficient vector search.
   - **Types of Indexes**:
     - Flat (Brute Force)
     - IVF (Inverted File Index)
     - HNSW (Hierarchical Navigable Small World)
     - PQ (Product Quantization)
     - SQ (Scalar Quantization)

4. **Embedder**: Converts text to vectors using various models.
   - Sentence Transformer
   - OpenAI Embeddings
   - Cohere

### Methods in Index Class
1. **Construct the Index**: Initializes a new index.
2. **Search Vectors in the Index**: Retrieves vectors closest to the query vector.

### Methods in Table Class
1. **Initialise the Table**: Sets up a new table for storing vectors.
2. **Add Vectors to the Table**: Inserts new vectors into the table.
3. **Search Vectors in the Table**: Finds vectors within the table based on a query.
4. **Query the Index**: Searches the index using table data.
5. **Delete Vectors from the Table**: Removes vectors from the table.

### TODO:
- [x] Implement Flat Index
- [x] Implement IVF Index
- [x] Implement HNSW Index
- [ ] Implement PQ Index
- [ ] Implement SQ Index
- [ ] **Persist to Disk**: Implement serialization and deserialization for data persistence.
- [ ] **Add Metadata Filtering**: Allow filtering search results based on metadata.
- [ ] **Search Across Multiple Tables**: Introduce join functionality for querying multiple tables.