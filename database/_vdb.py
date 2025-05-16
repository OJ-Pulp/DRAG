import os
import faiss
import numpy as np
from typing import List

class VectorDB:
    def __init__(self, dimension: int = 384, index_path: str = "index.index"):
        self.dimension = dimension
        self.index_path = index_path

        # Load or initialize the FAISS index with support for IDs, using IP (Inner Product) for cosine similarity
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Use IndexFlatIP (Inner Product) for cosine similarity
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

        self.next_id = self._get_next_id()

    def _get_next_id(self):
        return self.index.ntotal

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def add_document(self, embedding: List[float]) -> int:
        # Embed the text and normalize the vector
        embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding to unit length
        
        vector_id = self.next_id
        self.index.add_with_ids(
            np.array([embedding], dtype=np.float32),
            np.array([vector_id], dtype=np.int64)
        )
        self.next_id += 1
        self.save()
        return vector_id
    
    def delete_document(self, vector_ids: List[int]) -> None:
        ids = np.array(vector_ids, dtype=np.int64)
        self.index.remove_ids(faiss.IDSelectorBatch(ids))
        self.save()

    def search(self, query_vector: List[float], k: int = 5):
        if self.index.ntotal == 0:
            return [], []

        # Normalize the query vector for cosine similarity
        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)

        # Perform the search (inner product = cosine similarity, since vectors are normalized)
        cos_sims, indices = self.index.search(np.array([query_vector]), k)

        # Flatten the outputs
        cos_sims = cos_sims[0]
        indices = [int(i) for i in indices[0]]

        return cos_sims, indices