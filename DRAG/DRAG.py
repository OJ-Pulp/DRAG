from . import config
from .database.db import DocumentDatabase
from .nlp.toolkit import toolkit
from typing import List

class DRAG:
    def __init__(self, db_name, db_path, alpha: float = 0.55, beta: float = 0.35, gamma: float = 0.10) -> None:
        full_path = config.join_paths(db_path, db_name)
        self.db = DocumentDatabase(full_path)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.toolkit = toolkit(config.EMBEDDING_DIR,
                               config.RERANKER_DIR, 
                               config.STOP_WORDS_DIR,
                               alpha, beta, gamma)
    
    def change_db(self, db_path, db_name: str) -> None:
        """Change the database to a new one."""
        self.db.close()
        self.db = DocumentDatabase(config.join_paths(db_path, db_name))
    
    def delete_db(self, db_path, db_name: str) -> None:
        """Delete the current database."""
        self.db.delete_db(config.join_paths(db_path, db_name))

    def upload(self, text: str, source: str, metadata: dict = None) -> str:
        """Add a new document to the database."""
        data = self.toolkit.extract_file(text)
        data["source"] = source
        data["metadata"] = data.get("metadata", {}) | (metadata or {})
        doc_id = self.db.ingest_file(**data)
        return doc_id

    def delete(self, document_id: str) -> None:
        """
        Delete a document from the database.
        """
        self.db.delete_file(document_id)

    def update_metadata(self, document_id: str, metadata: dict) -> None:
        """
        Update the metadata of a document in the database.
        """
        self.db.change_metadata(document_id, metadata)
    
    def batch(self, batch_size: int = 100, offset: int = 0) -> List[dict]:
        """
        Get documents from the database in batches.
        """
        return self.db.batch(batch_size, offset)
    
    def clusters(self) -> List[List[int]]:
        """
        Return a list of clustered documents within the database each represented by a list of document IDs.
        """
        # 1. Get the overlap of terms between documents
        overlap_matrix, document_ids = self.db.global_matrix()
        # 2. Combine into a transition matrix
        transition_matrix = self.toolkit.normalize_and_weight([overlap_matrix], [1])
        # 4. Get the clusters using the transition matrix
        clusters = self.toolkit.clusters(transition_matrix)
        # 6. Create a list of clusters with the document ids
        clusters = [[document_ids[i] for i in cluster] for cluster in clusters]
        return clusters

    def summarize(self, document_id: str, top_k: int = 10) -> List[int]:
        """
        Summarize the document using TextRank.
        """
        # 1. Get sentence scores for the specified document
        sentence_scores = self.db.document_data(document_id, ["sentence_scores"])
        # 2. Rank the sentences
        top_sentences_idxs = self.toolkit.sort(sentence_scores, top_k)
        # 3. Get the text of the top sentences
        return self.db.sentences(document_id, top_sentences_idxs)
    
    def query_document(self, query: str, document_id: str, max_sentences: int = 10) -> List[str]:
        """
        Query a specific document and return the query focused summarization.
        """
        # 1. Get the document matrix for the specified document
        doc_matrix, document_sent_scores, sentence_lengths = self.db.document_data(document_id, ["sim_matrix", "sentence_scores", "sentence_lengths"])
        num_sentences = len(sentence_lengths)
        # 2. Tokenize the query
        query_terms = self.toolkit.preprocess_query(query)
        query_vector = self.toolkit.embed(query)
        # 3. Create sentence weights using the query terms
        keyword_matrix = self.db.sent_keyword_search(query_terms, document_id)
        keyword_sent_scores = self.toolkit.keyword_sent_scores(keyword_matrix, sentence_lengths)
        # 4. Create sentence weights using the semantic meaning of the query
        semantic_sents = self.db.sent_vector_search(query_vector, document_id, num_sentences)
        semantic_sent_scores = self.toolkit.semantic_sent_scores(semantic_sents, num_sentences)
        # 5. Create the query nodes by combining the semantic meaning of query, keywords used in query, and the document matrix (sentence scores)
        query_nodes = self.toolkit.normalize_and_weight([semantic_sent_scores, keyword_sent_scores, document_sent_scores], [self.alpha, self.beta, self.gamma])
        # 6. Create distance edges by inverting the document matrix (i.e. 1 - doc_matrix)
        query_edges = (1 - doc_matrix)
        # 7. Use the query related nodes and the document matrix to get the top sentences
        top_sentence_idxs = self.toolkit.query_document(query_nodes, query_edges, max_sentences)
        # 8. Get the text of the top sentences
        return self.db.sentences(document_id, top_sentence_idxs)
 
    def query(self, query: str, max_sentences: int = 10, max_documens: int = 3) -> List[List[str]]:
        """
        Query the database and return the query focused summarization for all documents.
        """
        # 1. Tokenize the query
        query_terms = self.toolkit.preprocess_query(query)
        query_vector = self.toolkit.embed(query)
        # 2. Get the document ids that match the query terms and the vector search
        document_ids = self.db.document_search(query_vector=query_vector, query_terms=query_terms, overlap=True)
        results = []
        for doc_id in document_ids:
            # 3. Get the answer for the document id using the query focused summarization
            top_sentence_idxs = self.query_document(query, doc_id, max_sentences)
            # 4. Append the results to the list of results
            results.append(
                {
                    "document_id": doc_id,
                    "answer": top_sentence_idxs,
                }
            )
        if len(results) == 0:
            return []
        else:   
            # 5. Rerank the results using the reranker model 
            # Note: if the answer is too long, it will be truncated and the score will be based on the truncated answer
            return self.toolkit.rerank(query, results)[0:max_documens]