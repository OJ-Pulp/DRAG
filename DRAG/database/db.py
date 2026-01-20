from typing import List
import sqlite3
from ._vdb import VectorDB
import json
import pickle
import os

class DocumentDatabase:
    def __init__(self, db_path):
        self.db_path = db_path if db_path.endswith(".sqlite") else db_path + ".sqlite"
        self.conn = None
        self._initialize_database()
        self.vector_store = VectorDB(index_path=self.db_path.replace(".sqlite", ".index"))

    def _initialize_database(self):
        new_db = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path)

        if new_db:
            self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()

        cur.execute('''
            CREATE TABLE documents (
                doc_id INTEGER PRIMARY KEY,
                name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                sim_matrix BLOB,
                sentence_scores BLOB,
                sentence_lengths BLOB
            )
        ''')

        cur.execute('''
            CREATE TABLE sentences (
                doc_id INTEGER,
                sentence_index INTEGER,
                sentence_text TEXT,
                vector_id INTEGER,
                PRIMARY KEY (doc_id, sentence_index),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        cur.execute('''
            CREATE TABLE term_frequency (
                doc_id INTEGER,
                word_index INTEGER,
                word TEXT,
                frequencies_json TEXT,
                PRIMARY KEY (doc_id, word),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')

        self.conn.commit()

    def delete_db(self, db_path: str) -> None:
        """Delete the current database."""
        # Ensure the db_path ends with .sqlite
        if not db_path.endswith(".sqlite"):
            db_path += ".sqlite"

        # Prevent deletion of the currently connected database
        if self.db_path == db_path:
            raise ValueError("Cannot delete the current database. Please close the connection first.")
        # Delete the database file and its associated index file
        if os.path.exists(db_path) and os.path.exists(db_path.replace(".sqlite", ".index")):
            os.remove(db_path)
            os.remove(db_path.replace(".sqlite", ".index"))
        else:
            raise ValueError("Database path is invalid or does not exist. Make sure it ends with '.sqlite'")

    def ingest_file(self, source, matrix, sim_matrix, words, sentences, sentence_scores, sentence_lengths, embeddings, metadata):
        cur = self.conn.cursor()

        # Check if the document already exists
        cur.execute('SELECT doc_id FROM documents WHERE name = ?', (source,))
        existing_doc = cur.fetchone()
        if existing_doc:
            return existing_doc[0]

        # 1. Insert document
        metadata_json = json.dumps(metadata)
        cur.execute('''
            INSERT INTO documents (name, metadata, sim_matrix, sentence_scores, sentence_lengths)
            VALUES (?, ?, ?, ?, ?)
        ''', (source, metadata_json, pickle.dumps(sim_matrix), pickle.dumps(sentence_scores), pickle.dumps(sentence_lengths)))
        doc_id = cur.lastrowid

        # 2. Insert sentences
        for idx, sentence in enumerate(sentences):
            vector_id = self.vector_store.add_document(embeddings[idx])
            cur.execute('''
                INSERT INTO sentences (doc_id, sentence_index, sentence_text, vector_id)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, idx, sentence, vector_id))
 
        # 3. Insert term frequency rows
        for idx, word in enumerate(words):
            cur.execute('''
                INSERT INTO term_frequency (doc_id, word_index, word, frequencies_json)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, idx, word, json.dumps(list(matrix[idx, :]))))

        self.conn.commit()
        return doc_id
    
    def delete_file(self, doc_id):
        """
        Deletes a document and all its related data (sentences, term frequencies).
        """
        cur = self.conn.cursor()
        # Get vector IDs of sentences to delete from the vector store
        cur.execute('SELECT vector_id FROM sentences WHERE doc_id = ?', (doc_id,))
        vector_ids = [row[0] for row in cur.fetchall()]
        # Delete from vector store
        self.vector_store.delete_document(vector_ids)
    
        # Delete related records from child tables first
        cur.execute('DELETE FROM sentences WHERE doc_id = ?', (doc_id,))
        cur.execute('DELETE FROM term_frequency WHERE doc_id = ?', (doc_id,))

        # Then delete the document itself
        cur.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))

        self.conn.commit()

    def change_metadata(self, doc_id, metadata):
        """
        Updates the metadata of a document.
        """
        cur = self.conn.cursor()
        metadata_json = json.dumps(metadata)
        cur.execute('UPDATE documents SET metadata = ? WHERE doc_id = ?', (metadata_json, doc_id))
        self.conn.commit()

    def batch(self, batch_size, offset):
        """
        Retrieves a batch of documents with their metadata (excluding sim_matrix and sentences).

        Args:
            batch_size (int): Number of documents to retrieve.
            offset (int): Starting index to retrieve from.

        Returns:
            List[dict]: List of document dictionaries with keys: doc_id, name, timestamp, metadata.
        """
        cur = self.conn.cursor()

        # Step 1: Get doc_ids
        cur.execute('''
            SELECT doc_id
            FROM documents
            ORDER BY doc_id
            LIMIT ? OFFSET ?
        ''', (batch_size, offset))

        doc_ids = [row[0] for row in cur.fetchall()]

        # Step 2: Fetch desired fields using document_data
        documents = []
        for doc_id in doc_ids:
            data = self.document_data(doc_id, ["doc_id", "name", "timestamp", "metadata"])
            doc_dict = dict(zip(["doc_id", "name", "timestamp", "metadata"], data))
            documents.append(doc_dict)

        return documents
    
    def document_data(self, doc_id, data: List[str]):
        """
        Returns the requested data for a document.
        """
        cur = self.conn.cursor()
        placeholders = ', '.join(data)
        query = f'SELECT {placeholders} FROM documents WHERE doc_id = ?'
        cur.execute(query, (doc_id,))
        row = cur.fetchone()
        if row:
            row = list(row)
            for i, item in enumerate(row):
                if isinstance(item, bytes):
                    row[i] = pickle.loads(item)
        return tuple(row) if len(data) > 1 else row[0]
                
    def global_matrix(self):
        """
        Create a similarity matrix using term overlap between documents,
        then subtract each value in a row from the diagonal element of that row.
        Returns a list of lists of shape (num_docs, num_docs).
        """
        cur = self.conn.cursor()
        cur.execute('SELECT doc_id FROM documents ORDER BY doc_id')
        doc_ids = [row[0] for row in cur.fetchall()]
        num_docs = len(doc_ids)

        # Initialize the matrix with zeros
        kw_sim_matrix = [[0] * num_docs for _ in range(num_docs)]

        # Fill the similarity matrix
        for i in range(num_docs):
            for j in range(num_docs):
                if i <= j:  # Use symmetry
                    cur.execute('''
                        SELECT COUNT(*) FROM term_frequency tf1
                        JOIN term_frequency tf2 ON tf1.word = tf2.word
                        WHERE tf1.doc_id = ? AND tf2.doc_id = ?
                    ''', (doc_ids[i], doc_ids[j]))
                    overlap_count = cur.fetchone()[0]
                    kw_sim_matrix[i][j] = overlap_count
                    kw_sim_matrix[j][i] = overlap_count

        for i in range(num_docs):
            diag = kw_sim_matrix[i][i]
            for j in range(num_docs):
                kw_sim_matrix[i][j] /= diag

        return kw_sim_matrix, doc_ids

    def _doc_vector_search(self, query_vector):
        """
        Searches for the top k most similar documents to the given query vector.
        Returns a list of unique doc_ids.
        """
        _similarities, vector_ids = self.vector_store.search(query_vector)
        results = []

        for i, vector_id in enumerate(vector_ids):
            cur = self.conn.cursor()
            cur.execute(
                'SELECT doc_id FROM sentences WHERE vector_id = ?',
                (vector_id,)
            )
            row = cur.fetchone()
            if row:
                doc_id = row[0]
                if doc_id not in results:
                    results.append(doc_id)

        return results

    def _doc_keyword_search(self, terms, k: int = None):
        """
        Searches for documents containing any of the specified terms.
        Returns a list of unique doc_ids.
        """
        if not terms:
            return []

        cur = self.conn.cursor()
        placeholders = ', '.join('?' for _ in terms)
        query = f'''
            SELECT DISTINCT doc_id
            FROM term_frequency
            WHERE word IN ({placeholders})
        '''
        cur.execute(query, terms)
        if k is not None:
            rows = cur.fetchmany(k)
        else:
            rows = cur.fetchall()
        return [row[0] for row in rows]
    
    def document_search(self, query_terms: List[str], query_vector: List[float], overlap: bool = True):
        """
        Searches for documents that match the query terms and are similar to the query vector.
        Returns a list of unique doc_ids.
        """
        keywords_results = None
        vector_results = None
        if not query_terms and not query_vector:
            return results
        
        if query_terms is not None:
            keywords_results = self._doc_keyword_search(query_terms)
        if query_vector is not None:
            vector_results = self._doc_vector_search(query_vector)
        
        # If both keyword and vector search results are available, combine them based on the overlap parameter
        if overlap:
            results = set(keywords_results) & set(vector_results)
        else:
            results = set(keywords_results) | set(vector_results)
        return list(results)
    
    def sent_keyword_search(self, query_terms: List[str], doc_id: str):
        """
        Given a document ID and a list of terms, return a list of frequency vectors
        (parsed from JSON), ordered by word_index.
        """
        cur = self.conn.cursor()

        placeholders = ','.join('?' for _ in query_terms)
        query = f'''
            SELECT frequencies_json
            FROM term_frequency
            WHERE doc_id = ? AND word IN ({placeholders})
            ORDER BY word_index ASC
        '''

        cur.execute(query, (doc_id, *query_terms))
        rows = cur.fetchall()

        return [json.loads(freqs_json[0]) for freqs_json in rows]

    def sent_vector_search(self, query_vector: List[float], doc_id: str, k: int = 5):
        """
        Searches the vector store and filters results by doc_id.
        Returns a list of (sentence_index, similarity) for valid matches.
        """
        similarities, vector_ids = self.vector_store.search(query_vector, k)
        results = []

        cur = self.conn.cursor()
        for sim, vector_id in zip(similarities, vector_ids):
            cur.execute(
                'SELECT sentence_index FROM sentences WHERE vector_id = ? AND doc_id = ?',
                (vector_id, doc_id)
            )
            row = cur.fetchone()
            if row:
                sentence_index = row[0]
                results.append((sentence_index, sim))

        return results
        
    def sentences(self, doc_id, sentence_indices: List[int]):
        """
        Gets all or specified sentences from a document, ordered by sentence_index.
        """
        # Ensure sentence_indices is a list of integers
        if not all(isinstance(i, int) for i in sentence_indices):
            raise ValueError("All items in sentence_indices must be integers.")
        
        cur = self.conn.cursor()

        # Create placeholders for the IN clause
        placeholders = ','.join(['?'] * len(sentence_indices))

        # Define the SQL query with proper ordering
        query = f'''
            SELECT sentence_text FROM sentences
            WHERE doc_id = ? AND sentence_index IN ({placeholders})
            ORDER BY sentence_index
        '''

        # Combine doc_id with the sentence_indices list and execute the query
        cur.execute(query, (doc_id, *sentence_indices))

        # Fetch and return the result (i.e., list of sentences)
        return [row[0] for row in cur.fetchall()]

    def get_connection(self):
        return self.conn

    def close(self):
        self.vector_store.save()
        if self.conn:
            self.conn.close()
