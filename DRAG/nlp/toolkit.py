from .preprocess import preprocessor
from .models.loaders import SentenceEmbedder, Reranker
from .mathlib import overlap_sim_matrix, embed_sim_matrix, markov_chain, max_profit_route
import numpy as np
from typing import List, Union, Tuple

class toolkit:
    def __init__(self, embedder_path: str, reranker_path: str, stop_words_path: str, alpha: float, beta: float, gamma: float) -> None:
        self.preprocessor = preprocessor(stop_words_path)
        self.reranker = Reranker(reranker_path)
        self.embedder = SentenceEmbedder(embedder_path)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def extract_file(self, text: str, num_labels: int = 3) -> dict:
        """
        Extracts data from the text using the preprocessor.
        """
        # Chunk the text into sentences
        sentences = self.preprocessor.chunk(text)
        # Filter out noisy sentences
        sentences = self.preprocessor.clean(sentences)
        # Create embeddings and term frequency for the sentences
        embeddings = self.embed(sentences)
        tf_matrix, stemmed_words, original_words = self.preprocessor.tf(sentences, include_terms=True, keys=True)
        # Weight the TF matrix using BM25
        sentence_lengths = np.sum(tf_matrix, axis=0, keepdims=True).astype(int).flatten().tolist()
        bm25_matrix = self.preprocessor.bm25(tf_matrix, sentence_lengths)
        # Create a similarity matrix using the sentence embeddings and term overlap
        sim_matrix = self.doc_graph(tf_matrix, embeddings)
        # Create a list of the top k words for the topic model
        metadata = {
            "_tags" : [original_words[idx] for idx in self.topic_model(bm25_matrix, top_k=num_labels)]
        }
        return {
            "embeddings": embeddings,
            "matrix": bm25_matrix,
            "sim_matrix": sim_matrix,
            "words": stemmed_words,
            "sentences": sentences,
            "sentence_scores": markov_chain(sim_matrix),
            "sentence_lengths": sentence_lengths,
            "metadata": metadata
        }
    
    def doc_graph(self, tf_matrix: np.ndarray, embeddings: List[List[float]]) -> np.ndarray:
        """
        Creates a document graph based on the term frequency matrix.
        Returns a similarity matrix of the sentences.

        :param tf_matrix: The term frequency matrix for the document.
        :param embeddings: The sentence embeddings for the document. 
        :param alpha: The weight for the embedding similarity. Make majority if document does not contain niche information.
        :param beta: The weight for the term overlap similarity. Make majority if document contains niche information.
        :param gamma:The weight for the sentence proximity. Random walk between sentences that are close to each other. Adds uniform noise to the graph.
        :return: A similarity matrix of the sentences.
        """
        # Create a similarity matrix based on sentence embeddings and term overlap
        embed_matrix = embed_sim_matrix(embeddings)
        overlap_matrix = overlap_sim_matrix(tf_matrix)
        # Combine the embedding matrix with the overlap matrix
        num_sent = len(embeddings)
        # Prevent self-loops by setting the diagonal to 0
        embed_matrix[np.arange(num_sent), np.arange(num_sent)] = 0
        overlap_matrix[np.arange(num_sent), np.arange(num_sent)] = 0
        # Random walks between sentences
        upper = np.eye(num_sent, k=1)
        lower = np.eye(num_sent, k=-1)
        sentence_proximity = upper + lower
        # Create a transition matrix based on the semantic similarity, term overlap, and sentence proximity
        return self.normalize_and_weight([embed_matrix, overlap_matrix, sentence_proximity], [self.alpha, self.beta, self.gamma])
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds the text using the preprocessor.
        """
        if isinstance(text, str):
            return self.embedder.embed(text)
        else:
            return np.array([self.embedder.embed(sentence) for sentence in text], dtype=np.float32)
        
    def normalize_and_weight(self, matrix_list: List[np.ndarray], weight_list: List[float]) -> np.ndarray:
        """
        Creates a stochastic matrix/vector as a weighted sum of input matrices.
        Each matrix is normalized into a probability distribution before weighting.
        """
        # Make sure every matrix is a numpy array
        matrix_list = [np.array(m) for m in matrix_list]

        normalized = [
            m / np.clip(m.sum() if m.ndim == 1 else m.sum(axis=1, keepdims=True), 1e-10, None)
            for m in matrix_list
        ]

        weighted = [w * m for w, m in zip(weight_list, normalized)]
        return sum(weighted)

    def keyword_sent_scores(self, array: List[List[float]], sentence_lengths: List[int]) -> np.ndarray:
        """
        Creates a keyword node matrix from the term frequency matrix.
        The keyword node matrix is a 1 x sentence matrix where each element is the sum of the BM25 scores for each term in the sentence.
        The BM25 scores are calculated using the term frequency matrix and the sentence lengths.
        """
        query_matrix = np.array(array)
        bm25_matrix = self.preprocessor.bm25(query_matrix, sentence_lengths)
        return np.sum(bm25_matrix, axis=0)
    
    def semantic_sent_scores(self, pairs: List[Tuple[int, float]], k: int) -> List[float]:
        """
        Computes a sorted and padded similarity list.
        """
        if not pairs:
            return [0.0] * k

        # Convert to NumPy array for efficient operations
        arr = np.array(pairs, dtype=[('index', int), ('sim', float)])

        # Sort by sentence_index
        sorted_arr = np.sort(arr, order='index')

        # Extract the similarity scores
        similarities = sorted_arr['sim']

        # Pad or trim to length k
        if len(similarities) < k:
            similarities = np.pad(similarities, (0, k - len(similarities)), constant_values=0.0)
        else:
            similarities = similarities[:k]

        return similarities.tolist()

    def preprocess_query(self, text: str) -> List[str]:
        """
        Extracts the query terms from the text.
        """
        return self.preprocessor.preprocess_query(text)
    
    def sort(self, array: np.ndarray, top_k: int) -> List[int]:
        """
        Ranks the array and returns the top_k indices.
        """
        return [int(idx) for idx in np.argsort(array)[::-1][:top_k]]
    
    def rerank(self, query: str, results: List[List[str]], threshold: float = -4.0) -> List[float]:
        """
        Reranks the answers based on the query and filters out the results that are below the threshold.
        """
        pairs = [(query, " ".join(result["answer"])) for result in results]
        scores = self.reranker.rerank(pairs)
        sorted_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [result for result, score in sorted_results if score > threshold]
        
    def clusters(self, matrix: np.ndarray, expansion: int = 2, inflation: float = 2.0, threshold: float = 1e-6, tol=1e-6, max_iter: int = 1000) -> List[List[int]]:
        """
        Cluster a transitory matrix using the Markov chain.
        The clusters are formed by the nodes that are connected to each other.
        
        :param matrix: The transitory matrix.
        :param expansion: The expansion factor for the clusters.
        :param inflation: The inflation factor for the clusters.
        :param threshold: The threshold for the clusters.
        :param tol: The tolerance for convergence.
        :param max_iter: The maximum number of iterations.
        :return: A list of clusters, each cluster is a list of node indices.
        """
        n = matrix.shape[0]
        M = matrix.copy()
        
        for _ in range(max_iter):
            M_old = M.copy()

            # Step 3: Expansion
            M = np.linalg.matrix_power(M, expansion)

            # Step 4: Inflation
            M = np.power(M, inflation)
            
            # Step 5: Prune small values
            M[M < threshold] = 0

            # Step 6: Re-normalize
            column_sums = M.sum(axis=0, keepdims=True)
            column_sums[column_sums == 0] = 1  # Avoid division by zero
            M = M / column_sums

            # Step 7: Check convergence
            if np.allclose(M, M_old, atol=tol):
                break

        # Step 8: Interpret the final matrix
        clusters = []
        seen = set()

        for i in range(n):
            if i in seen:
                continue
            cluster = set(np.where(M[i] > 0)[0])
            if cluster:
                clusters.append(sorted(list(cluster)))
                seen.update(cluster)

        return clusters

    def topic_model(self, tf_matrix: np.ndarray, top_k: int = 3, damping: float = 0.85, max_iter: int = 1000) -> List[List[str]]:
        """
        Performs singel document topic modeling using TopicRank on the tf_matrix.
        Returns a list of indexes for topics within the tf_matrix
        """
        # 1. Create a similarity matrix based on term overlap
        sim_matrix = overlap_sim_matrix(tf_matrix.T)
        sim_matrix /= np.clip(np.sum(sim_matrix, axis=1, keepdims=True), 1e-100, None)
        # 2. Create a transition matrix based on the similarity matrix
        transition_matrix = (1-damping) * (np.ones(sim_matrix.shape) / sim_matrix.shape[0]) + damping * (sim_matrix)
        # 3. Perform power iteration until stready state
        word_scores = markov_chain(transition_matrix, max_iter=max_iter)
        # 4. Return the top_k word indices
        return np.argsort(word_scores)[::-1][:top_k]
    
    def query_document(self, query_nodes: np.ndarray, query_edges: np.ndarray, max_sentences: int) -> List[str]:
        """
        Queries a single document for the most relevant sentences based on the query terms.
        Returns the top_k sentence indices.
        """
        stop_rewards = query_nodes * 1000
        travel_costs = query_edges * 100
        required_number_of_stops = max_sentences
        return max_profit_route(stop_rewards, travel_costs, required_number_of_stops)