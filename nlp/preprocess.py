import numpy as np
import json
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List
import string
  
class preprocessor:
    def __init__(self, stop_words_path) -> None:
        self.stemmer = SnowballStemmer('english')
        self.stop_words_path = stop_words_path

    def preprocess_query(self, text: str, stem: bool = True) -> List[str]:
        """
        Extracts the query terms from the text.
        """
        text = self.chunk(text)
        text = self.tokenize(text)
        stop_words = self.stop_words()
        terms = []
        for chunk in text:
            for token in chunk:
                token = token.lower()
                if not token in stop_words and not token in string.punctuation and token.strip() != '':
                    terms.append(self.stem(token)) if stem else terms.append(token)
        
        return terms
    
    def tf(self, text: List[str], include_terms: bool = False, keys: bool = False) -> np.ndarray:
        text = self.tokenize(text)
        stop_words = self.stop_words()
        num_chunks = len(text)
        term_dict = {}
        if include_terms:
            original_tokens = []

        for i, chunk in enumerate(text):
            for token in chunk:
                token = token.lower()
                if not token in stop_words and not token in string.punctuation and token.strip() != '' and self._word_score(token) > 0:
                    base_token = self.stem(token)
                    if base_token not in term_dict:
                        term_dict[base_token] = np.zeros(num_chunks)
                        if include_terms:
                            original_tokens.append(token)
                    term_dict[base_token][i] += 1

        tf_matrix = np.vstack(list(term_dict.values()))
        if include_terms:
            if keys:
                return tf_matrix, list(term_dict.keys()), original_tokens
            else:
                return tf_matrix, original_tokens
        else:
            if keys:
                return tf_matrix, list(term_dict.keys())
            else:
                return tf_matrix

    
    def idf(self, tf: np.ndarray) -> np.ndarray:
        """
        Computes the IDF score for each term in the document.
        """
        num_sent = tf.shape[1]
        idf = np.log(num_sent / (np.sum(tf > 0, axis=1) + 1)).reshape(-1, 1)
        return idf
    
    def tfidf(self, tf: np.ndarray) -> np.ndarray:
        """
        Computes the TF-IDF score for each term in the document.
        """
        idf = self.idf(tf)
        tfidf_matrix = tf * idf
        return tfidf_matrix
    
    def bm25(self, tf: np.ndarray, sentence_lengths: List[int], k1: float = 1.2, b: float = 0.85) -> np.ndarray:
        """
        Computes the BM25 score for each term in the document.
        """
        sentence_lengths = np.array(sentence_lengths)
        idf = self.idf(tf)
        avgdl = np.mean(sentence_lengths)
        bm25_matrix = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * sentence_lengths / avgdl)) * idf
        bm25_matrix = np.clip(bm25_matrix, 0, None)  # Ensure non-negative values
        return bm25_matrix
    
    def stop_words(self) -> List[str]:
        """
        Returns a list of stop words.
        """
        stop_words = json.load(open(self.stop_words_path, "r", encoding="utf-8"))
        return stop_words
    
    def tokenize(self, text: List[str])-> List[str]:
        return [word_tokenize(text_chunk) for text_chunk in text]
    
    def stem(self, token):
        return self.stemmer.stem(token)
    
    def chunk(self, text: str) -> List[str]:
        """
        Splits the text into chunks based on sentence boundaries.
        """
        return sent_tokenize(text)
        
    def _word_score(self, word: str):
        """
        Computes the score of a word based on its length and the number of letters.
        """
        word_len = len(word)
        if word_len > 0 and word_len < 15:
            word_score = len([char for char in word if char in string.ascii_letters]) / word_len
        else:
            word_score = 0
        return word_score
    
    def _sentence_score(self, sentence: List[str]):
        """
        Computes the score of a sentence based on the number of letters and non-letters.
        """
        sentence_len = len(sentence)
        if sentence_len > 4 and sentence_len < 60:
            sentence_score = np.sum([self._word_score(word) for word in sentence]) / sentence_len
        else: 
            sentence_score = 0
        return sentence_score
    
    def _clean_text(self, text: str) -> str:
        """
        Removes extra spaces, non-ascii characters or numbers, and special characters from text e.g. (\n, \t, ...) from the text.
        """
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        text = ''.join([char for char in text if char in string.printable])

        return text
        
    def clean(self, sentences: List[str], threshold: float = 0.60) -> List[str]:
        """
        Cleans the text by removing non-letter characters and applying a scoring system to sentences.
        """
        # Remove extra spaces and special characters
        sentences = [self._clean_text(sentence) for sentence in sentences]

        # Tokenize sentences
        tokenized_sentences = self.tokenize(sentences)

        # Compute scores for each sentence
        sentence_scores = np.array([self._sentence_score(tokenized_sentences) for tokenized_sentences in tokenized_sentences])

        # Create a mask to filter out low-scoring sentences
        mask = sentence_scores > threshold

        return [sentence for sentence, keep in zip(sentences, mask) if keep]