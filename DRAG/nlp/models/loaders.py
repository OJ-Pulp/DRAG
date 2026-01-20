import numpy as np
import onnxruntime
from tokenizers import Tokenizer
from typing import List, Tuple

class Reranker:
    def __init__(self, path: str, max_length: int = 512):
        # Load ONNX model
        self.session = onnxruntime.InferenceSession(f"{path}/model.onnx")
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        self.max_length = max_length

    def preprocess(self, query: str, answer: str):
        # Encode the input pair
        encoded = self.tokenizer.encode(f"{query} [SEP] {answer}")
        
        # Truncate to max_length
        input_ids = encoded.ids[:self.max_length]
        attention_mask = (
            encoded.attention_mask[:self.max_length]
            if hasattr(encoded, "attention_mask") else [1] * len(input_ids)
        )

        # Convert to numpy arrays with batch dimension
        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.int64)

        return input_ids, attention_mask

    def rerank(self, pairs: List[Tuple[str, str]]) -> List[float]:
        input_name = self.session.get_inputs()[0].name
        attention_name = self.session.get_inputs()[1].name
        output_name = self.session.get_outputs()[0].name

        scores = []
        for query, answer in pairs:
            input_ids, attention_mask = self.preprocess(query, answer)
            outputs = self.session.run(
                [output_name],
                {
                    input_name: input_ids,
                    attention_name: attention_mask
                }
            )
            # Assuming model returns [batch_size, 1]
            score = float(outputs[0][0][0]) if outputs[0].ndim == 2 else float(outputs[0][0])
            scores.append(score)

        return scores


class SentenceEmbedder:
    def __init__(self, path: str, max_length: int = 512):
        self.session = onnxruntime.InferenceSession(f"{path}/model.onnx")
        self.tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        self.max_length = max_length

    def embed(self, sentence: str) -> np.ndarray:
        # Tokenize the input and truncate if needed
        encoding = self.tokenizer.encode(sentence)
        input_ids = encoding.ids[:self.max_length]
        attention_mask = (
            encoding.attention_mask[:self.max_length]
            if hasattr(encoding, "attention_mask")
            else [1] * len(input_ids)
        )

        # Convert to numpy arrays with batch dimension
        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run inference
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        outputs = self.session.run(None, inputs)
        last_hidden_state = outputs[0]  # [1, seq_len, hidden_size]

        # Mean pooling over non-masked tokens
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        masked = last_hidden_state * mask
        summed = masked.sum(axis=1)
        count = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = summed / count  # [1, hidden_size]

        return embedding.squeeze()  # [hidden_size]
