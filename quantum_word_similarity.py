"""
Quantum Word Similarity
Converted and patched from quantum-word-similarity.ipynb
"""

import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import time


def pad_to_power_of_two(vector: np.ndarray) -> np.ndarray:
    """Pads a vector with zeros to the next power of two."""
    target_len = 1 if len(vector) == 0 else 2 ** (len(vector) - 1).bit_length()
    if len(vector) == target_len:
        return vector
    padded = np.zeros(target_len, dtype=np.float64)
    padded[:len(vector)] = vector
    return padded


class SentenceTransformer:
    def __init__(self, model_name: str, **kwargs):
        # The **kwargs allows us to pass trust_remote_code=True to the Hugging Face backend
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)

    def encode(self, sentences: list[str]) -> np.ndarray:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Mean pooling
        embeddings = model_output.last_hidden_state.mean(dim=1).numpy()
        return embeddings


def quantum_inner_prod(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Creates a quantum circuit to calculate the inner product between two vectors."""

    # 1. Pad to power of two (Critical fix: 768d BERT vectors require padding to 1024)
    v1_padded = pad_to_power_of_two(vec1)
    v2_padded = pad_to_power_of_two(vec2)

    # 2. Normalize
    v1_norm = v1_padded / np.linalg.norm(v1_padded)
    v2_norm = v2_padded / np.linalg.norm(v2_padded)

    # 3. Concatenate and scale for the quantum state
    N = len(v1_norm)
    nqubits = math.ceil(np.log2(N))

    vec = np.concatenate((v1_norm, v2_norm)).astype(np.float64)
    vec = vec / np.linalg.norm(vec)

    # 4. Build circuit
    circ = QuantumCircuit(nqubits + 1)
    circ.initialize(vec.tolist(), range(nqubits + 1))
    circ.h(nqubits)

    # 5. Measure via exact Statevector (Qiskit 1.0+ compatible)
    state = Statevector(circ)
    amplitudes = np.real(state.data)

    # Sum the squared amplitudes where the ancilla qubit is 0
    p0 = np.sum(amplitudes[:N] ** 2)

    return float(2 * p0 - 1)


def main():

    start_time = time.perf_counter()

    test_sentences = [
        "I have two cats, a black one named Tom and a white one named Jerry.",
        "我有兩隻貓，一隻黑色嘅叫 Tom ，一隻白色嘅叫 Jerry 。"
    ]

    print("Loading language model...")
    # We can choose different models:
    # "distilbert-base-uncased"
    # "bert-base-uncased"
    # "roberta-base"
    # "xlm-roberta-base"
    # sentence-transformers/all-MiniLM-L6-v2
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # sentence-transformers/all-mpnet-base-v2
    # sentence-transformers/all-distilroberta-v1
    language_model = "distilbert-base-uncased"
    encoder = SentenceTransformer(language_model)

    print(f"Encoding sentences using {language_model}...")
    embeddings = encoder.encode(test_sentences)

    # Classical Cosine Similarity
    v1, v2 = embeddings[0], embeddings[1]
    classical_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Quantum Inner Product
    print("Running quantum interference circuit...")
    quantum_sim = quantum_inner_prod(v1, v2)

    print("\n--- Results ---")
    print(f"Sentence 1: {test_sentences[0]}")
    print(f"Sentence 2: {test_sentences[1]}")
    print(f"Classical Cosine Similarity: {classical_sim:.6f}")
    print(f"Quantum Circuit Similarity:  {quantum_sim:.6f}")
    print(f"Absolute Difference:         {abs(classical_sim - quantum_sim):.6e}")

    end_time = time.perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()