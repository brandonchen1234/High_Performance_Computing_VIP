"""Week 6-7: word vectors + quantum overlap comparison.

This script provides a reusable local workflow:

1. choose one or more local sentence-vector models
2. create sentence vectors for default or user-provided phrases
3. compute classical cosine similarities
4. compute quantum-style overlaps from a swap-test-style state
5. save charts plus CSV/JSON data artifacts
"""

# Libraries
from __future__ import annotations
import argparse
import json
import math
from itertools import product
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
import time

DEFAULT_SENTENCES = [
    "I have two cats, a black one named Tom and a white one named Jerry.",
    "Tom likes to chase Jerry around the house.",
    "The white house is a big house.",
    "NYU is a university in New York City.",
    "Jerry is a professor at NYU.",
    "Tom is twelve years old.",
    "Jerry is a cat.",
]

TRANSFORMER_MODEL_ALIASES = {
    "jina-v3": "jinaai/jina-embeddings-v3-hf",
    "xlm-roberta": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    "roberta": "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
    "distilbert": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "bert": "sentence-transformers/bert-base-nli-mean-tokens",
}

SPARSE_MODELS = ("tfidf", "count", "hashing")
SUPPORTED_MODELS = SPARSE_MODELS + tuple(TRANSFORMER_MODEL_ALIASES)
DEFAULT_QUANTUM_DIMENSIONS = 32


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalizes an input vector and returns a float64 copy."""

    magnitude = np.linalg.norm(vector)
    if math.isclose(magnitude, 0.0):
        raise ValueError("Cannot normalize a zero vector.")
    return vector / magnitude


def next_power_of_two(size: int) -> int:
    """Returns the smallest power of two that is >= `size`."""

    if size <= 1:
        return 1
    return 1 << math.ceil(math.log2(size))


def pad_to_power_of_two(vector: np.ndarray) -> np.ndarray:
    """Pads `vector` with zeros to a power-of-two length."""

    target = next_power_of_two(len(vector))
    if len(vector) == target:
        return vector
    padded = np.zeros(target, dtype=np.float64)
    padded[: len(vector)] = vector
    return padded


def _select_sentence_transformer_device() -> str | None:
    """Returns the best available local accelerator for dense embeddings."""

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return None


def _build_sparse_sentence_vectors(
    sentences: list[str], model: str
) -> tuple[np.ndarray, list[str]]:
    """Builds deterministic sentence vectors using a selected sparse model."""

    if model == "tfidf":
        vectorizer = TfidfVectorizer(lowercase=True, norm="l2")
        matrix = vectorizer.fit_transform(sentences)
        feature_names = list(vectorizer.get_feature_names_out())
        return np.asarray(matrix.toarray(), dtype=np.float64), feature_names

    if model == "count":
        vectorizer = CountVectorizer(lowercase=True)
        matrix = vectorizer.fit_transform(sentences)
        feature_names = list(vectorizer.get_feature_names_out())
        return np.asarray(matrix.toarray(), dtype=np.float64), feature_names

    vectorizer = HashingVectorizer(n_features=32, alternate_sign=False, norm="l2")
    matrix = vectorizer.transform(sentences)
    feature_names = [f"hash_{index}" for index in range(matrix.shape[1])]
    return np.asarray(matrix.toarray(), dtype=np.float64), feature_names


def _mean_pool_transformer_output(
    model_output: object, attention_mask: object
) -> object:
    """Pools token embeddings while ignoring padding tokens."""

    import torch

    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed_embeddings = torch.sum(token_embeddings * input_mask, dim=1)
    summed_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
    return summed_embeddings / summed_mask


def _build_jina_sentence_vectors(sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    """Builds dense sentence embeddings from Jina Embeddings v3."""

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_name = TRANSFORMER_MODEL_ALIASES["jina-v3"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = _select_sentence_transformer_device()
    if device:
        encoder = encoder.to(device)

    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    if device:
        encoded = encoded.to(device)

    with torch.no_grad():
        model_output = encoder(**encoded)

    embeddings = _mean_pool_transformer_output(model_output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    vectors = embeddings.cpu().numpy().astype(np.float64)
    feature_names = [f"embedding_{index}" for index in range(vectors.shape[1])]
    return vectors, feature_names


def _build_transformer_sentence_vectors(
    sentences: list[str], model: str
) -> tuple[np.ndarray, list[str]]:
    """Builds dense sentence embeddings from a SentenceTransformer alias."""

    if model == "jina-v3":
        return _build_jina_sentence_vectors(sentences)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install transformer dependencies with `uv sync` before using dense models."
        ) from error

    model_name = TRANSFORMER_MODEL_ALIASES[model]
    device = _select_sentence_transformer_device()
    encoder = SentenceTransformer(model_name, device=device)
    embeddings = encoder.encode(sentences, normalize_embeddings=True)
    vectors = np.asarray(embeddings, dtype=np.float64)
    feature_names = [f"embedding_{index}" for index in range(vectors.shape[1])]
    return vectors, feature_names


def build_sentence_vectors(
    sentences: list[str], model: str
) -> tuple[np.ndarray, list[str]]:
    """Builds sentence vectors using sparse or dense model aliases."""

    if model in SPARSE_MODELS:
        return _build_sparse_sentence_vectors(sentences, model)

    if model in TRANSFORMER_MODEL_ALIASES:
        return _build_transformer_sentence_vectors(sentences, model)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model}'. Choose one of: {supported}.")


def reduce_vectors_for_quantum(
    vectors: np.ndarray, target_dimensions: int
) -> tuple[np.ndarray, str]:
    """Reduces vectors before quantum simulation to keep circuits manageable."""

    if vectors.shape[1] <= target_dimensions:
        return vectors, f"none ({vectors.shape[1]} dimensions)"

    components = min(target_dimensions, vectors.shape[0], vectors.shape[1])
    reducer = PCA(n_components=components, random_state=2026)
    reduced = reducer.fit_transform(vectors)
    return np.asarray(reduced, dtype=np.float64), f"PCA to {components} dimensions"


def quantum_overlap_state(
    vec_a: np.ndarray, vec_b: np.ndarray
) -> tuple[float, QuantumCircuit]:
    """Builds overlap circuit and returns exact overlap estimate + circuit."""

    vec_a = normalize(pad_to_power_of_two(vec_a.astype(np.float64)))
    vec_b = normalize(pad_to_power_of_two(vec_b.astype(np.float64)))

    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must have same padded length.")

    dim = int(math.log2(len(vec_a)) + 0.5)
    ancilla = dim
    state = np.concatenate((vec_a, vec_b), dtype=np.float64) / math.sqrt(2.0)
    state = normalize(state)

    circuit = QuantumCircuit(dim + 1)
    circuit.initialize(state.tolist(), circuit.qubits)
    circuit.h(ancilla)

    statevector = Statevector.from_instruction(circuit)
    p0 = float(np.sum(np.abs(statevector.data[: len(statevector.data) // 2]) ** 2))
    overlap = 2.0 * p0 - 1.0

    return overlap, circuit


def estimate_overlap_from_p0(p0: float, shots: int) -> tuple[float, float]:
    """Simulates repeated readout on ancilla and returns estimate + stdev."""

    rng = np.random.default_rng(2026)
    zero_count = int(rng.binomial(shots, p0))
    p0_hat = zero_count / shots
    overlap_hat = 2.0 * p0_hat - 1.0
    overlap_std = 2.0 * math.sqrt(max(p0 * (1.0 - p0), 0.0) / shots)
    return overlap_hat, overlap_std


def pairwise_classical(vectors: np.ndarray) -> np.ndarray:
    """Returns cosine similarity matrix for real-valued vectors."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    return normalized @ normalized.T


def pairwise_quantum(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns exact overlap matrix and circuits for each phrase pair."""

    n = vectors.shape[0]
    overlaps = np.zeros((n, n), dtype=np.float64)
    circuits: np.ndarray = np.empty((n, n), dtype=object)

    for row in range(n):
        for col in range(row, n):
            overlap, circuit = quantum_overlap_state(vectors[row], vectors[col])
            overlaps[row, col] = overlap
            overlaps[col, row] = overlap
            circuits[row, col] = circuit
            circuits[col, row] = circuit

    return overlaps, circuits


def _shortened_label(value: str, width: int = 14) -> str:
    """Shortens a sentence label for compact chart/table rendering."""

    return value[:width]


def _shortened_labels(values: list[str], width: int = 14) -> list[str]:
    """Shortens multiple sentence labels for compact chart/table rendering."""

    return [_shortened_label(value, width=width) for value in values]


def _matrix_color_settings(
    matrix: np.ndarray, *, is_difference: bool
) -> tuple[float, float, str]:
    """Returns display bounds and colormap for a heatmap matrix."""

    matrix_min = float(np.min(matrix))
    matrix_max = float(np.max(matrix))

    if is_difference:
        upper = matrix_max if not math.isclose(matrix_max, 0.0) else 1.0
        return 0.0, upper, "magma"

    if matrix_min >= 0.0:
        return 0.0, 1.0, "viridis"

    bound = max(abs(matrix_min), abs(matrix_max), 1.0)
    return -bound, bound, "RdBu_r"


def _configure_matrix_axes(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    *,
    is_difference: bool = False,
) -> None:
    """Configures heatmap ticks and adaptive color scaling."""

    vmin, vmax, cmap = _matrix_color_settings(matrix, is_difference=is_difference)
    axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(f"{title}\nscale {vmin:.3g} to {vmax:.3g}")
    ticks = list(range(len(labels)))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)


def format_table(matrix: np.ndarray, labels: list[str]) -> str:
    """Formats a small Markdown table string for quick sharing."""

    shortened_labels = _shortened_labels(labels)
    header = "| idx | " + " | ".join(shortened_labels) + " |\n"
    separator = "| --- " + "|---" * len(labels) + "|\n"
    rows: list[str] = []
    for row_index, row in enumerate(matrix):
        values = [f"{value:.3f}" for value in row]
        rows.append(f"| {row_index} | " + " | ".join(values) + " |")
    return header + separator + "\n".join(rows)


def _artifact_name(model: str, suffix: str) -> str:
    """Returns a model-scoped artifact filename."""

    return f"{model}_{suffix}"


def save_heatmaps(
    sentences: list[str],
    classical: np.ndarray,
    quantum: np.ndarray,
    out_dir: Path,
    *,
    model: str,
) -> Path:
    """Saves classical, quantum, and absolute-difference heatmaps."""

    diff = np.abs(classical - quantum)
    labels = _shortened_labels(sentences)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, matrix, title, is_difference in (
        (axes[0], classical, "classical_cosine", False),
        (axes[1], quantum, "quantum_overlap", False),
        (axes[2], diff, "absolute_difference", True),
    ):
        _configure_matrix_axes(axis, matrix, labels, title, is_difference=is_difference)
        image = axis.images[0]
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure_path = out_dir / _artifact_name(model, "similarity_matrices.png")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def save_anchor_bar_chart(
    sentences: list[str],
    classical: np.ndarray,
    quantum: np.ndarray,
    anchor: int,
    out_dir: Path,
    *,
    model: str,
) -> Path:
    """Saves a bar chart comparing anchor row between classical and quantum."""

    indices = np.arange(len(sentences))
    width = 0.35
    labels = _shortened_labels(sentences)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        indices - width / 2, classical[anchor], width=width, label="Classical cosine"
    )
    axis.bar(indices + width / 2, quantum[anchor], width=width, label="Quantum overlap")

    axis.set_title(f"Sentence {anchor} against all sentences ({model})")
    axis.set_xticks(indices)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Similarity")
    axis.legend()
    fig.tight_layout()

    path = out_dir / _artifact_name(model, "anchor_overlap_bars.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_circuit_diagram(circuit: QuantumCircuit, out_dir: Path, *, model: str) -> Path:
    """Saves one circuit diagram that can be shown in the presentation."""

    matplotlib.use("Agg")
    figure = circuit.draw(output="mpl")
    path = out_dir / _artifact_name(model, "swap_circuit.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_matrix_outputs(
    sentences: list[str],
    model: str,
    classical: np.ndarray,
    quantum: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Saves CSV and JSON data used to make the heatmaps."""

    diff = np.abs(classical - quantum)
    csv_path = out_dir / _artifact_name(model, "matrix.csv")
    json_path = out_dir / _artifact_name(model, "matrix.json")

    csv_rows = ["sentence_i,sentence_j,classical,quantum,absolute_difference"]
    for row_index, col_index in product(range(len(sentences)), repeat=2):
        csv_rows.append(
            f"{row_index},{col_index},{classical[row_index, col_index]:.8f},{quantum[row_index, col_index]:.8f},{diff[row_index, col_index]:.8f}"
        )
    csv_path.write_text("\n".join(csv_rows))

    json_path.write_text(
        json.dumps(
            {
                "model": model,
                "phrases": sentences,
                "classical": classical.tolist(),
                "quantum": quantum.tolist(),
                "absolute_difference": diff.tolist(),
            },
            indent=2,
        )
    )

    return csv_path, json_path


def _load_phrases_from_file(path: Path) -> list[str]:
    """Loads phrases from a JSON array or newline-delimited text file."""

    raw = path.read_text()
    if path.suffix.lower() == ".json":
        phrases = json.loads(raw)
        if not isinstance(phrases, list) or not all(
            isinstance(value, str) for value in phrases
        ):
            raise ValueError("Phrase JSON file must contain an array of strings.")
        return phrases

    return [line.strip() for line in raw.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    """Parses CLI options for reusable phrase/model experiments."""

    parser = argparse.ArgumentParser(
        description="Generate sentence similarity matrices with local vector models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfidf"],
        choices=SUPPORTED_MODELS,
        help="One or more local vector models to run.",
    )
    parser.add_argument(
        "--phrases",
        nargs="+",
        help="Phrases to compare. Quote each phrase separately.",
    )
    parser.add_argument(
        "--phrases-file",
        type=Path,
        help="Path to a JSON array or newline-delimited text file of phrases.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts",
        help="Directory where generated artifacts are written. Defaults to ./artifacts relative to the project root.",
    )
    parser.add_argument(
        "--quantum-dimensions",
        type=int,
        default=DEFAULT_QUANTUM_DIMENSIONS,
        help="Maximum dimensions used by the quantum overlap circuit after optional PCA reduction.",
    )
    return parser.parse_args()


def resolve_sentences(args: argparse.Namespace) -> list[str]:
    """Returns phrases from CLI, file, or built-in defaults."""

    if args.phrases and args.phrases_file:
        raise ValueError("Use either --phrases or --phrases-file, not both.")

    if args.phrases:
        return args.phrases

    if args.phrases_file:
        return _load_phrases_from_file(args.phrases_file)

    return DEFAULT_SENTENCES


def run_model(
    sentences: list[str],
    model: str,
    artifacts_dir: Path,
    *,
    quantum_dimensions: int,
) -> None:
    """Runs one vector model and writes all artifacts."""

    if len(sentences) < 2:
        raise ValueError("At least two phrases are required.")

    if len(sentences) > 10:
        pair_count = len(sentences) * (len(sentences) + 1) // 2
        print(
            f"Warning: {len(sentences)} phrases require {pair_count} quantum overlap computations."
        )

    vectors, feature_names = build_sentence_vectors(sentences, model)
    classical = pairwise_classical(vectors)
    quantum_vectors, reduction_note = reduce_vectors_for_quantum(
        vectors, quantum_dimensions
    )
    quantum = pairwise_classical(quantum_vectors)
    _, circuits = pairwise_quantum(quantum_vectors)
    anchor_overlap = float(quantum[0, 1])

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    heatmaps_path = save_heatmaps(
        sentences, classical, quantum, artifacts_dir, model=model
    )
    bars_path = save_anchor_bar_chart(
        sentences, classical, quantum, anchor=0, out_dir=artifacts_dir, model=model
    )
    circuit_path = save_circuit_diagram(circuits[0, 1], artifacts_dir, model=model)
    csv_path, json_path = save_matrix_outputs(
        sentences, model, classical, quantum, artifacts_dir
    )
    anchor_p0 = (anchor_overlap + 1.0) / 2.0
    sampled_overlap, sample_std = estimate_overlap_from_p0(p0=anchor_p0, shots=2000)

    diff_max = float(np.max(np.abs(classical - quantum)))
    cosine_txt = format_table(classical, sentences)
    report_path = artifacts_dir / _artifact_name(model, "summary.txt")
    report_path.write_text(
        "Week 6-7 artifact summary\n\n"
        f"Embedding model: {model}\n"
        f"Number of sentences: {len(sentences)}\n"
        f"Vocabulary/features size: {len(feature_names)}\n"
        f"Quantum vector reduction: {reduction_note}\n"
        f"max |classical - quantum|: {diff_max:.6f}\n"
        f"Demo sampling check (pair 0-1, shots=2000): overlap={sampled_overlap:.6f}, std={sample_std:.6f}\n"
        f"Demo sample p0 check (pair 0-1): anchor_p0={anchor_p0:.6f}\n\n"
        "Classical cosine matrix (rounded):\n"
        f"{cosine_txt}\n"
    )

    print(f"Week 6-7 workflow complete for model: {model}")
    print(f"Artifacts: {heatmaps_path}")
    print(f"          {bars_path}")
    print(f"          {circuit_path}")
    print(f"          {csv_path}")
    print(f"          {json_path}")
    print(f"          {report_path}")
    print(f"max |classical - quantum| = {diff_max:.6f}")
    print(f"anchor pair overlap: {anchor_overlap:.6f}")


def main() -> None:
    """Runs week 6-7 experiment and writes all showcase artifacts."""

    start_time = time.perf_counter()

    matplotlib.use("Agg")
    args = parse_args()
    sentences = resolve_sentences(args)

    for model in args.models:
        run_model(
            sentences=sentences,
            model=model,
            artifacts_dir=args.out_dir,
            quantum_dimensions=args.quantum_dimensions,
        )

    end_time = time.perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
