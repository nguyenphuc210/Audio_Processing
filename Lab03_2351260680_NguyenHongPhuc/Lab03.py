"""Lab 3: isolated-word recognition with left-to-right Gaussian HMMs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


SAMPLE_RATE = 16_000
N_MFCC = 13
LABELS = ("mo", "dong", "len", "xuong", "dung")
LM_WEIGHTS = (0.0, 0.1, 0.3, 1.0)
EPSILON = 1e-8
LABEL_ALIASES = {
    "mo": "mo",
    "dong": "dong",
    "len": "len",
    "xuong": "xuong",
    "dung": "dung",
}
M4A_FILENAME = re.compile(r"^(mo|dong|len|xuong|dung)[_-]?(\d+)(?:[_-].*)?$", re.IGNORECASE)


def normalize_label(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return LABEL_ALIASES.get(normalized.lower())


def infer_recording_label(path: Path) -> tuple[str, str] | None:
    for parent in reversed(path.parent.parts):
        label = normalize_label(parent)
        if label:
            return label, path.stem
    match = M4A_FILENAME.fullmatch(path.stem)
    if match:
        return match.group(1).lower(), f"{match.group(1).lower()}_{int(match.group(2)):02d}"
    return None


def convert_m4a_dataset(data_dir: Path) -> int:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg and reopen the terminal, "
            "or place pre-converted WAVs in the dataset folders."
        )

    jobs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    occupied_outputs: list[Path] = []
    planned_outputs: set[Path] = set()
    existing_outputs: list[Path] = []
    roots = [data_dir / "train", data_dir / "test", data_dir / "robustness"]

    for source_root in roots:
        if not source_root.is_dir():
            continue
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or source.suffix.lower() != ".m4a":
                continue
            inferred = infer_recording_label(source)
            if inferred is None:
                missing_labels.append(source)
                continue
            label, output_stem = inferred
            if source_root.name == "robustness":
                relative_parent = source.parent.relative_to(source_root)
                label_parent = next(
                    (part for part in reversed(relative_parent.parts) if normalize_label(part)),
                    None,
                )
                condition_parts = (
                    relative_parent.parts[: relative_parent.parts.index(label_parent)]
                    if label_parent is not None
                    else relative_parent.parts
                )
                output_dir = source_root.joinpath(*condition_parts, label)
            else:
                output_dir = source_root / label
            destination = output_dir / f"{output_stem}.wav"
            if destination.exists():
                try:
                    audio_info = sf.info(destination)
                except (OSError, RuntimeError) as error:
                    raise ValueError(f"Existing WAV output is unreadable: {destination}") from error
                if (
                    audio_info.samplerate != SAMPLE_RATE
                    or audio_info.channels != 1
                    or audio_info.subtype != "PCM_16"
                ):
                    occupied_outputs.append(destination)
                else:
                    existing_outputs.append(destination)
            if destination in planned_outputs:
                occupied_outputs.append(destination)
            planned_outputs.add(destination)
            jobs.append((source, destination))

    if missing_labels:
        files = ", ".join(str(path) for path in missing_labels)
        raise ValueError(
            "Could not infer a class label for these M4A files. Put each file under a "
            f"label folder or name it like mo1.m4a/dong_02.m4a: {files}"
        )
    if occupied_outputs:
        files = ", ".join(str(path) for path in occupied_outputs)
        raise FileExistsError(
            "M4A conversion would overwrite or duplicate WAV destinations. "
            f"Move or remove the conflicting WAV files first: {files}"
        )

    converted: list[Path] = []
    try:
        for source, destination in jobs:
            if destination in existing_outputs:
                print(f"Using existing compatible WAV: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            converted.append(destination)
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg could not convert '{source}': {completed.stderr.strip()}"
                )
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg did not produce a valid WAV file for '{source}'.")
            print(f"Converted {source.name} -> {destination}")
    except (OSError, RuntimeError):
        for destination in converted:
            destination.unlink(missing_ok=True)
        raise
    print(
        f"Converted {len(jobs) - len(existing_outputs)} M4A file(s); "
        f"reused {len(existing_outputs)} compatible WAV file(s)."
    )
    return len(jobs) - len(existing_outputs)


def list_wavs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.name.lower(),
    )


def discover_labels(data_dir: Path) -> list[str]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_labels = {path.name for path in train_dir.iterdir() if path.is_dir()} if train_dir.is_dir() else set()
    test_labels = {path.name for path in test_dir.iterdir() if path.is_dir()} if test_dir.is_dir() else set()
    labels = sorted(train_labels | test_labels)
    if not train_labels or not test_labels:
        raise FileNotFoundError(
            f"Expected class folders under '{train_dir}' and '{test_dir}'. "
            "See README.md for the required dataset layout."
        )
    if train_labels != test_labels:
        raise ValueError(
            f"Train/test labels differ. Train-only: {sorted(train_labels - test_labels)}; "
            f"test-only: {sorted(test_labels - train_labels)}"
        )
    return labels


def extract_features(path: Path, feature_set: str = "dynamic") -> np.ndarray:
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Audio is empty or contains non-finite samples: {path}")

    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    if waveform.size == 0:
        raise ValueError(f"Audio contains no usable signal after silence trimming: {path}")
    waveform = librosa.effects.preemphasis(waveform, coef=0.97)

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window="hamming",
        n_mels=26,
        fmin=20,
        fmax=SAMPLE_RATE / 2,
    )
    if feature_set == "static":
        features = mfcc.T
    elif feature_set == "dynamic":
        frame_count = mfcc.shape[1]
        if frame_count >= 3:
            width = min(9, frame_count if frame_count % 2 else frame_count - 1)
            delta = librosa.feature.delta(mfcc, order=1, width=width)
            delta2 = librosa.feature.delta(mfcc, order=2, width=width)
        else:
            delta = np.zeros_like(mfcc)
            delta2 = np.zeros_like(mfcc)
        features = np.vstack((mfcc, delta, delta2)).T
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / np.maximum(std, EPSILON)
    if not np.isfinite(features).all():
        raise ValueError(f"Feature extraction produced NaN or infinity: {path}")
    return features.astype(np.float64, copy=False)


def make_left_to_right_hmm(
    n_states: int,
    covariance_type: str = "diag",
    n_iter: int = 80,
    random_state: int = 0,
) -> GaussianHMM:
    if n_states < 2:
        raise ValueError("A left-to-right HMM must have at least two states.")
    if covariance_type not in {"diag", "full"}:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,
        init_params="mc",
        params="mc",
        random_state=random_state,
        implementation="log",
    )
    start = np.zeros(n_states)
    start[0] = 1.0
    transitions = np.zeros((n_states, n_states))
    for state in range(n_states - 1):
        transitions[state, state] = 0.7
        transitions[state, state + 1] = 0.3
    transitions[-1, -1] = 1.0
    model.startprob_ = start
    model.transmat_ = transitions
    return model


def read_feature_sets(
    files_by_label: dict[str, list[Path]], feature_set: str
) -> dict[str, list[np.ndarray]]:
    return {
        label: [extract_features(path, feature_set) for path in paths]
        for label, paths in files_by_label.items()
    }


def train_models(
    train_features: dict[str, list[np.ndarray]],
    n_states: int = 5,
    covariance_type: str = "diag",
) -> tuple[dict[str, GaussianHMM], float]:
    models: dict[str, GaussianHMM] = {}
    started = time.perf_counter()
    for label, sequences in train_features.items():
        if not sequences:
            raise ValueError(f"No training utterances found for class '{label}'.")
        lengths = [len(sequence) for sequence in sequences]
        observations = np.vstack(sequences)
        model = make_left_to_right_hmm(n_states, covariance_type)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(observations, lengths)
        if not np.array_equal(model.startprob_, np.eye(1, n_states, 0)[0]):
            raise RuntimeError(f"Training changed the left-to-right start probabilities for '{label}'.")
        if not np.allclose(model.transmat_, _left_to_right_matrix(n_states)):
            raise RuntimeError(f"Training changed the fixed left-to-right topology for '{label}'.")
        if caught:
            warning_text = "; ".join(str(item.message) for item in caught)
            print(f"WARNING [{label}, {covariance_type}, {n_states} states]: {warning_text}")
        models[label] = model
        print(
            f"Trained {label}: {len(sequences)} utterances, {len(observations)} frames, "
            f"converged={model.monitor_.converged}, iterations={model.monitor_.iter}"
        )
        print(f"  transmat_:\n{np.round(model.transmat_, 3)}")
    return models, time.perf_counter() - started


def _left_to_right_matrix(n_states: int) -> np.ndarray:
    transitions = np.zeros((n_states, n_states))
    for state in range(n_states - 1):
        transitions[state, state] = 0.7
        transitions[state, state + 1] = 0.3
    transitions[-1, -1] = 1.0
    return transitions


def score_models(features: np.ndarray, models: dict[str, GaussianHMM]) -> dict[str, float]:
    return {label: float(model.score(features)) for label, model in models.items()}


def evaluate_hmms(
    models: dict[str, GaussianHMM],
    test_features: dict[str, list[tuple[Path, np.ndarray]]],
) -> dict[str, Any]:
    labels = list(models)
    truth: list[str] = []
    predictions: list[str] = []
    rows: list[dict[str, Any]] = []
    elapsed = 0.0

    for actual, utterances in test_features.items():
        for path, features in utterances:
            started = time.perf_counter()
            scores = score_models(features, models)
            elapsed += time.perf_counter() - started
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            prediction = ranked[0][0]
            truth.append(actual)
            predictions.append(prediction)
            rows.append(
                {
                    "file": str(path),
                    "true_label": actual,
                    "prediction": prediction,
                    "top1_minus_top2": ranked[0][1] - ranked[1][1] if len(ranked) > 1 else math.nan,
                    **{f"score_{label}": scores[label] for label in labels},
                }
            )

    return {
        "labels": labels,
        "truth": truth,
        "predictions": predictions,
        "rows": rows,
        "accuracy": float(accuracy_score(truth, predictions)) if truth else math.nan,
        "confusion": confusion_matrix(truth, predictions, labels=labels),
        "seconds_per_trial": elapsed / len(truth) if truth else math.nan,
    }


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    cumulative_cost, path = librosa.sequence.dtw(
        X=left.T,
        Y=right.T,
        metric="euclidean",
        backtrack=True,
    )
    return float(cumulative_cost[-1, -1] / max(len(path), 1))


def evaluate_dtw(
    train_features: dict[str, list[np.ndarray]],
    test_features: dict[str, list[tuple[Path, np.ndarray]]],
) -> dict[str, Any]:
    labels = list(train_features)
    truth: list[str] = []
    predictions: list[str] = []
    elapsed = 0.0
    for actual, utterances in test_features.items():
        for _, features in utterances:
            started = time.perf_counter()
            distances = {
                label: min(dtw_distance(features, template) for template in templates)
                for label, templates in train_features.items()
            }
            elapsed += time.perf_counter() - started
            truth.append(actual)
            predictions.append(min(distances, key=distances.get))
    return {
        "labels": labels,
        "truth": truth,
        "predictions": predictions,
        "accuracy": float(accuracy_score(truth, predictions)) if truth else math.nan,
        "confusion": confusion_matrix(truth, predictions, labels=labels),
        "seconds_per_trial": elapsed / len(truth) if truth else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(result: dict[str, Any], path: Path, title: str) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion"],
        display_labels=result["labels"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160)
    plt.close(display.figure_)


def save_feature_heatmaps(
    train_files: dict[str, list[Path]], output_dir: Path
) -> None:
    examples: list[tuple[str, Path, np.ndarray]] = []
    for label, files in train_files.items():
        if files and len(examples) < 2:
            examples.append((label, files[0], extract_features(files[0], "dynamic")))
    if len(examples) < 2:
        raise ValueError("At least two classes with a training recording are required for feature heatmaps.")
    for label, path, features in examples:
        figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for axis, block, title in zip(
            axes,
            (features[:, :13], features[:, 13:26], features[:, 26:39]),
            ("MFCC", "Delta MFCC", "Delta-Delta MFCC"),
        ):
            image = axis.imshow(block.T, aspect="auto", origin="lower", interpolation="nearest")
            axis.set_ylabel(title)
            figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        axes[-1].set_xlabel("Frame")
        figure.suptitle(f"{label}: {path.name}")
        figure.tight_layout()
        figure.savefig(output_dir / f"features_{label}.png", dpi=160)
        plt.close(figure)


def save_viterbi_plots(
    models: dict[str, GaussianHMM],
    test_features: dict[str, list[tuple[Path, np.ndarray]]],
    output_dir: Path,
) -> None:
    examples = [(label, path, features) for label, items in test_features.items() for path, features in items]
    if len(examples) < 2:
        raise ValueError("At least two test utterances are required to create two Viterbi plots.")
    for index, (actual, path, features) in enumerate(examples[:2], start=1):
        candidate = models[actual]
        log_probability, states = candidate.decode(features, algorithm="viterbi")
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.step(np.arange(len(states)), states, where="post")
        axis.set(xlabel="Frame", ylabel="HMM state", yticks=range(candidate.n_components))
        axis.set_title(f"Viterbi: actual={actual}, model={actual}, logP={log_probability:.2f}")
        figure.tight_layout()
        figure.savefig(output_dir / f"viterbi_{index}_{path.stem}.png", dpi=160)
        plt.close(figure)


def _train_test_feature_cache(
    train_files: dict[str, list[Path]],
    test_files: dict[str, list[Path]],
    feature_set: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[tuple[Path, np.ndarray]]]]:
    train = read_feature_sets(train_files, feature_set)
    test = {
        label: [(path, extract_features(path, feature_set)) for path in paths]
        for label, paths in test_files.items()
    }
    return train, test


def _metric_row(
    name: str,
    result: dict[str, Any],
    train_seconds: float,
    n_states: int | str,
    feature_set: str,
    covariance_type: str,
) -> dict[str, Any]:
    return {
        "experiment": name,
        "n_states": n_states,
        "feature_set": feature_set,
        "covariance_type": covariance_type,
        "accuracy": result["accuracy"],
        "train_seconds": train_seconds,
        "test_seconds_per_trial": result["seconds_per_trial"],
        "n_test": len(result["truth"]),
    }


def _top_confusion_pair(result: dict[str, Any]) -> str:
    counts = result["confusion"].copy()
    np.fill_diagonal(counts, 0)
    if counts.size == 0 or counts.max() == 0:
        return "none"
    actual_index, predicted_index = np.unravel_index(np.argmax(counts), counts.shape)
    return f"{result['labels'][actual_index]}->{result['labels'][predicted_index]} ({counts[actual_index, predicted_index]})"


def parse_lexicon(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    if not path.is_file():
        print(f"INFO: No lexicon at {path}; add pronunciations from your instructor before the lexicon extension.")
        return entries
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Invalid lexicon line {line_number}: expected WORD followed by at least one phone.")
        word, phones = fields[0], fields[1:]
        if word in entries:
            raise ValueError(f"Duplicate lexicon entry '{word}' on line {line_number}.")
        entries[word] = phones
    return entries


def build_bigram_counts(corpus_path: Path) -> tuple[Counter[str], Counter[tuple[str, str]], set[str]]:
    unigram: Counter[str] = Counter()
    bigram: Counter[tuple[str, str]] = Counter()
    vocabulary: set[str] = set()
    if not corpus_path.is_file():
        return unigram, bigram, vocabulary
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        vocabulary.update(words)
        unigram.update(words)
        bigram.update(zip(words, words[1:]))
    return unigram, bigram, vocabulary


def bigram_probability(
    previous: str,
    current: str,
    unigram: Counter[str],
    bigram: Counter[tuple[str, str]],
    vocabulary_size: int,
    smoothing: float = 0.5,
) -> float:
    if vocabulary_size <= 0 or smoothing <= 0:
        raise ValueError("Bigram smoothing and vocabulary size must be positive.")
    denominator = unigram[previous] + smoothing * vocabulary_size
    return (bigram[(previous, current)] + smoothing) / denominator


def run_bigram_reranking(
    context_path: Path,
    data_dir: Path,
    test_files: dict[str, list[Path]],
    test_features: dict[str, list[tuple[Path, np.ndarray]]],
    models: dict[str, GaussianHMM],
    corpus_path: Path,
    output_dir: Path,
) -> None:
    if not context_path.is_file():
        print(
            f"INFO: Bigram re-ranking needs '{context_path}' with columns wav,previous_word; "
            "add contexts for the isolated test recordings to run E6."
        )
        return
    unigram, bigram, vocabulary = build_bigram_counts(corpus_path)
    if not vocabulary:
        print(f"INFO: Bigram corpus '{corpus_path}' is missing or empty; skipping LM re-ranking.")
        return

    path_to_feature = {
        path.resolve(): features
        for entries in test_features.values()
        for path, features in entries
    }
    actual_by_path = {
        path.resolve(): label
        for label, paths in test_files.items()
        for path in paths
    }
    contexts: dict[Path, str] = {}
    with context_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("wav") or not row.get("previous_word"):
                raise ValueError("Each contexts.csv row must contain wav and previous_word.")
            raw_path = Path(row["wav"])
            resolved = (data_dir / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
            if resolved not in path_to_feature:
                raise ValueError(f"Context file references a WAV that is not in the test set: {row['wav']}")
            contexts[resolved] = row["previous_word"]

    rows: list[dict[str, Any]] = []
    for path, previous in contexts.items():
        features = path_to_feature[path]
        acoustic = score_models(features, models)
        frame_count = len(features)
        for weight in LM_WEIGHTS:
            combined = {
                label: acoustic[label] / frame_count
                + weight
                * math.log(
                    bigram_probability(previous, label, unigram, bigram, len(vocabulary))
                )
                for label in models
            }
            rows.append(
                {
                    "wav": str(path),
                    "previous_word": previous,
                    "true_label": actual_by_path[path],
                    "lm_weight": weight,
                    "prediction": max(combined, key=combined.get),
                    **{f"combined_{label}": combined[label] for label in models},
                }
            )
    write_csv(output_dir / "bigram_reranking.csv", rows)
    for weight in LM_WEIGHTS:
        subset = [row for row in rows if row["lm_weight"] == weight]
        accuracy = accuracy_score(
            [row["true_label"] for row in subset],
            [row["prediction"] for row in subset],
        )
        print(f"Bigram re-ranking lambda={weight}: accuracy={accuracy:.3f} ({len(subset)} trials)")


def run_experiments(
    data_dir: Path,
    output_dir: Path,
    labels: list[str],
    train_files: dict[str, list[Path]],
    test_files: dict[str, list[Path]],
    baseline_models: dict[str, GaussianHMM],
    baseline_train_features: dict[str, list[np.ndarray]],
    baseline_test_features: dict[str, list[tuple[Path, np.ndarray]]],
    baseline_result: dict[str, Any],
    baseline_train_seconds: float,
    baseline_dtw: dict[str, Any],
    baseline_dtw_train_seconds: float,
) -> None:
    metrics = [
        _metric_row("hmm_5_dynamic_diag", baseline_result, baseline_train_seconds, 5, "dynamic", "diag"),
        _metric_row("dtw_13_static", baseline_dtw, baseline_dtw_train_seconds, "-", "static", "-"),
    ]

    for states in (3, 5, 7):
        if states == 5:
            seconds = baseline_train_seconds
            result = baseline_result
        else:
            models, seconds = train_models(baseline_train_features, states, "diag")
            result = evaluate_hmms(models, baseline_test_features)
        metrics.append(_metric_row(f"states_{states}", result, seconds, states, "dynamic", "diag"))

    for feature_set in ("static", "dynamic"):
        if feature_set == "dynamic":
            train, test = baseline_train_features, baseline_test_features
            models = baseline_models
            seconds = baseline_train_seconds
            result = baseline_result
        else:
            train, test = _train_test_feature_cache(train_files, test_files, feature_set)
            models, seconds = train_models(train, 5, "diag")
            result = evaluate_hmms(models, test)
        metrics.append(_metric_row(f"features_{feature_set}", result, seconds, 5, feature_set, "diag"))

    for covariance in ("diag", "full"):
        if covariance == "diag":
            models = baseline_models
            seconds = baseline_train_seconds
            result = baseline_result
            metrics.append(
                _metric_row("covariance_diag", result, seconds, 5, "dynamic", covariance)
            )
            continue
        try:
            models, seconds = train_models(baseline_train_features, 5, covariance)
            result = evaluate_hmms(models, baseline_test_features)
            metrics.append(
                _metric_row(f"covariance_{covariance}", result, seconds, 5, "dynamic", covariance)
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            metrics.append(
                {
                    "experiment": f"covariance_{covariance}",
                    "n_states": 5,
                    "feature_set": "dynamic",
                    "covariance_type": covariance,
                    "accuracy": "",
                    "train_seconds": "",
                    "test_seconds_per_trial": "",
                    "n_test": sum(map(len, test_files.values())),
                    "error": str(error),
                }
            )
            print(f"WARNING: covariance={covariance} experiment failed: {error}")

    write_csv(output_dir / "experiment_metrics.csv", metrics)

    robustness_root = data_dir / "robustness"
    if robustness_root.is_dir():
        robustness_rows: list[dict[str, Any]] = []
        for condition in sorted(path for path in robustness_root.iterdir() if path.is_dir()):
            condition_files = {
                label: list_wavs(condition / label)
                for label in labels
                if list_wavs(condition / label)
            }
            if not condition_files:
                print(f"WARNING: No class WAV files found under robustness condition '{condition}'.")
                continue
            missing = set(labels) - set(condition_files)
            if missing:
                print(f"WARNING: Robustness condition '{condition.name}' has no files for {sorted(missing)}.")
            condition_test = {
                label: [(path, extract_features(path, "dynamic")) for path in paths]
                for label, paths in condition_files.items()
            }
            hmm_result = evaluate_hmms(baseline_models, condition_test)
            write_csv(
                output_dir / f"robustness_predictions_{condition.name}.csv",
                hmm_result["rows"],
            )
            robustness_rows.append(
                {
                    "condition": condition.name,
                    "system": "HMM",
                    "accuracy": hmm_result["accuracy"],
                    "n_test": len(hmm_result["truth"]),
                }
            )
            save_confusion_matrix(
                hmm_result,
                output_dir / f"confusion_robustness_{condition.name}.png",
                f"HMM robustness: {condition.name}",
            )
        write_csv(output_dir / "robustness_metrics.csv", robustness_rows)
    else:
        print("INFO: Add mismatch recordings under dataset/robustness/<condition>/<label>/ to run E5.")


def validate_data(data_dir: Path) -> tuple[list[str], dict[str, list[Path]], dict[str, list[Path]]]:
    labels = discover_labels(data_dir)
    train_files = {label: list_wavs(data_dir / "train" / label) for label in labels}
    test_files = {label: list_wavs(data_dir / "test" / label) for label in labels}
    errors: list[str] = []
    for label in labels:
        if len(train_files[label]) < 6:
            errors.append(f"{label}: expected at least 6 training WAV files, found {len(train_files[label])}")
        if len(test_files[label]) < 2:
            errors.append(f"{label}: expected at least 2 test WAV files, found {len(test_files[label])}")
    if errors:
        raise ValueError("Dataset is incomplete:\n- " + "\n- ".join(errors))

    train_paths = {path.resolve() for paths in train_files.values() for path in paths}
    test_paths = {path.resolve() for paths in test_files.values() for path in paths}
    overlap = train_paths & test_paths
    if overlap:
        raise ValueError("Data leakage: the same WAV file is present in both train and test.")

    print("Dataset inventory:")
    for label in labels:
        print(f"  {label}: {len(train_files[label])} train, {len(test_files[label])} test")
    return labels, train_files, test_files


def run(data_dir: Path, output_dir: Path, contexts_path: Path, corpus_path: Path, lexicon_path: Path) -> None:
    labels, train_files, test_files = validate_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lexicon = parse_lexicon(lexicon_path)
    print(f"Lexicon entries loaded: {len(lexicon)}")

    save_feature_heatmaps(train_files, output_dir)
    train_features, test_features = _train_test_feature_cache(train_files, test_files, "dynamic")
    baseline_models, train_seconds = train_models(train_features, n_states=5, covariance_type="diag")
    result = evaluate_hmms(baseline_models, test_features)
    save_confusion_matrix(result, output_dir / "confusion_hmm.png", "5-state Gaussian HMM")
    save_viterbi_plots(baseline_models, test_features, output_dir)
    write_csv(output_dir / "likelihood_scores.csv", result["rows"])

    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for label, model in baseline_models.items():
        joblib.dump(model, model_dir / f"{label}.joblib")

    static_train_features, static_test_features = _train_test_feature_cache(
        train_files, test_files, "static"
    )
    dtw_train_seconds = 0.0
    dtw_result = evaluate_dtw(static_train_features, static_test_features)
    write_csv(
        output_dir / "dtw_predictions.csv",
        [
            {"true_label": actual, "prediction": predicted}
            for actual, predicted in zip(dtw_result["truth"], dtw_result["predictions"])
        ],
    )
    save_confusion_matrix(dtw_result, output_dir / "confusion_dtw.png", "MFCC template DTW baseline")

    primary_metrics = {
        "hmm": {
            "accuracy": result["accuracy"],
            "train_seconds": train_seconds,
            "test_seconds_per_trial": result["seconds_per_trial"],
            "n_train_per_class": {label: len(train_files[label]) for label in labels},
        },
        "dtw": {
            "accuracy": dtw_result["accuracy"],
            "train_seconds": dtw_train_seconds,
            "test_seconds_per_trial": dtw_result["seconds_per_trial"],
            "n_templates_per_class": {label: len(train_files[label]) for label in labels},
        },
        "n_test": len(result["truth"]),
        "labels": labels,
    }
    comparison_rows = [
        {
            "metric": "accuracy",
            "DTW": dtw_result["accuracy"],
            "HMM": result["accuracy"],
            "notes": "Same train/test split",
        },
        {
            "metric": "training_seconds",
            "DTW": dtw_train_seconds,
            "HMM": train_seconds,
            "notes": "Template DTW has no iterative model fitting",
        },
        {
            "metric": "test_seconds_per_trial",
            "DTW": dtw_result["seconds_per_trial"],
            "HMM": result["seconds_per_trial"],
            "notes": "Scoring time; excludes feature extraction",
        },
        {
            "metric": "top_confusion_pair",
            "DTW": _top_confusion_pair(dtw_result),
            "HMM": _top_confusion_pair(result),
            "notes": "Actual->predicted (count)",
        },
        {
            "metric": "alignment",
            "DTW": "warping path",
            "HMM": "Viterbi state path",
            "notes": "",
        },
    ]
    write_csv(output_dir / "comparison_dtw_hmm.csv", comparison_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(primary_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"HMM accuracy: {result['accuracy']:.3f}")
    print(f"DTW accuracy: {dtw_result['accuracy']:.3f}")

    run_experiments(
        data_dir,
        output_dir,
        labels,
        train_files,
        test_files,
        baseline_models,
        train_features,
        test_features,
        result,
        train_seconds,
        dtw_result,
        dtw_train_seconds,
    )
    run_bigram_reranking(
        contexts_path,
        data_dir,
        test_files,
        test_features,
        baseline_models,
        corpus_path,
        output_dir,
    )
    from Lab03_report import generate_report

    report_path = generate_report(
        data_dir,
        output_dir,
        output_dir / "Lab03_report.docx",
    )
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Report saved to: {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="Dataset root folder.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output folder.")
    parser.add_argument(
        "--skip-m4a-conversion",
        action="store_true",
        help="Skip automatic M4A conversion and use WAV files already in the dataset.",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=None,
        help="CSV mapping test WAVs to previous-word context for bigram re-ranking.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Small whitespace-tokenized command-sequence corpus for bigram counts.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("lexicon.txt"),
        help="Pronunciation lexicon supplied by the instructor.",
    )
    arguments = parser.parse_args()
    if not arguments.skip_m4a_conversion:
        convert_m4a_dataset(arguments.data)
    run(
        arguments.data,
        arguments.output,
        arguments.contexts or arguments.data / "contexts.csv",
        arguments.corpus or arguments.data / "commands.txt",
        arguments.lexicon,
    )


if __name__ == "__main__":
    main()
