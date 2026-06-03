"""
train_kge_models.py
-------------------
Train several Knowledge Graph Embedding models on the Vienna KG.

This script is the LO1-oriented training entry point. It trains the same
structural RDF graph with TransE, ComplEx, and RotatE, then writes comparable
link-prediction metrics and flat embeddings for downstream tasks.

Why these models?
  - TransE: simple translational baseline.
  - ComplEx: can represent asymmetric relations better than DistMult.
  - RotatE: relation-as-rotation model, useful for asymmetric/inverse patterns.

Usage:
    python train_kge_models.py
    python train_kge_models.py --models TransE ComplEx RotatE --epochs 120
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdflib import Graph, Literal

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


ROOT = Path(__file__).resolve().parent
TTL_FILE = ROOT / "vienna_kg_entities.ttl"
ATTRS_FILE = ROOT / "vienna_kg_attributes.json"

ARTIFACT_DIR = ROOT / "artifacts"
KG_DIR = ARTIFACT_DIR / "kg"
MODEL_DIR = ARTIFACT_DIR / "models"
EMBEDDING_DIR = ARTIFACT_DIR / "embeddings"
RESULT_DIR = ARTIFACT_DIR / "results"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory_name: str
    model_kwargs: dict


MODEL_SPECS = {
    "transe": ModelSpec(
        name="TransE",
        directory_name="transe",
        model_kwargs={"embedding_dim": 100, "scoring_fct_norm": 1},
    ),
    "complex": ModelSpec(
        name="ComplEx",
        directory_name="complex",
        model_kwargs={"embedding_dim": 100},
    ),
    "rotate": ModelSpec(
        name="RotatE",
        directory_name="rotate",
        model_kwargs={"embedding_dim": 100},
    ),
}


def ensure_artifact_dirs() -> None:
    for directory in [KG_DIR, MODEL_DIR, EMBEDDING_DIR, RESULT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def copy_source_artifacts() -> None:
    """Keep a copy of the KG inputs beside the generated model artifacts."""
    if TTL_FILE.exists():
        shutil.copy2(TTL_FILE, KG_DIR / TTL_FILE.name)
    if ATTRS_FILE.exists():
        shutil.copy2(ATTRS_FILE, KG_DIR / ATTRS_FILE.name)


def normalize_model_name(name: str) -> str:
    key = name.lower()
    aliases = {
        "complex": "complex",
        "complexe": "complex",
        "complEx".lower(): "complex",
        "rotate": "rotate",
        "rotatE".lower(): "rotate",
        "transe": "transe",
    }
    if key not in aliases:
        valid = ", ".join(spec.name for spec in MODEL_SPECS.values())
        raise ValueError(f"Unknown model '{name}'. Choose from: {valid}")
    return aliases[key]


def parse_object_triples(ttl_file: Path) -> tuple[np.ndarray, dict]:
    """
    Parse RDF and keep only triples whose object is not a literal.

    Literal-object triples such as rdfs:label "1" are useful for display, but
    bad for KGE training because PyKEEN treats literal values as entities.
    """
    graph = Graph()
    graph.parse(ttl_file, format="turtle")

    triples: list[tuple[str, str, str]] = []
    skipped_literals = 0
    for subject, predicate, obj in graph:
        if isinstance(obj, Literal):
            skipped_literals += 1
            continue
        triples.append((str(subject), str(predicate), str(obj)))

    summary = {
        "source_file": str(ttl_file),
        "rdf_triples_total": len(graph),
        "object_triples_used": len(triples),
        "literal_triples_skipped": skipped_literals,
    }
    return np.array(triples, dtype=str), summary


def choose_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("No GPU detected, using CPU")
    return "cpu"


def flattened_entity_embeddings(model) -> np.ndarray:
    """Return entity embeddings as a 2D real-valued numpy array."""
    with torch.no_grad():
        tensor = model.entity_representations[0](indices=None).detach().cpu()

    if torch.is_complex(tensor):
        tensor = torch.view_as_real(tensor)

    array = tensor.numpy()
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def save_flat_embeddings(model, triples_factory: TriplesFactory, path: Path) -> int:
    all_embeddings = flattened_entity_embeddings(model)

    flat_embeddings = {}
    for entity_uri, entity_id in triples_factory.entity_to_id.items():
        if "flat_" in entity_uri:
            flat_embeddings[entity_uri] = all_embeddings[entity_id].tolist()

    with path.open("w", encoding="utf-8") as file:
        json.dump(flat_embeddings, file)

    return len(flat_embeddings)


def ghost_entity_count(triples_factory: TriplesFactory) -> int:
    count = 0
    for entity in triples_factory.entity_to_id:
        label = entity.rsplit("/", 1)[-1]
        try:
            float(label)
            count += 1
        except ValueError:
            pass
    return count


def metric_value(metric_results, name: str) -> float:
    value = metric_results.get_metric(name)
    return float(value) if value is not None else float("nan")


def train_one_model(
    spec: ModelSpec,
    train_tf: TriplesFactory,
    valid_tf: TriplesFactory,
    test_tf: TriplesFactory,
    full_tf: TriplesFactory,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    print(f"\n=== Training {spec.name} ===")

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model=spec.name,
        model_kwargs=spec.model_kwargs,
        optimizer="Adam",
        optimizer_kwargs={"lr": learning_rate},
        loss="MarginRankingLoss",
        loss_kwargs={"margin": 1.0},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
        },
        stopper="early",
        stopper_kwargs={
            "frequency": 10,
            "patience": 5,
            "relative_delta": 0.002,
        },
        random_seed=42,
        device=device,
    )

    output_dir = MODEL_DIR / spec.directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(output_dir)

    embedding_path = EMBEDDING_DIR / f"{spec.directory_name}_flat_embeddings.json"
    flat_count = save_flat_embeddings(result.model, full_tf, embedding_path)

    metrics = result.metric_results
    row = {
        "model": spec.name,
        "model_dir": str(output_dir),
        "embedding_file": str(embedding_path),
        "flat_embeddings": flat_count,
        "mrr": metric_value(metrics, "mean_reciprocal_rank"),
        "hits_at_1": metric_value(metrics, "hits_at_1"),
        "hits_at_3": metric_value(metrics, "hits_at_3"),
        "hits_at_10": metric_value(metrics, "hits_at_10"),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    print(
        f"{spec.name}: MRR={row['mrr']:.4f}, "
        f"Hits@1={row['hits_at_1']:.4f}, Hits@10={row['hits_at_10']:.4f}"
    )
    print(f"Saved model to {output_dir}")
    print(f"Saved flat embeddings to {embedding_path}")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["TransE", "ComplEx", "RotatE"],
        help="Models to train. Choices: TransE ComplEx RotatE",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    ensure_artifact_dirs()
    copy_source_artifacts()

    device = choose_device()
    print(f"Using device: {device}")

    print(f"\nParsing KG: {TTL_FILE}")
    triples_array, dataset_summary = parse_object_triples(TTL_FILE)
    print(f"  RDF triples total      : {dataset_summary['rdf_triples_total']}")
    print(f"  Object triples used    : {dataset_summary['object_triples_used']}")
    print(f"  Literal triples skipped: {dataset_summary['literal_triples_skipped']}")

    tf = TriplesFactory.from_labeled_triples(triples_array)
    train_tf, valid_tf, test_tf = tf.split([0.8, 0.1, 0.1], random_state=42)

    dataset_summary.update(
        {
            "entities": tf.num_entities,
            "relations": tf.num_relations,
            "train_triples": train_tf.num_triples,
            "valid_triples": valid_tf.num_triples,
            "test_triples": test_tf.num_triples,
            "ghost_entities_after_literal_filter": ghost_entity_count(tf),
        }
    )
    with (RESULT_DIR / "kge_dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, indent=2)

    print(f"  Entities              : {tf.num_entities}")
    print(f"  Relations             : {tf.num_relations}")
    print(f"  Ghost entities         : {dataset_summary['ghost_entities_after_literal_filter']}")

    selected_specs = [MODEL_SPECS[normalize_model_name(name)] for name in args.models]
    rows = []
    for spec in selected_specs:
        rows.append(
            train_one_model(
                spec=spec,
                train_tf=train_tf,
                valid_tf=valid_tf,
                test_tf=test_tf,
                full_tf=tf,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        )

    output_csv = RESULT_DIR / "kge_model_comparison.csv"
    write_csv(output_csv, rows)
    print(f"\nSaved KGE comparison to {output_csv}")


if __name__ == "__main__":
    main()
