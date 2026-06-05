"""
train_literal_kge_models.py
---------------------------
Train literal-aware Knowledge Graph Embedding models on the Vienna KG.

This script adds a fair LiteralE-style experiment for LO1. It trains
ComplExLiteral with numeric flat literals, while excluding price to avoid
target leakage in the downstream price-prediction task.

Literal features used:
  rooms, size, floor, location_quality, transit_score, latitude, longitude

Usage:
    python train_literal_kge_models.py
    python train_literal_kge_models.py --epochs 120
    python train_literal_kge_models.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesNumericLiteralsFactory

from train_kge_models import (
    ATTRS_FILE,
    EMBEDDING_DIR,
    KG_DIR,
    MODEL_DIR,
    RESULT_DIR,
    TTL_FILE,
    choose_device,
    copy_source_artifacts,
    ensure_artifact_dirs,
    flattened_entity_embeddings,
    ghost_entity_count,
    metric_value,
    parse_object_triples,
)


LITERAL_FEATURES = [
    "rooms",
    "size",
    "floor",
    "location_quality",
    "transit_score",
    "latitude",
    "longitude",
]


@dataclass(frozen=True)
class LiteralModelSpec:
    name: str
    directory_name: str
    model_kwargs: dict


MODEL_SPECS = {
    "complexliteral": LiteralModelSpec(
        name="ComplExLiteral",
        directory_name="complex_literal",
        model_kwargs={"embedding_dim": 100, "input_dropout": 0.2},
    ),
}


def load_attributes(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def feature_statistics(attributes: dict, features: list[str]) -> dict:
    stats = {}
    for feature in features:
        values = [
            float(item[feature])
            for item in attributes.values()
            if item.get(feature) is not None
        ]
        if not values:
            stats[feature] = {"mean": 0.0, "std": 1.0, "count": 0}
            continue

        array = np.array(values, dtype=float)
        std = float(array.std())
        stats[feature] = {
            "mean": float(array.mean()),
            "std": std if std > 0 else 1.0,
            "count": int(array.size),
        }
    return stats


def build_numeric_literal_triples(
    attributes: dict,
    features: list[str],
    stats: dict,
) -> np.ndarray:
    rows = []
    for entity_uri, item in attributes.items():
        for feature in features:
            value = item.get(feature)
            if value is None:
                continue
            normalized = (float(value) - stats[feature]["mean"]) / stats[feature]["std"]
            rows.append(
                (
                    entity_uri,
                    f"http://example.org/viennakg/literal/{feature}",
                    f"{normalized:.8f}",
                )
            )
    return np.array(rows, dtype=str)


def build_literal_factory(
    triples: np.ndarray,
    numeric_literal_triples: np.ndarray,
) -> TriplesNumericLiteralsFactory:
    return TriplesNumericLiteralsFactory.from_labeled_triples(
        triples=triples,
        numeric_triples=numeric_literal_triples,
    )


def save_flat_embeddings(model, triples_factory: TriplesNumericLiteralsFactory, path: Path) -> int:
    all_embeddings = flattened_entity_embeddings(model)

    flat_embeddings = {}
    for entity_uri, entity_id in triples_factory.entity_to_id.items():
        if "flat_" in entity_uri:
            flat_embeddings[entity_uri] = all_embeddings[entity_id].tolist()

    with path.open("w", encoding="utf-8") as file:
        json.dump(flat_embeddings, file)

    return len(flat_embeddings)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_literal_model(
    spec: LiteralModelSpec,
    train_tf: TriplesNumericLiteralsFactory,
    valid_tf: TriplesNumericLiteralsFactory,
    test_tf: TriplesNumericLiteralsFactory,
    full_tf: TriplesNumericLiteralsFactory,
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
        "literal_features": ",".join(LITERAL_FEATURES),
        "price_literal_included": False,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ComplExLiteral"],
        help="Literal-aware models to train. Currently supported: ComplExLiteral",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the literal-aware triples factory, then stop before training.",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    copy_source_artifacts()
    KG_DIR.mkdir(parents=True, exist_ok=True)

    triples_array, dataset_summary = parse_object_triples(TTL_FILE)
    attributes = load_attributes(ATTRS_FILE)
    stats = feature_statistics(attributes, LITERAL_FEATURES)
    numeric_literal_triples = build_numeric_literal_triples(
        attributes=attributes,
        features=LITERAL_FEATURES,
        stats=stats,
    )

    literal_metadata = {
        "literal_features": LITERAL_FEATURES,
        "price_literal_included": False,
        "normalization": stats,
        "numeric_literal_triples": int(numeric_literal_triples.shape[0]),
        **dataset_summary,
    }
    with (RESULT_DIR / "literal_feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(literal_metadata, file, indent=2)

    tf = build_literal_factory(triples_array, numeric_literal_triples)
    train_tf, valid_tf, test_tf = tf.split([0.8, 0.1, 0.1], random_state=42)

    print(f"RDF triples total      : {dataset_summary['rdf_triples_total']}")
    print(f"Object triples used    : {dataset_summary['object_triples_used']}")
    print(f"Literal triples skipped: {dataset_summary['literal_triples_skipped']}")
    print(f"Numeric literal triples: {numeric_literal_triples.shape[0]}")
    print(f"Literal features       : {', '.join(LITERAL_FEATURES)}")
    print("Price literal included : no")
    print(f"Entities               : {tf.num_entities}")
    print(f"Relations              : {tf.num_relations}")
    print(f"Literal columns        : {len(tf.literals_to_id)}")
    print(f"Ghost entities         : {ghost_entity_count(tf)}")

    if args.dry_run:
        print("\nDry run complete. No model was trained.")
        return

    device = choose_device()
    selected = []
    for model_name in args.models:
        key = model_name.lower()
        if key not in MODEL_SPECS:
            valid = ", ".join(spec.name for spec in MODEL_SPECS.values())
            raise ValueError(f"Unknown literal model '{model_name}'. Choose from: {valid}")
        selected.append(MODEL_SPECS[key])

    rows = []
    for spec in selected:
        rows.append(
            train_one_literal_model(
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

    output_csv = RESULT_DIR / "literal_kge_model_comparison.csv"
    write_csv(output_csv, rows)
    print(f"\nSaved literal-aware KGE comparison to {output_csv}")


if __name__ == "__main__":
    main()
