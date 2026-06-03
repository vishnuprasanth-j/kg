"""
evaluate_price.py
-----------------
Required downstream price-prediction experiment for the Vienna KG project.

The KGE models are trained for link prediction. This script evaluates whether
their flat embeddings help in a separate supervised rent-prediction task.

Experiments per embedding model:
  1. mean baseline
  2. attributes only
  3. embeddings only
  4. attributes + embeddings

Usage:
    python evaluate_price.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
ATTRS_FILE = ROOT / "vienna_kg_attributes.json"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings"
RESULT_DIR = ROOT / "artifacts" / "results"
PREDICTION_DIR = ROOT / "artifacts" / "predictions"

DEFAULT_ATTR_KEYS = [
    "rooms",
    "size",
    "floor",
    "location_quality",
    "transit_score",
    "latitude",
    "longitude",
]


def load_attributes(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_embeddings(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def model_name_from_embedding_file(path: Path) -> str:
    return path.name.replace("_flat_embeddings.json", "")


def attribute_matrix(attributes: dict, uris: list[str], keys: list[str]) -> np.ndarray:
    rows = []
    for uri in uris:
        item = attributes[uri]
        rows.append([item.get(key) for key in keys])
    return np.array(rows, dtype=float)


def embedding_matrix(embeddings: dict, uris: list[str]) -> np.ndarray:
    return np.array([embeddings[uri] for uri in uris], dtype=float)


def target_vector(attributes: dict, uris: list[str], target: str) -> np.ndarray:
    values = []
    for uri in uris:
        item = attributes[uri]
        price = float(item["price"])
        if target == "price":
            values.append(price)
        elif target == "log_price":
            values.append(np.log1p(price))
        elif target == "price_per_m2":
            size = item.get("size")
            if not size or float(size) <= 0:
                values.append(np.nan)
            else:
                values.append(price / float(size))
        else:
            raise ValueError(f"Unknown target: {target}")
    return np.array(values, dtype=float)


def ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def evaluate_features(
    X: np.ndarray,
    y: np.ndarray,
    experiment: str,
    model_name: str,
    target: str,
    cv: KFold,
) -> tuple[dict, np.ndarray]:
    if experiment == "baseline_mean":
        estimator = DummyRegressor(strategy="mean")
    else:
        estimator = ridge_pipeline()

    predictions = cross_val_predict(estimator, X, y, cv=cv)
    row = {
        "model": model_name,
        "target": target,
        "experiment": experiment,
        "n": len(y),
        "r2": r2_score(y, predictions),
        "mae": mean_absolute_error(y, predictions),
    }
    return row, predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_embedding_file(
    embedding_file: Path,
    attributes: dict,
    attr_keys: list[str],
    targets: list[str],
    cv: KFold,
) -> list[dict]:
    model_name = model_name_from_embedding_file(embedding_file)
    embeddings = load_embeddings(embedding_file)

    common = sorted(
        uri
        for uri in embeddings
        if uri in attributes and attributes[uri].get("price") is not None
    )
    if not common:
        print(f"No common flats for {embedding_file}")
        return []

    X_attr_all = attribute_matrix(attributes, common, attr_keys)
    X_emb_all = embedding_matrix(embeddings, common)
    X_combined_all = np.hstack([X_attr_all, X_emb_all])

    rows = []
    for target in targets:
        y_all = target_vector(attributes, common, target)
        keep = ~np.isnan(y_all)
        uris = [uri for uri, ok in zip(common, keep) if ok]
        y = y_all[keep]

        matrices = {
            "baseline_mean": np.zeros((len(y), 1)),
            "attributes_only": X_attr_all[keep],
            "embeddings_only": X_emb_all[keep],
            "attributes_plus_embeddings": X_combined_all[keep],
        }

        prediction_rows = []
        for experiment, X in matrices.items():
            row, predictions = evaluate_features(
                X=X,
                y=y,
                experiment=experiment,
                model_name=model_name,
                target=target,
                cv=cv,
            )
            rows.append(row)

            if target == "price" and experiment == "attributes_plus_embeddings":
                for uri, actual, predicted in zip(uris, y, predictions):
                    prediction_rows.append(
                        {
                            "uri": uri,
                            "actual_price": actual,
                            "predicted_price": predicted,
                            "absolute_error": abs(actual - predicted),
                        }
                    )

        if prediction_rows:
            output = PREDICTION_DIR / f"{model_name}_price_predictions.csv"
            write_csv(output, prediction_rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=Path, default=ATTRS_FILE)
    parser.add_argument("--embedding-dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["price", "price_per_m2"],
        choices=["price", "log_price", "price_per_m2"],
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    attributes = load_attributes(args.attributes)
    embedding_files = sorted(args.embedding_dir.glob("*_flat_embeddings.json"))
    if not embedding_files:
        raise FileNotFoundError(
            f"No embedding files found in {args.embedding_dir}. "
            "Run train_kge_models.py first."
        )

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_rows = []
    for embedding_file in embedding_files:
        print(f"Evaluating {embedding_file.name}")
        rows = evaluate_embedding_file(
            embedding_file=embedding_file,
            attributes=attributes,
            attr_keys=DEFAULT_ATTR_KEYS,
            targets=args.targets,
            cv=cv,
        )
        all_rows.extend(rows)

    output = RESULT_DIR / "price_prediction_results.csv"
    write_csv(output, all_rows)

    print(f"\nSaved price prediction results to {output}")
    for row in all_rows:
        if row["target"] == "price":
            print(
                f"{row['model']:8s} {row['experiment']:28s} "
                f"R2={row['r2']:.3f} MAE={row['mae']:.1f}"
            )


if __name__ == "__main__":
    main()
