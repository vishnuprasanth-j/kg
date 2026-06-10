"""
evaluate_embedding_neighbors.py
-------------------------------
Evaluate what "similar flats" means in each learned embedding space.

The Streamlit app retrieves neighbours using Euclidean distance between the
exported flat embeddings. This script applies the same rule to every flat and
summarizes the characteristics shared by its top-k neighbours.

It also includes:
  - random: chance-level neighbours
  - attributes_only: Euclidean neighbours from standardized non-price features

Price is used only as an external evaluation characteristic. It does not
define embedding similarity and was excluded from ComplExLiteral training.

Usage:
    python evaluate_embedding_neighbors.py
    python evaluate_embedding_neighbors.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import Graph
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
ATTRS_FILE = ROOT / "vienna_kg_attributes.json"
TTL_FILE = ROOT / "vienna_kg_entities.ttl"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings"
RESULT_DIR = ROOT / "artifacts" / "results"

ATTRIBUTE_FEATURES = [
    "rooms",
    "size",
    "floor",
    "location_quality",
    "transit_score",
    "latitude",
    "longitude",
]

MODEL_NAMES = {
    "transe": "TransE",
    "complex": "ComplEx",
    "rotate": "RotatE",
    "complex_literal": "ComplExLiteral",
}


def load_attributes(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_embeddings(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def model_name(path: Path) -> str:
    key = path.name.replace("_flat_embeddings.json", "")
    return MODEL_NAMES.get(key, key)


def load_districts(path: Path) -> dict[str, str]:
    graph = Graph()
    graph.parse(path, format="turtle")
    mapping = {}
    for subject, predicate, obj in graph:
        if str(predicate).rsplit("/", 1)[-1] == "inDistrict":
            mapping[str(subject)] = str(obj).rsplit("/", 1)[-1]
    return mapping


def build_flat_table(
    attributes: dict,
    districts: dict[str, str],
    uris: list[str],
) -> pd.DataFrame:
    rows = []
    for uri in uris:
        item = attributes[uri]
        size = item.get("size")
        price = item.get("price")
        rent_per_m2 = np.nan
        if size not in (None, 0) and price is not None:
            rent_per_m2 = float(price) / float(size)

        rows.append(
            {
                "uri": uri,
                "flat": uri.rsplit("/", 1)[-1],
                "district": districts.get(uri, "unknown"),
                "price": price,
                "rent_per_m2": rent_per_m2,
                **{
                    feature: item.get(feature)
                    for feature in ATTRIBUTE_FEATURES
                },
            }
        )
    return pd.DataFrame(rows)


def nearest_indices(matrix: np.ndarray, top_k: int) -> np.ndarray:
    count = matrix.shape[0]
    if count <= top_k:
        raise ValueError(
            f"Need more than {top_k} flats, but only {count} are available."
        )

    finder = NearestNeighbors(
        n_neighbors=top_k + 1,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    )
    finder.fit(matrix)
    _, indices = finder.kneighbors(matrix)

    neighbours = np.empty((count, top_k), dtype=int)
    for query_index, candidates in enumerate(indices):
        selected = candidates[candidates != query_index][:top_k]
        if selected.size != top_k:
            raise RuntimeError(
                f"Could not find {top_k} non-self neighbours for row {query_index}."
            )
        neighbours[query_index] = selected
    return neighbours


def random_indices(count: int, top_k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    neighbours = np.empty((count, top_k), dtype=int)
    all_indices = np.arange(count)
    for query_index in range(count):
        candidates = np.delete(all_indices, query_index)
        neighbours[query_index] = rng.choice(
            candidates,
            size=top_k,
            replace=False,
        )
    return neighbours


def haversine_km(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    neighbour_lat: np.ndarray,
    neighbour_lon: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(query_lat)
    lon1 = np.radians(query_lon)
    lat2 = np.radians(neighbour_lat)
    lon2 = np.radians(neighbour_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def nanmean_rows(values: np.ndarray) -> np.ndarray:
    valid_count = np.sum(~np.isnan(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(
        total,
        valid_count,
        out=np.full(values.shape[0], np.nan),
        where=valid_count > 0,
    )


def evaluate_neighbours(
    method: str,
    flats: pd.DataFrame,
    neighbour_indices: np.ndarray,
    top_k: int,
) -> tuple[dict, pd.DataFrame]:
    query_indices = np.arange(len(flats))[:, None]
    numeric_columns = [
        "size",
        "rooms",
        "transit_score",
        "location_quality",
        "price",
        "rent_per_m2",
    ]

    per_query = pd.DataFrame(
        {
            "method": method,
            "uri": flats["uri"],
            "flat": flats["flat"],
            "top_k": top_k,
        }
    )

    districts = flats["district"].astype(str).to_numpy()
    same_district = (
        districts[neighbour_indices] == districts[query_indices]
    ).mean(axis=1)
    per_query["same_district_rate"] = same_district

    for column in numeric_columns:
        values = pd.to_numeric(flats[column], errors="coerce").to_numpy(float)
        differences = np.abs(
            values[neighbour_indices] - values[query_indices]
        )
        per_query[f"mean_{column}_difference"] = nanmean_rows(differences)

    latitudes = pd.to_numeric(flats["latitude"], errors="coerce").to_numpy(float)
    longitudes = pd.to_numeric(
        flats["longitude"],
        errors="coerce",
    ).to_numpy(float)
    geo_distances = haversine_km(
        latitudes[query_indices],
        longitudes[query_indices],
        latitudes[neighbour_indices],
        longitudes[neighbour_indices],
    )
    per_query["mean_geographic_distance_km"] = nanmean_rows(geo_distances)

    summary = {
        "method": method,
        "flats_evaluated": len(flats),
        "top_k": top_k,
    }
    metric_columns = [
        column
        for column in per_query.columns
        if column.startswith("same_") or column.startswith("mean_")
    ]
    for column in metric_columns:
        summary[column] = float(per_query[column].mean())
        summary[f"median_query_{column}"] = float(per_query[column].median())
    return summary, per_query


def attribute_neighbours(flats: pd.DataFrame, top_k: int) -> np.ndarray:
    matrix = flats[ATTRIBUTE_FEATURES].to_numpy(dtype=float)
    matrix = SimpleImputer(strategy="median").fit_transform(matrix)
    matrix = StandardScaler().fit_transform(matrix)
    return nearest_indices(matrix, top_k)


def print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "method",
        "same_district_rate",
        "mean_size_difference",
        "mean_rooms_difference",
        "mean_transit_score_difference",
        "mean_location_quality_difference",
        "mean_geographic_distance_km",
        "mean_price_difference",
        "mean_rent_per_m2_difference",
    ]
    display = summary[columns].copy()
    display["same_district_rate"] *= 100
    display = display.rename(
        columns={
            "same_district_rate": "same_district_%",
            "mean_size_difference": "size_diff_m2",
            "mean_rooms_difference": "rooms_diff",
            "mean_transit_score_difference": "transit_diff",
            "mean_location_quality_difference": "location_quality_diff",
            "mean_geographic_distance_km": "geo_diff_km",
            "mean_price_difference": "rent_diff_eur",
            "mean_rent_per_m2_difference": "rent_per_m2_diff_eur",
        }
    )
    print("\nEmbedding neighbourhood evaluation")
    print("=" * 115)
    print(display.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=Path, default=ATTRS_FILE)
    parser.add_argument("--ttl", type=Path, default=TTL_FILE)
    parser.add_argument("--embedding-dir", type=Path, default=EMBEDDING_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embedding_files = sorted(
        args.embedding_dir.glob("*_flat_embeddings.json")
    )
    if not embedding_files:
        raise FileNotFoundError(
            f"No embedding files found in {args.embedding_dir}."
        )

    attributes = load_attributes(args.attributes)
    districts = load_districts(args.ttl)
    embeddings_by_model = {
        model_name(path): load_embeddings(path)
        for path in embedding_files
    }

    shared_uris = set(attributes)
    for embeddings in embeddings_by_model.values():
        shared_uris &= set(embeddings)
    shared_uris = sorted(shared_uris)
    if len(shared_uris) <= args.top_k:
        raise ValueError(
            "Too few flats are shared by all embedding models for evaluation."
        )

    flats = build_flat_table(attributes, districts, shared_uris)
    print(f"Embedding models found: {', '.join(embeddings_by_model)}")
    print(f"Shared flats evaluated: {len(flats)}")
    print(f"Neighbours per flat   : {args.top_k}")
    print("Similarity rule       : Euclidean distance, matching app.py")
    print("Price defines similarity: no")

    summaries = []
    per_query_frames = []

    random_neighbours = random_indices(len(flats), args.top_k, args.seed)
    summary, details = evaluate_neighbours(
        "Random",
        flats,
        random_neighbours,
        args.top_k,
    )
    summaries.append(summary)
    per_query_frames.append(details)

    attr_neighbours = attribute_neighbours(flats, args.top_k)
    summary, details = evaluate_neighbours(
        "AttributesOnly",
        flats,
        attr_neighbours,
        args.top_k,
    )
    summaries.append(summary)
    per_query_frames.append(details)

    for name, embeddings in embeddings_by_model.items():
        matrix = np.array(
            [embeddings[uri] for uri in shared_uris],
            dtype=float,
        )
        neighbour_indices = nearest_indices(matrix, args.top_k)
        summary, details = evaluate_neighbours(
            name,
            flats,
            neighbour_indices,
            args.top_k,
        )
        summaries.append(summary)
        per_query_frames.append(details)

    summary_df = pd.DataFrame(summaries)
    details_df = pd.concat(per_query_frames, ignore_index=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.result_dir / "embedding_neighbor_summary.csv"
    details_path = args.result_dir / "embedding_neighbor_per_flat.csv"
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print_summary(summary_df)
    print(f"\nSaved summary to  : {summary_path}")
    print(f"Saved per-flat data: {details_path}")


if __name__ == "__main__":
    main()
