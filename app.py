"""
app.py
------
Streamlit demo for the Vienna Transit-Aware Flat Price Knowledge Graph.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TTL_FILE = ROOT / "vienna_kg_entities.ttl"
ATTRS_FILE = ROOT / "vienna_kg_attributes.json"
RESULT_DIR = ROOT / "artifacts" / "results"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings"
PREDICTION_DIR = ROOT / "artifacts" / "predictions"


@st.cache_data
def load_attributes() -> dict:
    with ATTRS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def ttl_relation_counts() -> dict:
    counts = {
        "isNearStop": 0,
        "isOnLine": 0,
        "inDistrict": 0,
        "hasLocationQuality": 0,
    }
    if not TTL_FILE.exists():
        return counts

    with TTL_FILE.open(encoding="utf-8") as file:
        for line in file:
            for relation in counts:
                if f"ex:{relation}" in line:
                    counts[relation] += 1
    return counts


def local_name(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def flat_label(uri: str, attrs: dict) -> str:
    item = attrs[uri]
    price = item.get("price")
    size = item.get("size")
    transit = item.get("transit_score")
    return f"{local_name(uri)} | EUR {price:.0f} | {size or '?'} m2 | transit {transit}"


def attributes_table(attrs: dict, uris: list[str]) -> pd.DataFrame:
    rows = []
    for uri in uris:
        item = attrs[uri]
        rows.append(
            {
                "flat": local_name(uri),
                "price": item.get("price"),
                "size": item.get("size"),
                "rooms": item.get("rooms"),
                "floor": item.get("floor"),
                "transit_score": item.get("transit_score"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
            }
        )
    return pd.DataFrame(rows)


def find_similar_flats(
    embeddings: dict,
    attrs: dict,
    query_uri: str,
    top_n: int,
) -> pd.DataFrame:
    uris = [uri for uri in embeddings if uri in attrs]
    matrix = np.array([embeddings[uri] for uri in uris], dtype=float)
    query_index = uris.index(query_uri)
    distances = np.linalg.norm(matrix - matrix[query_index], axis=1)
    order = np.argsort(distances)

    result_uris = []
    result_distances = []
    for idx in order:
        if uris[idx] == query_uri:
            continue
        result_uris.append(uris[idx])
        result_distances.append(float(distances[idx]))
        if len(result_uris) >= top_n:
            break

    table = attributes_table(attrs, result_uris)
    table.insert(1, "embedding_distance", result_distances)
    return table


def flat_triple_lines(flat_uri: str, limit: int = 30) -> list[str]:
    name = local_name(flat_uri)
    lines = []
    if not TTL_FILE.exists():
        return lines
    with TTL_FILE.open(encoding="utf-8") as file:
        for line in file:
            if f"ex:{name}" in line:
                lines.append(line.strip())
                if len(lines) >= limit:
                    break
    return lines


def main() -> None:
    st.set_page_config(
        page_title="Vienna Transit KG",
        page_icon=None,
        layout="wide",
    )

    st.title("Vienna Transit-Aware Flat Price Knowledge Graph")

    attrs = load_attributes()
    flat_uris = sorted(attrs)

    tab_overview, tab_models, tab_flat, tab_similar, tab_price = st.tabs(
        [
            "KG Overview",
            "Model Comparison",
            "Flat Explorer",
            "Similar Flats",
            "Price Prediction",
        ]
    )

    with tab_overview:
        counts = ttl_relation_counts()
        dataset_summary_path = RESULT_DIR / "kge_dataset_summary.json"
        summary = load_json(str(dataset_summary_path)) if dataset_summary_path.exists() else {}

        cols = st.columns(4)
        cols[0].metric("Flats", f"{len(attrs):,}")
        cols[1].metric("Triples Used", f"{summary.get('object_triples_used', 0):,}")
        cols[2].metric("Entities", f"{summary.get('entities', 0):,}")
        cols[3].metric("Relations", f"{summary.get('relations', 0):,}")

        st.dataframe(
            pd.DataFrame(
                [{"relation": key, "count": value} for key, value in counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_models:
        comparison_path = RESULT_DIR / "kge_model_comparison.csv"
        if comparison_path.exists():
            df = load_csv(str(comparison_path))
            columns = ["model", "mrr", "hits_at_1", "hits_at_3", "hits_at_10"]
            st.dataframe(df[columns], use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("model")[["mrr", "hits_at_10"]])
        else:
            st.info("Run train_kge_models.py to generate model comparison results.")

    with tab_flat:
        selected = st.selectbox(
            "Flat",
            flat_uris,
            format_func=lambda uri: flat_label(uri, attrs),
        )
        st.dataframe(attributes_table(attrs, [selected]), use_container_width=True)

        lines = flat_triple_lines(selected)
        if lines:
            st.code("\n".join(lines), language="ttl")
        else:
            st.info("No direct flat lines found in the Turtle file.")

    with tab_similar:
        embedding_files = sorted(EMBEDDING_DIR.glob("*_flat_embeddings.json"))
        if not embedding_files:
            st.info("Run train_kge_models.py to create flat embeddings.")
        else:
            selected_embedding = st.selectbox(
                "Embedding model",
                embedding_files,
                format_func=lambda path: path.name.replace("_flat_embeddings.json", ""),
            )
            embeddings = load_json(str(selected_embedding))
            available_flats = sorted(uri for uri in embeddings if uri in attrs)
            selected_flat = st.selectbox(
                "Query flat",
                available_flats,
                format_func=lambda uri: flat_label(uri, attrs),
            )
            top_n = st.slider("Number of similar flats", 3, 20, 10)
            table = find_similar_flats(embeddings, attrs, selected_flat, top_n)
            st.dataframe(table, use_container_width=True, hide_index=True)

    with tab_price:
        result_path = RESULT_DIR / "price_prediction_results.csv"
        if result_path.exists():
            results = load_csv(str(result_path))
            st.dataframe(results, use_container_width=True, hide_index=True)

            price_rows = results[results["target"] == "price"]
            if not price_rows.empty:
                chart = price_rows.pivot(
                    index="model",
                    columns="experiment",
                    values="mae",
                )
                st.bar_chart(chart)
        else:
            st.info("Run evaluate_price.py to generate price prediction results.")

        prediction_files = sorted(PREDICTION_DIR.glob("*_price_predictions.csv"))
        if prediction_files:
            selected_prediction = st.selectbox(
                "Predictions",
                prediction_files,
                format_func=lambda path: path.name,
            )
            predictions = load_csv(str(selected_prediction))
            st.dataframe(predictions.head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
