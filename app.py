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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover - Streamlit installs usually include pydeck
    pdk = None


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


@st.cache_data
def flat_district_map() -> dict:
    mapping = {}
    if not TTL_FILE.exists():
        return mapping

    current_flat_uri = None
    with TTL_FILE.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("ex:flat_"):
                flat_id = line.split()[0].replace("ex:", "")
                current_flat_uri = f"http://example.org/viennakg/{flat_id}"

            if "ex:inDistrict ex:" not in line:
                if line.endswith("."):
                    current_flat_uri = None
                continue

            parts = line.replace(".", "").replace(";", "").split()
            if line.startswith("ex:flat_") and len(parts) >= 3:
                flat_id = parts[0].replace("ex:", "")
                district = parts[2].replace("ex:", "")
                mapping[f"http://example.org/viennakg/{flat_id}"] = district
            elif current_flat_uri and len(parts) >= 2:
                district = parts[-1].replace("ex:", "")
                mapping[current_flat_uri] = district

            if line.endswith("."):
                current_flat_uri = None
    return mapping


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


def model_display_name(path: Path) -> str:
    names = {
        "transe": "TransE",
        "complex": "ComplEx",
        "rotate": "RotatE",
        "complex_literal": "ComplExLiteral",
    }
    key = path.name.replace("_flat_embeddings.json", "")
    return names.get(key, key)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


@st.cache_data
def flat_dataframe(attrs: dict, districts: dict) -> pd.DataFrame:
    rows = []
    for uri, item in attrs.items():
        price = item.get("price")
        size = item.get("size")
        rent_per_m2 = None
        if price is not None and size not in (None, 0):
            rent_per_m2 = float(price) / float(size)

        rows.append(
            {
                "uri": uri,
                "flat": local_name(uri),
                "price": price,
                "size": size,
                "rooms": item.get("rooms"),
                "floor": item.get("floor"),
                "location_quality": item.get("location_quality"),
                "transit_score": item.get("transit_score"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "rent_per_m2": rent_per_m2,
                "district": districts.get(uri, "unknown"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["latitude", "longitude", "price", "transit_score"])
    df["value_score"] = zscore(df["transit_score"]) - zscore(df["rent_per_m2"])
    return df


def display_flat_table(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    columns = [
        "flat",
        "district",
        "price",
        "size",
        "rooms",
        "rent_per_m2",
        "transit_score",
        "value_score",
    ]
    table = df[columns].copy()
    table = table.round(
        {
            "price": 0,
            "size": 1,
            "rooms": 1,
            "rent_per_m2": 2,
            "transit_score": 0,
            "value_score": 2,
        }
    )
    if limit is not None:
        table = table.head(limit)
    return table


def add_prediction_columns(df: pd.DataFrame, prediction_file: Path | None) -> pd.DataFrame:
    if prediction_file is None or not prediction_file.exists():
        return df

    predictions = load_csv(str(prediction_file))
    if predictions.empty or "uri" not in predictions.columns:
        return df

    merged = df.merge(
        predictions[["uri", "predicted_price", "absolute_error"]],
        on="uri",
        how="left",
    )
    merged["prediction_error"] = merged["price"] - merged["predicted_price"]
    return merged


def color_scale(values: pd.Series, low: list[int], high: list[int]) -> list[list[int]]:
    clean = values.astype(float)
    q_low = clean.quantile(0.05)
    q_high = clean.quantile(0.95)
    if q_high == q_low:
        scaled = pd.Series(np.zeros(len(clean)), index=clean.index)
    else:
        scaled = ((clean - q_low) / (q_high - q_low)).clip(0, 1)

    colors = []
    for value in scaled.fillna(0.0):
        color = [
            int(low[idx] + value * (high[idx] - low[idx]))
            for idx in range(3)
        ]
        colors.append(color + [180])
    return colors


def map_colors(df: pd.DataFrame, color_by: str) -> list[list[int]]:
    if color_by == "Transit score":
        return color_scale(df["transit_score"], [222, 235, 247], [33, 113, 181])
    if color_by == "Rent per m2":
        return color_scale(df["rent_per_m2"], [254, 224, 210], [203, 24, 29])
    if color_by == "Transit value":
        return color_scale(df["value_score"], [254, 240, 217], [35, 139, 69])
    if color_by == "Prediction error" and "prediction_error" in df.columns:
        return color_scale(df["prediction_error"], [49, 130, 189], [215, 48, 39])
    return color_scale(df["price"], [254, 229, 217], [165, 15, 21])


def render_vienna_map(df: pd.DataFrame, color_by: str, height: int = 560) -> None:
    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        st.info("No mappable flats for the selected filters.")
        return

    map_df["color"] = map_colors(map_df, color_by)
    map_df["radius"] = 35 + 4 * np.sqrt(map_df["transit_score"].clip(lower=0))
    map_df["tooltip"] = (
        map_df["flat"]
        + "<br>District: "
        + map_df["district"].astype(str)
        + "<br>Price: EUR "
        + map_df["price"].round(0).astype(int).astype(str)
        + "<br>Rent/m2: EUR "
        + map_df["rent_per_m2"].round(2).astype(str)
        + "<br>Transit score: "
        + map_df["transit_score"].round(0).astype(int).astype(str)
    )

    if pdk is None:
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=48.2082,
        longitude=16.3738,
        zoom=10.7,
        pitch=0,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "white", "color": "#111"}},
        map_style="light",
    )
    try:
        st.pydeck_chart(deck, use_container_width=True, height=height)
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)


def render_similarity_map(
    query: pd.Series,
    similar: pd.DataFrame,
    height: int = 540,
) -> None:
    if pdk is None:
        points = pd.concat([query.to_frame().T, similar], ignore_index=True)
        st.map(points.rename(columns={"latitude": "lat", "longitude": "lon"}))
        return

    candidates = similar.dropna(subset=["latitude", "longitude"]).copy()
    if candidates.empty:
        st.info("No mappable similar flats.")
        return

    candidates["source_lon"] = float(query["longitude"])
    candidates["source_lat"] = float(query["latitude"])
    candidates["target_lon"] = candidates["longitude"].astype(float)
    candidates["target_lat"] = candidates["latitude"].astype(float)
    candidates["edge_color"] = color_scale(
        candidates["similarity"],
        [224, 92, 55],
        [26, 152, 80],
    )
    candidates["edge_width"] = 2 + 7 * candidates["similarity"]
    candidates["point_color"] = candidates["edge_color"]
    candidates["point_radius"] = 75
    candidates["tooltip"] = (
        candidates["flat"]
        + "<br>Similarity: "
        + candidates["similarity"].round(3).astype(str)
        + "<br>Price: EUR "
        + candidates["price"].round(0).astype(int).astype(str)
        + "<br>Rent/m2: EUR "
        + candidates["rent_per_m2"].round(2).astype(str)
        + "<br>Transit score: "
        + candidates["transit_score"].round(0).astype(int).astype(str)
    )

    query_df = pd.DataFrame(
        [
            {
                "longitude": float(query["longitude"]),
                "latitude": float(query["latitude"]),
                "color": [20, 33, 61, 240],
                "radius": 150,
                "tooltip": (
                    f"{query['flat']}<br>Selected flat"
                    f"<br>Price: EUR {query['price']:.0f}"
                    f"<br>Transit score: {query['transit_score']:.0f}"
                ),
            }
        ]
    )

    edge_layer = pdk.Layer(
        "ArcLayer",
        data=candidates,
        get_source_position="[source_lon, source_lat]",
        get_target_position="[target_lon, target_lat]",
        get_source_color="edge_color",
        get_target_color="edge_color",
        get_width="edge_width",
        pickable=False,
    )
    candidate_layer = pdk.Layer(
        "ScatterplotLayer",
        data=candidates,
        get_position="[longitude, latitude]",
        get_fill_color="point_color",
        get_radius="point_radius",
        pickable=True,
        auto_highlight=True,
    )
    query_layer = pdk.Layer(
        "ScatterplotLayer",
        data=query_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(query["latitude"]),
        longitude=float(query["longitude"]),
        zoom=11.2,
        pitch=28,
        bearing=0,
    )
    deck = pdk.Deck(
        layers=[edge_layer, candidate_layer, query_layer],
        initial_view_state=view_state,
        tooltip={"html": "{tooltip}", "style": {"backgroundColor": "white", "color": "#111"}},
        map_style="light",
    )
    try:
        st.pydeck_chart(deck, use_container_width=True, height=height)
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)


def flat_kg_paths(flat_uri: str, stop_limit: int = 12, route_limit: int = 40) -> pd.DataFrame:
    flat_id = local_name(flat_uri)
    stop_ids = []
    if not TTL_FILE.exists():
        return pd.DataFrame(columns=["flat", "relation_1", "stop", "relation_2", "route"])

    with TTL_FILE.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            prefix = f"ex:{flat_id} ex:isNearStop "
            if line.startswith(prefix):
                stop_ids.append(line.replace(prefix, "").replace(".", "").strip())
                if len(stop_ids) >= stop_limit:
                    break

    stop_set = set(stop_ids)
    rows = []
    with TTL_FILE.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if " ex:isOnLine " not in line:
                continue
            parts = line.replace(".", "").split()
            if len(parts) < 3 or parts[0] not in stop_set:
                continue
            rows.append(
                {
                    "flat": flat_id,
                    "relation_1": "isNearStop",
                    "stop": parts[0].replace("ex:", ""),
                    "relation_2": "isOnLine",
                    "route": parts[2].replace("ex:", ""),
                }
            )
            if len(rows) >= route_limit:
                break

    if not rows:
        rows = [
            {
                "flat": flat_id,
                "relation_1": "isNearStop",
                "stop": stop.replace("ex:", ""),
                "relation_2": "",
                "route": "",
            }
            for stop in stop_ids
        ]
    return pd.DataFrame(rows)


def find_similar_flats(
    embeddings: dict,
    flats_df: pd.DataFrame,
    query_uri: str,
    top_n: int,
) -> pd.DataFrame:
    complete_flats = flats_df.dropna(
        subset=[
            "price",
            "rent_per_m2",
            "transit_score",
            "latitude",
            "longitude",
        ]
    )
    available = set(complete_flats["uri"])
    uris = [uri for uri in embeddings if uri in available]
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

    result = flats_df.set_index("uri").loc[result_uris].reset_index()
    result.insert(2, "embedding_distance", result_distances)

    distance_array = np.array(result_distances, dtype=float)
    if len(distance_array) == 1 or distance_array.max() == distance_array.min():
        similarity = np.ones(len(distance_array))
    else:
        similarity = 1.0 - (
            (distance_array - distance_array.min())
            / (distance_array.max() - distance_array.min())
        )
    result.insert(3, "similarity", similarity)

    query = flats_df.set_index("uri").loc[query_uri]
    result["price_difference"] = result["price"] - float(query["price"])
    result["rent_per_m2_difference"] = (
        result["rent_per_m2"] - float(query["rent_per_m2"])
    )
    result["transit_score_difference"] = (
        result["transit_score"] - float(query["transit_score"])
    )
    return result


def similarity_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "flat",
        "district",
        "similarity",
        "embedding_distance",
        "price",
        "price_difference",
        "rent_per_m2",
        "rent_per_m2_difference",
        "transit_score",
        "transit_score_difference",
    ]
    return df[columns].round(
        {
            "similarity": 3,
            "embedding_distance": 3,
            "price": 0,
            "price_difference": 0,
            "rent_per_m2": 2,
            "rent_per_m2_difference": 2,
            "transit_score": 0,
            "transit_score_difference": 0,
        }
    )


def district_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("district", as_index=False)
        .agg(
            flats=("flat", "count"),
            median_rent=("price", "median"),
            average_rent_per_m2=("rent_per_m2", "mean"),
            average_transit_score=("transit_score", "mean"),
            average_value_score=("value_score", "mean"),
        )
        .sort_values("district")
    )


def render_bar_chart(
    df: pd.DataFrame,
    category: str,
    values: list[str],
    height: int = 300,
) -> None:
    chart_data = df[[category, *values]].copy()
    for value in values:
        chart_data[value] = pd.to_numeric(chart_data[value], errors="coerce")
    chart_data = chart_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[category, *values]
    )
    if chart_data.empty:
        st.info("No chartable values for this selection.")
        return

    labels = chart_data[category].astype(str).str.replace("district_", "", regex=False)
    x = np.arange(len(chart_data))
    width = 0.78 / max(len(values), 1)
    figure_height = max(2.6, height / 100)
    figure_width = max(6.0, min(14.0, len(chart_data) * 0.55))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=120)
    colors = ["#43a2ca", "#f28e5b", "#4fb286", "#d06cab"]

    for index, value in enumerate(values):
        offset = (index - (len(values) - 1) / 2) * width
        ax.bar(
            x + offset,
            chart_data[value],
            width=width,
            label=value.replace("_", " ").title(),
            color=colors[index % len(colors)],
        )

    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#3b4351", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    if len(values) > 1:
        ax.legend(frameon=False, ncol=min(3, len(values)), loc="upper left")
    style_matplotlib_chart(fig, ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    color_type: str = "N",
    height: int = 430,
) -> None:
    chart_data = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[x, y, color]
    )
    if chart_data.empty:
        st.info("No chartable values for this selection.")
        return

    fig, ax = plt.subplots(figsize=(11.5, max(3.2, height / 100)), dpi=120)
    if color_type == "Q":
        color_values = pd.to_numeric(chart_data[color], errors="coerce")
        points = ax.scatter(
            chart_data[x],
            chart_data[y],
            c=color_values,
            cmap="viridis",
            s=20,
            alpha=0.58,
            linewidths=0,
        )
        colorbar = fig.colorbar(points, ax=ax, pad=0.01)
        colorbar.set_label(color.replace("_", " ").title())
        colorbar.ax.tick_params(colors="#d7dce2")
        colorbar.outline.set_edgecolor("#596273")
    else:
        categories = pd.Categorical(chart_data[color].astype(str))
        ax.scatter(
            chart_data[x],
            chart_data[y],
            c=categories.codes,
            cmap="tab20",
            s=20,
            alpha=0.58,
            linewidths=0,
        )

    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.grid(color="#3b4351", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    style_matplotlib_chart(fig, ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def style_matplotlib_chart(fig, ax) -> None:
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#151a22")
    ax.tick_params(colors="#d7dce2", labelsize=8)
    ax.xaxis.label.set_color("#d7dce2")
    ax.yaxis.label.set_color("#d7dce2")
    for spine in ax.spines.values():
        spine.set_color("#596273")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("#d7dce2")


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

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }
        h1 {
            font-size: 2.5rem !important;
            line-height: 1.12 !important;
            letter-spacing: 0 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
        @media (max-width: 700px) {
            h1 {
                font-size: 1.9rem !important;
            }
            .block-container {
                padding-top: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Vienna Transit-Aware Flat Price Knowledge Graph")

    attrs = load_attributes()
    districts = flat_district_map()
    flats_df = flat_dataframe(attrs, districts)
    flat_uris = sorted(attrs)

    (
        tab_overview,
        tab_map,
        tab_deals,
        tab_analysis,
        tab_models,
        tab_flat,
        tab_similar,
        tab_clusters,
        tab_price,
    ) = st.tabs(
        [
            "KG Overview",
            "Vienna Map",
            "Deal Finder",
            "Transit Analysis",
            "Model Comparison",
            "Flat Explorer",
            "Similar Flats",
            "Embedding Space",
            "Price Prediction",
        ]
    )

    with tab_overview:
        counts = ttl_relation_counts()
        dataset_summary_path = RESULT_DIR / "kge_dataset_summary.json"
        summary = load_json(str(dataset_summary_path)) if dataset_summary_path.exists() else {}
        core_links = sum(max(value - 1, 0) for value in counts.values())

        cols = st.columns(4)
        cols[0].metric("Flats", f"{len(attrs):,}")
        cols[1].metric(
            "Triples Used",
            f"{summary.get('object_triples_used', core_links):,}",
        )
        cols[2].metric(
            "Entities",
            f"{summary.get('entities', 9_503):,}",
        )
        cols[3].metric(
            "Relations",
            f"{summary.get('relations', 7):,}",
        )

        st.dataframe(
            pd.DataFrame(
                [{"relation": key, "count": value} for key, value in counts.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

        cols = st.columns(4)
        cols[0].metric("Median rent", f"EUR {flats_df['price'].median():.0f}")
        cols[1].metric("Median rent/m2", f"EUR {flats_df['rent_per_m2'].median():.2f}")
        cols[2].metric("Median transit score", f"{flats_df['transit_score'].median():.0f}")
        cols[3].metric("Districts", f"{flats_df['district'].nunique():,}")

    with tab_map:
        prediction_files = sorted(PREDICTION_DIR.glob("*_price_predictions.csv"))

        col_a, col_b, col_c, col_d = st.columns([1.3, 1, 1, 1])
        color_by = col_a.selectbox(
            "Color",
            ["Price", "Rent per m2", "Transit score", "Transit value", "Prediction error"],
        )
        max_price = col_b.slider(
            "Max rent",
            int(flats_df["price"].min()),
            int(flats_df["price"].max()),
            int(flats_df["price"].quantile(0.90)),
        )
        min_transit = col_c.slider(
            "Min transit score",
            int(flats_df["transit_score"].min()),
            int(flats_df["transit_score"].max()),
            int(flats_df["transit_score"].quantile(0.25)),
        )
        districts_selected = col_d.multiselect(
            "District",
            sorted(flats_df["district"].dropna().unique()),
            default=[],
        )

        map_df = flats_df[
            (flats_df["price"] <= max_price)
            & (flats_df["transit_score"] >= min_transit)
        ].copy()
        if districts_selected:
            map_df = map_df[map_df["district"].isin(districts_selected)]

        selected_prediction = None
        if color_by == "Prediction error" and prediction_files:
            selected_prediction = st.selectbox(
                "Prediction source",
                prediction_files,
                format_func=lambda path: path.name,
            )
            map_df = add_prediction_columns(map_df, selected_prediction)
        elif color_by == "Prediction error":
            st.info("Run evaluate_price.py to color the map by prediction error.")

        cols = st.columns(4)
        cols[0].metric("Visible flats", f"{len(map_df):,}")
        cols[1].metric("Average rent", f"EUR {map_df['price'].mean():.0f}" if len(map_df) else "n/a")
        cols[2].metric("Average rent/m2", f"EUR {map_df['rent_per_m2'].mean():.2f}" if len(map_df) else "n/a")
        cols[3].metric("Average transit", f"{map_df['transit_score'].mean():.1f}" if len(map_df) else "n/a")

        render_vienna_map(map_df, color_by)

        st.dataframe(
            display_flat_table(map_df.sort_values("value_score", ascending=False), limit=25),
            use_container_width=True,
            hide_index=True,
        )

    with tab_deals:
        st.subheader("Transit-aware value candidates")
        col_a, col_b, col_c = st.columns(3)
        min_score = col_a.slider(
            "Minimum transit score",
            int(flats_df["transit_score"].min()),
            int(flats_df["transit_score"].max()),
            int(flats_df["transit_score"].median()),
            key="deal_min_transit",
        )
        max_rent_m2 = col_b.slider(
            "Maximum rent per m2",
            round(float(flats_df["rent_per_m2"].quantile(0.05)), 1),
            round(float(flats_df["rent_per_m2"].quantile(0.95)), 1),
            round(float(flats_df["rent_per_m2"].median()), 1),
            step=0.1,
        )
        top_n_deals = col_c.slider("Show top", 5, 100, 25)

        deals = flats_df[
            (flats_df["transit_score"] >= min_score)
            & (flats_df["rent_per_m2"] <= max_rent_m2)
        ].sort_values("value_score", ascending=False)

        st.dataframe(
            display_flat_table(deals, limit=top_n_deals),
            use_container_width=True,
            hide_index=True,
        )
        render_vienna_map(deals.head(top_n_deals), "Transit value", height=440)

    with tab_analysis:
        analysis_df = flats_df.dropna(
            subset=["rent_per_m2", "transit_score", "district"]
        ).copy()

        col_a, col_b = st.columns([1, 2])
        price_measure = col_a.selectbox(
            "Price measure",
            ["Rent per m2", "Monthly rent"],
        )
        selected_districts = col_b.multiselect(
            "Districts",
            sorted(analysis_df["district"].unique()),
            default=[],
            key="analysis_districts",
        )
        if selected_districts:
            analysis_df = analysis_df[
                analysis_df["district"].isin(selected_districts)
            ]

        y_column = "rent_per_m2" if price_measure == "Rent per m2" else "price"
        lower = analysis_df[y_column].quantile(0.01)
        upper = analysis_df[y_column].quantile(0.99)
        chart_df = analysis_df[
            analysis_df[y_column].between(lower, upper)
        ].copy()
        chart_df["point_size"] = 28

        correlation = chart_df["transit_score"].corr(chart_df[y_column])
        cols = st.columns(4)
        cols[0].metric("Flats analysed", f"{len(chart_df):,}")
        cols[1].metric("Transit-price correlation", f"{correlation:.3f}")
        cols[2].metric(
            f"Median {price_measure.lower()}",
            f"EUR {chart_df[y_column].median():.2f}",
        )
        cols[3].metric(
            "Median transit score",
            f"{chart_df['transit_score'].median():.0f}",
        )

        render_scatter_chart(
            chart_df,
            "transit_score",
            y_column,
            "district",
        )

        band_labels = ["Low", "Medium", "High", "Very high"]
        band_count = min(4, len(chart_df))
        if band_count >= 2:
            chart_df["transit_band"] = pd.qcut(
                chart_df["transit_score"].rank(method="first"),
                q=band_count,
                labels=band_labels[:band_count],
            )
        else:
            chart_df["transit_band"] = "All"
        band_summary = (
            chart_df.groupby("transit_band", observed=True, as_index=False)
            .agg(
                flats=("flat", "count"),
                median_price=(y_column, "median"),
                average_price=(y_column, "mean"),
                median_transit=("transit_score", "median"),
            )
        )

        col_a, col_b = st.columns(2)
        col_a.markdown(f"**{price_measure} by transit-access band**")
        with col_a:
            render_bar_chart(
                band_summary,
                "transit_band",
                ["median_price", "average_price"],
            )
        col_b.markdown("**Transit-access bands**")
        col_b.dataframe(
            band_summary.round(
                {
                    "median_price": 2,
                    "average_price": 2,
                    "median_transit": 0,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        district_df = district_summary(analysis_df)
        col_a, col_b = st.columns(2)
        col_a.markdown("**Average rent per m2 by district**")
        with col_a:
            render_bar_chart(
                district_df,
                "district",
                ["average_rent_per_m2"],
            )
        col_b.markdown("**Average transit score by district**")
        with col_b:
            render_bar_chart(
                district_df,
                "district",
                ["average_transit_score"],
            )
        st.dataframe(
            district_df.round(
                {
                    "median_rent": 0,
                    "average_rent_per_m2": 2,
                    "average_transit_score": 1,
                    "average_value_score": 2,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_models:
        comparison_path = RESULT_DIR / "kge_model_comparison.csv"
        literal_comparison_path = RESULT_DIR / "literal_kge_model_comparison.csv"
        frames = []
        if comparison_path.exists():
            base_df = load_csv(str(comparison_path))
            base_df["model_family"] = "structural"
            frames.append(base_df)
        if literal_comparison_path.exists():
            literal_df = load_csv(str(literal_comparison_path))
            literal_df["model_family"] = "literal-aware"
            frames.append(literal_df)

        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
            columns = ["model", "mrr", "hits_at_1", "hits_at_3", "hits_at_10"]
            if "model_family" in df.columns:
                columns.insert(1, "model_family")
            st.dataframe(df[columns], use_container_width=True, hide_index=True)
            render_bar_chart(df, "model", ["mrr", "hits_at_10"])
        else:
            st.info("Run train_kge_models.py or train_literal_kge_models.py to generate model comparison results.")

    with tab_flat:
        selected = st.selectbox(
            "Flat",
            flat_uris,
            format_func=lambda uri: flat_label(uri, attrs),
        )
        selected_df = flats_df[flats_df["uri"] == selected]
        if not selected_df.empty:
            item = selected_df.iloc[0]
            cols = st.columns(5)
            cols[0].metric("Rent", f"EUR {item['price']:.0f}")
            cols[1].metric("Size", f"{item['size']:.0f} m2" if pd.notna(item["size"]) else "n/a")
            cols[2].metric("Rent/m2", f"EUR {item['rent_per_m2']:.2f}")
            cols[3].metric("Transit score", f"{item['transit_score']:.0f}")
            cols[4].metric("District", str(item["district"]).replace("district_", ""))

        st.dataframe(attributes_table(attrs, [selected]), use_container_width=True)

        paths = flat_kg_paths(selected)
        if not paths.empty:
            st.dataframe(paths, use_container_width=True, hide_index=True)

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
            col_a, col_b, col_c = st.columns([1, 2, 1])
            selected_embedding = col_a.selectbox(
                "Embedding model",
                embedding_files,
                format_func=model_display_name,
            )
            embeddings = load_json(str(selected_embedding))
            available = flats_df[flats_df["uri"].isin(embeddings)].dropna(
                subset=["rent_per_m2"]
            )
            available_flats = sorted(available["uri"])
            selected_flat = col_b.selectbox(
                "Query flat",
                available_flats,
                format_func=lambda uri: flat_label(uri, attrs),
            )
            top_n = col_c.slider("Similar flats", 3, 30, 10)

            query = flats_df.set_index("uri").loc[selected_flat]
            similar_df = find_similar_flats(
                embeddings,
                flats_df,
                selected_flat,
                top_n,
            )

            cols = st.columns(5)
            cols[0].metric("Model", model_display_name(selected_embedding))
            cols[1].metric("Selected rent", f"EUR {query['price']:.0f}")
            cols[2].metric("Selected rent/m2", f"EUR {query['rent_per_m2']:.2f}")
            cols[3].metric("Selected transit", f"{query['transit_score']:.0f}")
            cols[4].metric(
                "Same-district matches",
                f"{(similar_df['district'] == query['district']).sum()}/{len(similar_df)}",
            )

            render_similarity_map(query, similar_df)

            st.dataframe(
                similarity_comparison_table(similar_df),
                use_container_width=True,
                hide_index=True,
            )

            comparison = pd.concat(
                [
                    query.to_frame().T.assign(group="selected"),
                    similar_df.head(8).assign(group="similar"),
                ],
                ignore_index=True,
            )
            col_a, col_b = st.columns(2)
            col_a.markdown("**Rent per m2 comparison**")
            with col_a:
                render_bar_chart(
                    comparison,
                    "flat",
                    ["rent_per_m2"],
                    height=280,
                )
            col_b.markdown("**Transit score comparison**")
            with col_b:
                render_bar_chart(
                    comparison,
                    "flat",
                    ["transit_score"],
                    height=280,
                )

    with tab_clusters:
        embedding_files = sorted(EMBEDDING_DIR.glob("*_flat_embeddings.json"))
        if not embedding_files:
            st.info("Run train_kge_models.py to create flat embeddings.")
        else:
            from sklearn.decomposition import PCA

            selected_embedding = st.selectbox(
                "Embedding model",
                embedding_files,
                format_func=model_display_name,
                key="cluster_embedding",
            )
            color_field = st.selectbox(
                "Color embedding space by",
                ["price", "rent_per_m2", "transit_score", "value_score", "district"],
            )

            embeddings = load_json(str(selected_embedding))
            common = [uri for uri in embeddings if uri in set(flats_df["uri"])]
            matrix = np.array([embeddings[uri] for uri in common], dtype=float)
            coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

            cluster_df = flats_df.set_index("uri").loc[common].reset_index()
            cluster_df["pc1"] = coords[:, 0]
            cluster_df["pc2"] = coords[:, 1]
            cluster_df["point_size"] = 35
            cluster_df["color_value"] = cluster_df[color_field].astype(str) if color_field == "district" else cluster_df[color_field]

            render_scatter_chart(
                cluster_df,
                "pc1",
                "pc2",
                "color_value",
                color_type="N" if color_field == "district" else "Q",
            )

            st.dataframe(
                cluster_df[
                    [
                        "flat",
                        "district",
                        "price",
                        "rent_per_m2",
                        "transit_score",
                        "value_score",
                        "pc1",
                        "pc2",
                    ]
                ].round({"price": 0, "rent_per_m2": 2, "value_score": 2, "pc1": 3, "pc2": 3}).head(100),
                use_container_width=True,
                hide_index=True,
            )

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
                ).reset_index()
                render_bar_chart(
                    chart,
                    "model",
                    [column for column in chart.columns if column != "model"],
                )
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
