# Vienna Transit-Aware Rental Knowledge Graph

This project constructs a Knowledge Graph (KG) from Vienna rental listings
and public transport data. It compares several Knowledge Graph Embedding
(KGE) models through link prediction and evaluates whether the learned flat
embeddings are useful for downstream rent prediction and similarity search.

The implementation supports the course learning outcomes:

- **LO1:** understand and apply Knowledge Graph Embeddings.
- **LO7:** apply a system to create a Knowledge Graph.

## Quick Start

Python 3.11 is recommended.

```bash
python -m venv venv
```

On Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The app expects the processed KG files and generated experiment artifacts.
If `artifacts/` is missing, run the experiment commands below first.

## Reproduce the Experiments

Rebuild the KG when the required raw input data is available:

```bash
python conversion_v2.py
```

Train the structural KGE models:

```bash
python train_kge_models.py --models TransE ComplEx RotatE --epochs 120
```

Train the literal-aware model:

```bash
python train_literal_kge_models.py --epochs 120
```

Run the downstream rent-prediction experiments:

```bash
python evaluate_price.py
```

Evaluate nearest neighbours in every embedding space:

```bash
python evaluate_embedding_neighbors.py --top-k 10
```

Generated outputs are stored under:

```text
artifacts/embeddings/
artifacts/models/
artifacts/predictions/
artifacts/results/
```

PyKEEN uses CUDA automatically when a compatible CUDA-enabled PyTorch
installation and GPU are available. Training also works on CPU but takes
longer.

## Pipeline

1. `conversion_v2.py` combines rental listings and Vienna GTFS data.
2. Structural relations are written to `vienna_kg_entities.ttl`.
3. Numeric flat attributes are written to `vienna_kg_attributes.json`.
4. `train_kge_models.py` trains TransE, ComplEx, and RotatE for link
   prediction.
5. `train_literal_kge_models.py` trains ComplExLiteral using numeric
   attributes, excluding rent to prevent target leakage.
6. `evaluate_price.py` compares a mean baseline, attributes, embeddings, and
   attributes plus embeddings for rent prediction.
7. `evaluate_embedding_neighbors.py` evaluates what nearby flats in each
   embedding space have in common.
8. `app.py` presents the KG, model results, Vienna map, flat explorer, and
   embedding-space analysis in Streamlit.

## Repository Structure

```text
app.py                             Streamlit application
conversion_v2.py                   KG construction pipeline
train_kge_models.py                TransE, ComplEx, and RotatE training
train_literal_kge_models.py        ComplExLiteral training
evaluate_price.py                  Downstream rent regression
evaluate_embedding_neighbors.py    Similar-flat evaluation
vienna_kg_entities.ttl             Structural Knowledge Graph
vienna_kg_attributes.json          Numeric attributes and regression target
flat_info.json                     Rental-listing input snapshot
gtfs/                              Local Vienna public transport input files
artifacts/                         Generated models, embeddings, and results
```

`artifacts/` is generated locally and intentionally ignored by Git.

## Data Requirements

KG construction expects:

```text
flat_info.json
gtfs/stops.txt
gtfs/routes.txt
gtfs/trips.txt
gtfs/stop_times.txt
```

`flat_info.json` is a local rental-listing snapshot collected from publicly
visible Willhaben advertisements for this academic mini-project.

The large GTFS file `gtfs/stop_times.txt` is intentionally not committed
because it is approximately 402 MB in the dataset used for this project.
Committing this single generated data file would make cloning and submitting
the source-code repository impractical.

Download one consistent Vienna GTFS dataset and place the required files at
the paths above before rebuilding the KG. See `gtfs/README.md` for the local
directory layout. The processed KG files are included, so rebuilding is not
required merely to inspect the graph or project source.

## Evaluation Design

The KGE models are trained for link prediction, measured with MRR, Hits@1,
and Hits@10. Rent prediction is a separate supervised downstream experiment
measured with R-squared and MAE.

ComplExLiteral receives rooms, size, floor, location quality, transit score,
latitude, and longitude as numeric literals. Rent is deliberately excluded
from KGE training, ensuring that price-prediction and neighbour results do
not contain target leakage.
