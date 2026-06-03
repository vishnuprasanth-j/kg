"""
train_embeddings.py
-------------------
Loads vienna_kg.ttl, trains a TransE embedding model using PyKEEN,
then uses the flat embeddings to predict prices via linear regression.

Usage:
    pip install pykeen rdflib scikit-learn numpy pandas matplotlib
    python train_embeddings.py

Output:
    - models/transe/  (saved PyKEEN model)
    - flat_embeddings.json  (embedding vector per flat)
    - results_summary.txt   (evaluation metrics)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PARSE THE TURTLE FILE INTO TRIPLES
# ---------------------------------------------------------------------------

print("Step 1: Parsing vienna_kg.ttl ...")

from rdflib import Graph

g = Graph()
g.parse("vienna_kg.ttl", format="turtle")
print(f"  Loaded {len(g)} triples from vienna_kg.ttl")

# Convert RDF graph to list of (subject, predicate, object) string triples
# PyKEEN works with string-labeled triples
triples = []
for s, p, o in g:
    triples.append((str(s), str(p), str(o)))

print(f"  Total triples for PyKEEN: {len(triples)}")

# ---------------------------------------------------------------------------
# 2. EXTRACT FLAT PRICES (we need these for regression later)
# ---------------------------------------------------------------------------

print("\nStep 2: Extracting flat prices from graph ...")

PRICE_PRED = "http://example.org/viennakg/price"

flat_prices = {}
for s, p, o in g:
    if str(p) == PRICE_PRED:
        try:
            flat_prices[str(s)] = float(str(o))
        except ValueError:
            continue

print(f"  Found prices for {len(flat_prices)} flats")

# ---------------------------------------------------------------------------
# 3. BUILD PYKEEN DATASET FROM TRIPLES
# ---------------------------------------------------------------------------

print("\nStep 3: Building PyKEEN dataset ...")

import torch
from pykeen.triples import TriplesFactory

# Convert to numpy array of shape (N, 3)
triples_array = np.array(triples, dtype=str)

# Split into train / validation / test  (80 / 10 / 10)
tf = TriplesFactory.from_labeled_triples(triples_array)

train_tf, valid_tf, test_tf = tf.split([0.8, 0.1, 0.1], random_state=42)

print(f"  Train triples : {train_tf.num_triples}")
print(f"  Valid triples : {valid_tf.num_triples}")
print(f"  Test  triples : {test_tf.num_triples}")
print(f"  Entities      : {tf.num_entities}")
print(f"  Relations     : {tf.num_relations}")

# ---------------------------------------------------------------------------
# 4. TRAIN TRANSE
# ---------------------------------------------------------------------------

print("\nStep 4: Training TransE model ...")
print("  (this may take a few minutes)")

from pykeen.pipeline import pipeline

result = pipeline(
    training=train_tf,
    validation=valid_tf,
    testing=test_tf,
    model="TransE",
    model_kwargs=dict(
        embedding_dim=50,       # 50-dimensional vectors
        scoring_fct_norm=1,     # L1 norm (standard for TransE)
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.01),
    training_kwargs=dict(
        num_epochs=100,
        batch_size=256,
    ),
    loss="MarginRankingLoss",
    loss_kwargs=dict(margin=1.0),
    random_seed=42,
    device="cpu",
)

# Save the model
Path("models/transe").mkdir(parents=True, exist_ok=True)
result.save_to_directory("models/transe")
print("  Model saved to models/transe/")

# ---------------------------------------------------------------------------
# 5. EXTRACT FLAT EMBEDDINGS
# ---------------------------------------------------------------------------

print("\nStep 5: Extracting flat embeddings ...")

# Get the entity embedding matrix
model = result.model
entity_representation = model.entity_representations[0]
all_embeddings = entity_representation(indices=None).detach().cpu().numpy()

# Map entity labels to their indices in PyKEEN
entity_to_id = train_tf.entity_to_id  # use the full factory
# actually use tf (full) for entity mapping
entity_to_id = tf.entity_to_id

flat_embeddings = {}
for entity_uri, idx in entity_to_id.items():
    if "flat_" in entity_uri:
        flat_embeddings[entity_uri] = all_embeddings[idx].tolist()

print(f"  Extracted embeddings for {len(flat_embeddings)} flats")

# Save embeddings to JSON
with open("flat_embeddings.json", "w") as f:
    json.dump(flat_embeddings, f)
print("  Saved to flat_embeddings.json")

# ---------------------------------------------------------------------------
# 6. PRICE PREDICTION VIA LINEAR REGRESSION
# ---------------------------------------------------------------------------

print("\nStep 6: Price prediction using flat embeddings ...")

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Match flats that have both an embedding and a price
common_flats = [uri for uri in flat_embeddings if uri in flat_prices]
print(f"  Flats with both embedding and price: {len(common_flats)}")

if len(common_flats) < 10:
    print("  WARNING: Very few flats — results may not be reliable")

X = np.array([flat_embeddings[uri] for uri in common_flats])
y = np.array([flat_prices[uri] for uri in common_flats])

# Normalise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge regression with cross-validation
reg = Ridge(alpha=1.0)
cv_r2 = cross_val_score(reg, X_scaled, y, cv=5, scoring="r2")
cv_mae = cross_val_score(reg, X_scaled, y, cv=5,
                         scoring="neg_mean_absolute_error")

print(f"\n  === Price Prediction Results ===")
print(f"  R² (cross-val mean) : {cv_r2.mean():.3f}  (±{cv_r2.std():.3f})")
print(f"  MAE (cross-val mean): {-cv_mae.mean():.1f} EUR  (±{cv_mae.std():.1f})")

# ---------------------------------------------------------------------------
# 7. LINK PREDICTION EVALUATION (standard KGE metric)
# ---------------------------------------------------------------------------

print("\nStep 7: Link prediction evaluation ...")

from pykeen.evaluation import RankBasedEvaluator

evaluator = RankBasedEvaluator()
metrics = evaluator.evaluate(
    model=result.model,
    mapped_triples=test_tf.mapped_triples,
    additional_filter_triples=[
        train_tf.mapped_triples,
        valid_tf.mapped_triples,
    ],
)

hits1  = metrics.get_metric("hits_at_1")
hits10 = metrics.get_metric("hits_at_10")
mrr    = metrics.get_metric("mean_reciprocal_rank")

print(f"\n  === Link Prediction Results ===")
print(f"  MRR        : {mrr:.4f}")
print(f"  Hits@1     : {hits1:.4f}")
print(f"  Hits@10    : {hits10:.4f}")

# ---------------------------------------------------------------------------
# 8. SAVE RESULTS SUMMARY
# ---------------------------------------------------------------------------

summary = f"""
Vienna KG — TransE Embedding Results
=====================================

Dataset
-------
Total triples  : {len(triples)}
Entities       : {tf.num_entities}
Relations      : {tf.num_relations}
Flats with price: {len(common_flats)}

Model: TransE
  Embedding dim : 50
  Epochs        : 100
  Optimizer     : Adam (lr=0.01)

Link Prediction (on test set)
------------------------------
MRR            : {mrr:.4f}
Hits@1         : {hits1:.4f}
Hits@10        : {hits10:.4f}

Price Prediction (5-fold cross-val)
-------------------------------------
R²             : {cv_r2.mean():.3f} (±{cv_r2.std():.3f})
MAE            : {-cv_mae.mean():.1f} EUR (±{cv_mae.std():.1f})
"""

with open("results_summary.txt", "w") as f:
    f.write(summary)

print("\n" + summary)
print("Results saved to results_summary.txt")
print("\nAll done!")