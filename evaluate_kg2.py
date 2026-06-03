"""
train_embeddings.py
-------------------
Trains TransE on vienna_kg_entities.ttl (no numeric literals),
then runs three regression experiments:

  Experiment 1: Transit embeddings only          -> R², MAE
  Experiment 2: Flat attributes only             -> R², MAE
  Experiment 3: Embeddings + attributes combined -> R², MAE

Usage:
    python train_embeddings.py

Inputs:
    vienna_kg_entities.ttl
    vienna_kg_attributes.json

Outputs:
    models/transe/          (saved PyKEEN model)
    results_summary.txt
"""

import json
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TTL_FILE    = "./vienna_kg_entities.ttl"
ATTRS_FILE  = "./vienna_kg_attributes.json"
MODEL_DIR   = "./models/transe"

EMBEDDING_DIM = 100
NUM_EPOCHS    = 300
BATCH_SIZE    = 512
LEARNING_RATE = 0.01

# ---------------------------------------------------------------------------
# 0. CHECK GPU
# ---------------------------------------------------------------------------

import torch

if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    print(f"Using GPU for training")
else:
    device = "cpu"
    print("No GPU detected, using CPU")

# ---------------------------------------------------------------------------
# 1. PARSE TTL INTO TRIPLES
# ---------------------------------------------------------------------------

print("\nStep 1: Parsing vienna_kg_entities.ttl ...")

from rdflib import Graph

g = Graph()
g.parse(TTL_FILE, format="turtle")
print(f"  Loaded {len(g)} triples")

triples = []
for s, p, o in g:
    triples.append((str(s), str(p), str(o)))

print(f"  Total triples for PyKEEN: {len(triples)}")

# ---------------------------------------------------------------------------
# 2. LOAD ATTRIBUTES
# ---------------------------------------------------------------------------

print("\nStep 2: Loading flat attributes...")

with open(ATTRS_FILE, encoding="utf-8") as f:
    attributes = json.load(f)

print(f"  Flats with attributes: {len(attributes)}")

# Extract prices for regression
flat_prices = {uri: attrs["price"]
               for uri, attrs in attributes.items()
               if attrs.get("price") is not None}

print(f"  Flats with price: {len(flat_prices)}")

# ---------------------------------------------------------------------------
# 3. BUILD PYKEEN DATASET
# ---------------------------------------------------------------------------

print("\nStep 3: Building PyKEEN dataset...")

import numpy as np_arr
from pykeen.triples import TriplesFactory

triples_array = np.array(triples, dtype=str)
tf = TriplesFactory.from_labeled_triples(triples_array)
train_tf, valid_tf, test_tf = tf.split([0.8, 0.1, 0.1], random_state=42)

print(f"  Train: {train_tf.num_triples}")
print(f"  Valid: {valid_tf.num_triples}")
print(f"  Test : {test_tf.num_triples}")
print(f"  Entities  : {tf.num_entities}")
print(f"  Relations : {tf.num_relations}")

# ---------------------------------------------------------------------------
# 4. TRAIN TRANSE
# ---------------------------------------------------------------------------

print(f"\nStep 4: Training TransE on {device.upper()}...")
print(f"  Embedding dim : {EMBEDDING_DIM}")
print(f"  Epochs        : {NUM_EPOCHS}")
print(f"  Batch size    : {BATCH_SIZE}")

from pykeen.pipeline import pipeline

result = pipeline(
    training=train_tf,
    validation=valid_tf,
    testing=test_tf,
    model="TransE",
    model_kwargs=dict(
        embedding_dim=EMBEDDING_DIM,
        scoring_fct_norm=1,
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=LEARNING_RATE),
    training_kwargs=dict(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    ),
    loss="MarginRankingLoss",
    loss_kwargs=dict(margin=1.0),
    random_seed=42,
    device=device,
    stopper="early",
    stopper_kwargs=dict(
        frequency=10,
        patience=5,
        relative_delta=0.002,
    ),
)

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
result.save_to_directory(MODEL_DIR)
print(f"  Model saved to {MODEL_DIR}/")

# ---------------------------------------------------------------------------
# 5. EXTRACT FLAT EMBEDDINGS
# ---------------------------------------------------------------------------

print("\nStep 5: Extracting flat embeddings...")

model = result.model
entity_to_id = tf.entity_to_id

with torch.no_grad():
    all_embeddings = model.entity_representations[0](indices=None).cpu().numpy()

flat_embeddings = {}
for uri, idx in entity_to_id.items():
    if "flat_" in uri:
        flat_embeddings[uri] = all_embeddings[idx]

print(f"  Flat embeddings extracted: {len(flat_embeddings)}")

# Ghost entity check
ghost_count = 0
for entity in entity_to_id:
    label = entity.split('/')[-1]
    try:
        float(label)
        ghost_count += 1
    except ValueError:
        pass

if ghost_count > 0:
    print(f"  WARNING: {ghost_count} ghost entities still detected!")
else:
    print(f"  Ghost entity check: PASSED (0 numeric ghost entities)")

# ---------------------------------------------------------------------------
# 6. LINK PREDICTION EVALUATION
# ---------------------------------------------------------------------------

print("\nStep 6: Link prediction evaluation...")

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

mrr    = metrics.get_metric("mean_reciprocal_rank")
hits1  = metrics.get_metric("hits_at_1")
hits10 = metrics.get_metric("hits_at_10")

print(f"  MRR     : {mrr:.4f}")
print(f"  Hits@1  : {hits1:.4f}")
print(f"  Hits@10 : {hits10:.4f}")

# ---------------------------------------------------------------------------
# 7. THREE REGRESSION EXPERIMENTS
# ---------------------------------------------------------------------------

print("\nStep 7: Running 3 regression experiments...")

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def run_regression(X, y, name):
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg     = Ridge(alpha=1.0)
    r2      = cross_val_score(reg, X_scaled, y, cv=5, scoring="r2")
    mae     = cross_val_score(reg, X_scaled, y, cv=5,
                               scoring="neg_mean_absolute_error")
    print(f"\n  [{name}]")
    print(f"    R²  : {r2.mean():.3f}  (±{r2.std():.3f})")
    print(f"    MAE : {-mae.mean():.1f} EUR  (±{mae.std():.1f})")
    return r2.mean(), r2.std(), -mae.mean(), mae.std()


# Match flats with both embedding and attributes
common = [uri for uri in flat_embeddings
          if uri in flat_prices and uri in attributes]

print(f"  Flats used in regression: {len(common)}")

y = np.array([flat_prices[uri] for uri in common])

# --- Experiment 1: Embeddings only ---
X_emb = np.array([flat_embeddings[uri] for uri in common])
r2_e, r2_e_std, mae_e, mae_e_std = run_regression(X_emb, y, "Experiment 1: Embeddings only")

# --- Experiment 2: Attributes only ---
attr_keys = ["rooms", "size", "floor", "location_quality", "transit_score"]

def get_attr_vector(uri):
    a = attributes[uri]
    return [a.get(k) or 0.0 for k in attr_keys]

X_attr = np.array([get_attr_vector(uri) for uri in common])
r2_a, r2_a_std, mae_a, mae_a_std = run_regression(X_attr, y, "Experiment 2: Attributes only")

# --- Experiment 3: Embeddings + Attributes combined ---
X_combined = np.hstack([X_emb, X_attr])
r2_c, r2_c_std, mae_c, mae_c_std = run_regression(
    X_combined, y, "Experiment 3: Embeddings + Attributes combined")

# ---------------------------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------------------------

summary = f"""
Vienna KG — TransE Embedding Results
=====================================

Dataset
-------
TTL file        : {TTL_FILE}
Total triples   : {len(triples)}
Entities        : {tf.num_entities}
Relations       : {tf.num_relations}
Flats in KG     : {len(flat_embeddings)}
Ghost entities  : {ghost_count}

Model: TransE on {device.upper()}
  Embedding dim : {EMBEDDING_DIM}
  Epochs        : {NUM_EPOCHS} (with early stopping)
  Batch size    : {BATCH_SIZE}
  Optimizer     : Adam (lr={LEARNING_RATE})

Link Prediction (test set)
---------------------------
MRR             : {mrr:.4f}
Hits@1          : {hits1:.4f}
Hits@10         : {hits10:.4f}

Price Prediction (5-fold cross-val, {len(common)} flats)
---------------------------------------------------------
Experiment 1 — Embeddings only
  R²  : {r2_e:.3f} (±{r2_e_std:.3f})
  MAE : {mae_e:.1f} EUR (±{mae_e_std:.1f})

Experiment 2 — Attributes only (rooms, size, floor, location quality, transit score)
  R²  : {r2_a:.3f} (±{r2_a_std:.3f})
  MAE : {mae_a:.1f} EUR (±{mae_a_std:.1f})

Experiment 3 — Embeddings + Attributes combined
  R²  : {r2_c:.3f} (±{r2_c_std:.3f})
  MAE : {mae_c:.1f} EUR (±{mae_c_std:.1f})

Interpretation
--------------
Embedding contribution = R²(Exp3) - R²(Exp2) = {r2_c - r2_a:.3f}
"""

with open("results_summary.txt", "w") as f:
    f.write(summary)

print("\n" + summary)
print("Saved to results_summary.txt")
print("\nAll done!")