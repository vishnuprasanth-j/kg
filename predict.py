"""
predict_kg.py
-------------
Interactive prediction and similarity search for the Vienna KG.

Usage:
    python predict_kg.py

Commands:
    similar   - Find similar flats by embedding distance
    predict   - Link prediction (flat -> isNearStop -> ???)
    explore   - Show all triples for a flat
    check     - Run ghost entity sanity check
    exit      - Quit
"""

import json
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_DIR = "./models/transe"
TTL_FILE  = "./vienna_kg.ttl"

# ---------------------------------------------------------------------------
# 1. LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(model_dir):
    print("Loading model...")
    model_dir = Path(model_dir)

    # PyKEEN's official way to load a saved pipeline
    from pykeen.triples import TriplesFactory
    
    model = torch.load(
        model_dir / "trained_model.pkl",
        map_location="cpu",
        weights_only=False
    )
    model.eval()

    # Load triples factory the official PyKEEN way
    tf = TriplesFactory.from_path_binary(
        model_dir / "training_triples"
    )

    entity_to_id  = tf.entity_to_id
    id_to_entity  = {v: k for k, v in entity_to_id.items()}
    relation_to_id = tf.relation_to_id

    # Extract embeddings
    with torch.no_grad():
        all_embeddings = model.entity_representations[0](indices=None).cpu().numpy()

    print(f"  Entities       : {len(entity_to_id)}")
    print(f"  Relations      : {len(relation_to_id)}")
    print(f"  Embedding shape: {all_embeddings.shape}")

    return model, tf, all_embeddings, entity_to_id, id_to_entity, relation_to_id


# ---------------------------------------------------------------------------
# 2. SANITY CHECK
# ---------------------------------------------------------------------------

def sanity_check(entity_to_id):
    print("\n=== Sanity Check: Ghost Entity Detection ===")
    ghosts = []
    for entity in entity_to_id:
        label = entity.split('/')[-1]
        try:
            float(label)
            ghosts.append(label)
        except ValueError:
            pass

    if ghosts:
        print(f"  WARNING: Found {len(ghosts)} numeric ghost entities!")
        print(f"  Sample ghost entities: {ghosts[:15]}")
        print()
        print("  These are numeric literals (price, rooms, size etc.)")
        print("  being treated as real entities by PyKEEN.")
        print("  Same bug Christopher found: '0' appeared similar to Vienna.")
        print("  Fix: remove numeric literals from TTL -> move to attributes JSON.")
    else:
        print("  No ghost entities found. Your KG is clean!")
    print()


# ---------------------------------------------------------------------------
# 3. SIMILARITY SEARCH
# ---------------------------------------------------------------------------

def find_similar(query, entity_to_id, id_to_entity, all_embeddings, top_n=10):
    # Match entity
    matches = [e for e in entity_to_id if query.lower() in e.lower()]
    if not matches:
        print(f"  No entity matching '{query}'")
        return

    # Prefer flat matches
    flat_matches = [m for m in matches if "flat_" in m]
    if flat_matches:
        matches = flat_matches

    entity_uri = matches[0]
    entity_id  = entity_to_id[entity_uri]
    entity_emb = all_embeddings[entity_id]

    distances  = np.linalg.norm(all_embeddings - entity_emb, axis=1)
    sorted_ids = np.argsort(distances)

    print(f"\nTop {top_n} most similar to: {entity_uri.split('/')[-1]}")
    print("-" * 60)

    shown = 0
    for idx in sorted_ids:
        if idx == entity_id:
            continue
        label = id_to_entity[idx].split('/')[-1]
        dist  = distances[idx]

        # Flag ghost entities
        ghost = ""
        try:
            float(label)
            ghost = "  ← GHOST ENTITY (numeric literal)"
        except ValueError:
            pass

        print(f"  [{shown+1}] {label:40s}  dist={dist:.4f}{ghost}")
        shown += 1
        if shown >= top_n:
            break


# ---------------------------------------------------------------------------
# 4. LINK PREDICTION
# ---------------------------------------------------------------------------

def predict_links(head_query, rel_query, entity_to_id, id_to_entity,
                  relation_to_id, model, top_n=10):

    # Match head
    head_matches = [e for e in entity_to_id if head_query.lower() in e.lower()]
    if not head_matches:
        print(f"  No entity matching '{head_query}'")
        return
    head_uri = head_matches[0]
    head_id  = entity_to_id[head_uri]

    # Match relation
    rel_matches = [r for r in relation_to_id if rel_query.lower() in r.lower()]
    if not rel_matches:
        print(f"  No relation matching '{rel_query}'")
        print("  Available relations:")
        for r in relation_to_id:
            print(f"    {r.split('/')[-1]}")
        return
    rel_uri = rel_matches[0]
    rel_id  = relation_to_id[rel_uri]

    print(f"\nPredicting: {head_uri.split('/')[-1]} -> {rel_uri.split('/')[-1]} -> ???")
    print("-" * 60)

    n = len(entity_to_id)
    heads  = torch.tensor([head_id] * n)
    rels   = torch.tensor([rel_id]  * n)
    tails  = torch.tensor(list(range(n)))
    triples = torch.stack([heads, rels, tails], dim=1)

    with torch.no_grad():
        scores = model.score_hrt(triples).cpu().numpy().flatten()

    sorted_ids = np.argsort(scores)[::-1]

    for i, idx in enumerate(sorted_ids[:top_n]):
        label = id_to_entity[idx].split('/')[-1]
        ghost = ""
        try:
            float(label)
            ghost = "  ← GHOST ENTITY"
        except ValueError:
            pass
        print(f"  [{i+1}] {label:40s}  score={scores[idx]:.4f}{ghost}")


# ---------------------------------------------------------------------------
# 5. EXPLORE FLAT
# ---------------------------------------------------------------------------

def explore_flat(query, entity_to_id):
    from rdflib import Graph, URIRef

    matches = [e for e in entity_to_id if query.lower() in e.lower()
               and "flat_" in e.lower()]
    if not matches:
        print(f"  No flat matching '{query}'")
        return

    flat_uri = matches[0]
    print(f"\nLoading {TTL_FILE}...")
    g = Graph()
    g.parse(TTL_FILE, format="turtle")

    print(f"\nAll triples for: {flat_uri.split('/')[-1]}")
    print("-" * 60)
    for s, p, o in g.triples((URIRef(flat_uri), None, None)):
        pred = str(p).split('/')[-1]
        obj  = str(o).split('/')[-1]
        print(f"  {pred:30s}  ->  {obj}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    model, tf, all_embeddings, entity_to_id, id_to_entity, relation_to_id = load_model(MODEL_DIR)

    # Auto sanity check on startup
    sanity_check(entity_to_id)

    print("Commands: similar | predict | explore | check | exit")
    print("=" * 60)

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd == "exit":
            break

        elif cmd == "check":
            sanity_check(entity_to_id)

        elif cmd == "similar":
            q = input("  Entity (e.g. flat_0): ").strip()
            try:
                n = int(input("  How many? (default 10): ").strip() or 10)
            except ValueError:
                n = 10
            find_similar(q, entity_to_id, id_to_entity, all_embeddings, n)

        elif cmd == "predict":
            head = input("  Head entity (e.g. flat_0): ").strip()
            rel  = input("  Relation (e.g. isNearStop): ").strip()
            try:
                n = int(input("  How many? (default 10): ").strip() or 10)
            except ValueError:
                n = 10
            predict_links(head, rel, entity_to_id, id_to_entity,
                          relation_to_id, model, n)

        elif cmd == "explore":
            q = input("  Flat (e.g. flat_0): ").strip()
            explore_flat(q, entity_to_id)

        else:
            print("  Commands: similar | predict | explore | check | exit")


if __name__ == "__main__":
    main()