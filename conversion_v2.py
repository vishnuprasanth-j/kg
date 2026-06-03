"""
convert_to_turtle.py
--------------------
Converts GTFS transit data + Willhaben flat listings into:
  1. vienna_kg_entities.ttl   → structural triples only (for PyKEEN)
  2. vienna_kg_attributes.json → numeric attributes (for regression)

Key design decisions:
  - Numeric literals (price, rooms, size, floor) are NOT in the TTL
    because TransE treats literals as entities, polluting embeddings.
    (See: Gesese et al. 2021, "A Survey on KG Embeddings with Literals")
  - Location quality is bucketed into categories (low/medium/high/very_high)
    so it can be used as a proper entity relation in the KG
  - District nodes are derived from postcode
  - Transit score (unique lines within 500m) stored in attributes JSON

Usage:
    python convert_to_turtle.py

Inputs:
    gtfs/stops.txt, routes.txt, trips.txt, stop_times.txt
    flat_info.json

Outputs:
    vienna_kg_entities.ttl
    vienna_kg_attributes.json
"""

import json
import csv
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GTFS_DIR             = "./gtfs"
FLAT_JSON            = "./flat_info.json"
OUTPUT_TTL           = "./vienna_kg_entities.ttl"
OUTPUT_ATTRS         = "./vienna_kg_attributes.json"

NEAR_STOP_DISTANCE_M = 500
MIN_PRICE            = 300
MAX_PRICE            = 5000

# ---------------------------------------------------------------------------
# NAMESPACES
# ---------------------------------------------------------------------------

PREFIXES = """\
@prefix ex:     <http://example.org/viennakg/> .
@prefix rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix owl:    <http://www.w3.org/2002/07/owl#> .

"""

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def safe_id(raw):
    return raw.replace(" ", "_").replace("/", "_").replace(":", "_") \
              .replace("(", "").replace(")", "").replace(",", "") \
              .replace(".", "_").replace("-", "_")


def esc(value):
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def postcode_to_district(postcode):
    """Extract Vienna district from postcode e.g. 1010 -> district_01"""
    try:
        pc = str(postcode).strip()
        if len(pc) == 4 and pc.startswith("1"):
            district_num = int(pc[1:3])
            return f"district_{district_num:02d}"
    except (ValueError, TypeError):
        pass
    return None


def bucket_location_quality(value):
    """
    Convert numeric location quality to categorical entity.
    Returns None if value is missing.
    Adjust buckets based on what Counter showed for your data.
    """
    try:
        v = float(str(value).split("||||")[0])
        if v <= 1:
            return "quality_very_low"
        elif v <= 2:
            return "quality_low"
        elif v <= 3:
            return "quality_medium"
        elif v <= 4:
            return "quality_high"
        else:
            return "quality_very_high"
    except (ValueError, TypeError):
        return None


ROUTE_TYPE_LABELS = {
    "0": "Tram", "1": "Metro", "2": "Rail",
    "3": "Bus",  "4": "Ferry", "5": "CableCar",
}

# ---------------------------------------------------------------------------
# 1. LOAD GTFS
# ---------------------------------------------------------------------------

print("Loading GTFS data...")
stops_raw      = read_csv(f"{GTFS_DIR}/stops.txt")
routes_raw     = read_csv(f"{GTFS_DIR}/routes.txt")
trips_raw      = read_csv(f"{GTFS_DIR}/trips.txt")
stop_times_raw = read_csv(f"{GTFS_DIR}/stop_times.txt")

trip_to_route = {t["trip_id"]: t["route_id"] for t in trips_raw}

print("Deriving stop->route relationships...")
stop_routes = defaultdict(set)
for st in stop_times_raw:
    route_id = trip_to_route.get(st["trip_id"])
    if route_id:
        stop_routes[st["stop_id"]].add(route_id)

stops = []
for s in stops_raw:
    try:
        stops.append({
            "id":   safe_id(s["stop_id"]),
            "raw":  s["stop_id"],
            "name": s["stop_name"],
            "lat":  float(s["stop_lat"]),
            "lon":  float(s["stop_lon"]),
        })
    except (ValueError, KeyError):
        continue

print(f"  Stops  : {len(stops)}")
print(f"  Routes : {len(routes_raw)}")

# ---------------------------------------------------------------------------
# 2. LOAD FLATS
# ---------------------------------------------------------------------------

print("Loading flat data...")
with open(FLAT_JSON, encoding="utf-8") as f:
    flats_raw = json.load(f)

flats = []
for i, flat in enumerate(flats_raw):
    try:
        lat = float(flat["LATITUDE"])
        lon = float(flat["LONGITUDE"])

        # Use monthly rent only
        price_raw = flat.get("RENT/PER_MONTH_LETTINGS") or flat.get("PRICE")
        if not price_raw:
            continue
        price = float(str(price_raw).split("||||")[0])

        # Filter unrealistic prices
        if price < MIN_PRICE or price > MAX_PRICE:
            continue

        # Rooms
        rooms_raw = flat.get("NUMBER_OF_ROOMS", "")
        rooms = None
        try:
            rooms = float(str(rooms_raw).split("||||")[0])
        except (ValueError, TypeError):
            pass

        # Size
        size_raw = flat.get("ESTATE_SIZE/LIVING_AREA") or flat.get("ESTATE_SIZE", "")
        size = None
        try:
            size = float(str(size_raw).split("||||")[0])
        except (ValueError, TypeError):
            pass

        # Floor
        floor_raw = flat.get("FLOOR", "")
        floor = None
        try:
            floor = float(str(floor_raw).split("||||")[0])
        except (ValueError, TypeError):
            pass

        # Location quality (raw number for attributes, bucketed for KG)
        lq_raw = flat.get("LOCATION_QUALITY", "")
        lq_num = None
        try:
            lq_num = float(str(lq_raw).split("||||")[0])
        except (ValueError, TypeError):
            pass
        lq_bucket = bucket_location_quality(lq_raw)

        # Postcode and district
        postcode = str(flat.get("POSTCODE", "")).split("||||")[0].strip()
        district = postcode_to_district(postcode)

        # Heading
        heading = str(flat.get("HEADING", "")).split("||||")[0].strip()

        flats.append({
            "id":        f"flat_{i}",
            "lat":       lat,
            "lon":       lon,
            "price":     price,
            "rooms":     rooms,
            "size":      size,
            "floor":     floor,
            "lq_num":    lq_num,
            "lq_bucket": lq_bucket,
            "postcode":  postcode,
            "district":  district,
            "heading":   heading,
        })
    except (ValueError, KeyError):
        continue

print(f"  Valid flats: {len(flats)}")

# ---------------------------------------------------------------------------
# 3. COMPUTE isNearStop + transit score
# ---------------------------------------------------------------------------

print(f"Computing isNearStop (threshold={NEAR_STOP_DISTANCE_M}m)...")

# For each flat: list of nearby stop IDs
flat_nearby_stops = {}
for flat in flats:
    nearby = []
    for stop in stops:
        dist = haversine_m(flat["lat"], flat["lon"], stop["lat"], stop["lon"])
        if dist <= NEAR_STOP_DISTANCE_M:
            nearby.append(stop["id"])
    flat_nearby_stops[flat["id"]] = nearby

# Compute transit score = unique lines reachable within 500m
flat_transit_scores = {}
for flat in flats:
    nearby_stop_ids_raw = []
    for stop in stops:
        dist = haversine_m(flat["lat"], flat["lon"], stop["lat"], stop["lon"])
        if dist <= NEAR_STOP_DISTANCE_M:
            nearby_stop_ids_raw.append(stop["raw"])

    unique_lines = set()
    for raw_stop_id in nearby_stop_ids_raw:
        unique_lines.update(stop_routes.get(raw_stop_id, set()))
    flat_transit_scores[flat["id"]] = len(unique_lines)

total_near = sum(len(v) for v in flat_nearby_stops.values())
print(f"  isNearStop triples: {total_near}")

# ---------------------------------------------------------------------------
# 4. WRITE TURTLE (entities only — no numeric literals)
# ---------------------------------------------------------------------------

print(f"Writing {OUTPUT_TTL}...")

# Collect all districts and quality buckets for ontology
all_districts = set(f["district"] for f in flats if f["district"])
all_qualities = set(f["lq_bucket"] for f in flats if f["lq_bucket"])

with open(OUTPUT_TTL, "w", encoding="utf-8") as out:

    out.write(PREFIXES)

    # --- Ontology / Class definitions ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Ontology\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for cls in ["Flat", "Stop", "TransitLine", "District", "LocationQuality"]:
        out.write(f"ex:{cls} a owl:Class ;\n")
        out.write(f'    rdfs:label "{cls}" .\n\n')

    for prop, domain, range_ in [
        ("isNearStop",          "Flat",        "Stop"),
        ("isOnLine",            "Stop",        "TransitLine"),
        ("inDistrict",          "Flat",        "District"),
        ("hasLocationQuality",  "Flat",        "LocationQuality"),
    ]:
        out.write(f"ex:{prop} a owl:ObjectProperty ;\n")
        out.write(f"    rdfs:domain ex:{domain} ;\n")
        out.write(f"    rdfs:range  ex:{range_} .\n\n")

    # --- District nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Districts\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for d in sorted(all_districts):
        out.write(f"ex:{d}\n")
        out.write(f'    a ex:District ;\n')
        out.write(f'    rdfs:label "{d.replace("_", " ").title()}" .\n\n')

    # --- Location quality nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Location Quality Categories\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for q in sorted(all_qualities):
        out.write(f"ex:{q}\n")
        out.write(f'    a ex:LocationQuality ;\n')
        out.write(f'    rdfs:label "{q.replace("_", " ").title()}" .\n\n')

    # --- Route nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Transit Lines\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for route in routes_raw:
        rid   = safe_id(route["route_id"])
        rtype = ROUTE_TYPE_LABELS.get(route.get("route_type", "3"), "Transit")
        rname = esc(route.get("route_short_name") or route.get("route_long_name", rid))
        out.write(f"ex:route_{rid}\n")
        out.write(f'    a ex:TransitLine ;\n')
        out.write(f'    rdfs:label "{rname}" ;\n')
        out.write(f'    ex:routeType "{rtype}" .\n\n')

    # --- Stop nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Transit Stops\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for stop in stops:
        out.write(f"ex:stop_{stop['id']}\n")
        out.write(f'    a ex:Stop ;\n')
        out.write(f'    rdfs:label "{esc(stop["name"])}" .\n\n')

    # --- Stop isOnLine Route ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Stop -> isOnLine -> Route\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for raw_stop_id, route_ids in stop_routes.items():
        sid = safe_id(raw_stop_id)
        for rid in route_ids:
            out.write(f"ex:stop_{sid} ex:isOnLine ex:route_{safe_id(rid)} .\n")
    out.write("\n")

    # --- Flat nodes (NO numeric literals) ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Flat Listings (structural relations only)\n")
    out.write("# ---------------------------------------------------------------\n\n")

    
    for flat in flats:
        fid = flat["id"]
        lines = [f"ex:{fid}", "    a ex:Flat"]
        if flat["heading"]:
            lines.append(f'    rdfs:label "{esc(flat["heading"])}"')
        if flat["district"]:
            lines.append(f'    ex:inDistrict ex:{flat["district"]}')
        if flat["lq_bucket"]:
            lines.append(f'    ex:hasLocationQuality ex:{flat["lq_bucket"]}')
        # Join with semicolons, last line gets a dot
        out.write(lines[0] + "\n")
        for line in lines[1:-1]:
            out.write(line + " ;\n")
        out.write(lines[-1] + " .\n\n")

    # --- isNearStop ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Flat -> isNearStop -> Stop\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for flat in flats:
        for stop_id in flat_nearby_stops[flat["id"]]:
            out.write(f"ex:{flat['id']} ex:isNearStop ex:stop_{stop_id} .\n")
    out.write("\n")

# ---------------------------------------------------------------------------
# 5. WRITE ATTRIBUTES JSON (all numeric data)
# ---------------------------------------------------------------------------

print(f"Writing {OUTPUT_ATTRS}...")

attributes = {}
for flat in flats:
    fid = f"http://example.org/viennakg/{flat['id']}"
    attributes[fid] = {
        "price":          flat["price"],
        "rooms":          flat["rooms"],
        "size":           flat["size"],
        "floor":          flat["floor"],
        "location_quality": flat["lq_num"],
        "transit_score":  flat_transit_scores[flat["id"]],
        "latitude":       flat["lat"],
        "longitude":      flat["lon"],
    }

with open(OUTPUT_ATTRS, "w", encoding="utf-8") as f:
    json.dump(attributes, f, indent=2)

# ---------------------------------------------------------------------------
# 6. SUMMARY
# ---------------------------------------------------------------------------

print("\nDone!")
print(f"\nSummary:")
print(f"  Routes          : {len(routes_raw)}")
print(f"  Stops           : {len(stops)}")
print(f"  Flats           : {len(flats)}")
print(f"  Districts       : {len(all_districts)}")
print(f"  Quality buckets : {len(all_qualities)}")
print(f"  isNearStop      : {total_near}")
print(f"\nOutputs:")
print(f"  {OUTPUT_TTL}   ← feed this to PyKEEN")
print(f"  {OUTPUT_ATTRS} ← use this for regression")