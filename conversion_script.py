"""
convert_to_turtle.py
--------------------
Converts GTFS transit data + Willhaben flat listings into RDF Turtle (.ttl)
for the Vienna Transit-Aware Flat Price Knowledge Graph.

Usage:
    python convert_to_turtle.py

Expected inputs:
    - gtfs/stops.txt
    - gtfs/routes.txt
    - gtfs/trips.txt
    - gtfs/stop_times.txt
    - flat_info.json

Output:
    - vienna_kg.ttl
"""

import json
import csv
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — adjust paths if needed
# ---------------------------------------------------------------------------

GTFS_DIR = "./gtfs"                  # folder containing your GTFS .txt files
FLAT_JSON = "./flat_info.json"       # output of extractor.py
OUTPUT_TTL = "./vienna_kg2.ttl"

NEAR_STOP_DISTANCE_M = 500           # metres threshold for flat isNearStop stop

# ---------------------------------------------------------------------------
# NAMESPACES
# ---------------------------------------------------------------------------

PREFIXES = """\
@prefix ex:   <http://example.org/viennakg/> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix schema: <https://schema.org/> .

"""

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in metres between two lat/lon points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def safe_id(raw: str) -> str:
    """Turn an arbitrary string into a safe URI local name."""
    return raw.replace(" ", "_").replace("/", "_").replace(":", "_") \
              .replace("(", "").replace(")", "").replace(",", "")


def esc(value: str) -> str:
    """Escape a string literal for Turtle."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

# ---------------------------------------------------------------------------
# 1. LOAD GTFS DATA
# ---------------------------------------------------------------------------

print("Loading GTFS data...")

stops_raw      = read_csv(f"{GTFS_DIR}/stops.txt")
routes_raw     = read_csv(f"{GTFS_DIR}/routes.txt")
trips_raw      = read_csv(f"{GTFS_DIR}/trips.txt")
stop_times_raw = read_csv(f"{GTFS_DIR}/stop_times.txt")

# Build lookup: trip_id -> route_id
trip_to_route = {t["trip_id"]: t["route_id"] for t in trips_raw}

# Build unique stop -> set of route_ids  (deduplicated from stop_times)
print("Deriving stop->route relationships (may take a moment)...")
stop_routes = defaultdict(set)
for st in stop_times_raw:
    route_id = trip_to_route.get(st["trip_id"])
    if route_id:
        stop_routes[st["stop_id"]].add(route_id)

# ---------------------------------------------------------------------------
# 2. LOAD FLAT DATA
# ---------------------------------------------------------------------------

print("Loading flat data...")
with open(FLAT_JSON, encoding="utf-8") as f:
    flats_raw = json.load(f)

# Keep only flats with valid coordinates and a price
flats = []



for i, flat in enumerate(flats_raw):
    try:
        lat = float(flat["LATITUDE"])
        lon = float(flat["LONGITUDE"])
        price_raw = flat.get("PRICE") or flat.get("RENT/PER_MONTH_LETTINGS")
        if not price_raw:
            continue
        price = float(price_raw.split("||||")[0])
        flats.append({
            "id": f"flat_{i}",
            "lat": lat,
            "lon": lon,
            "price": price,
            "postcode": flat.get("POSTCODE", "").split("||||")[0],
            "heading": flat.get("HEADING", "").split("||||")[0],
            "rooms": flat.get("NUMBER_OF_ROOMS", "").split("||||")[0],
            "size": flat.get("ESTATE_SIZE/LIVING_AREA", flat.get("ESTATE_SIZE", "")).split("||||")[0],
            "floor": flat.get("FLOOR", "").split("||||")[0],
        })
    except (ValueError, KeyError):
        continue

print(f"  {len(flats)} valid flats loaded")

# Parse stops into usable dicts
stops = []
for s in stops_raw:
    try:
        stops.append({
            "id": safe_id(s["stop_id"]),
            "raw_id": s["stop_id"],
            "name": s["stop_name"],
            "lat": float(s["stop_lat"]),
            "lon": float(s["stop_lon"]),
        })
    except (ValueError, KeyError):
        continue

print(f"  {len(stops)} valid stops loaded")
print(f"  {len(routes_raw)} routes loaded")

# ---------------------------------------------------------------------------
# 3. COMPUTE isNearStop RELATIONSHIPS
# ---------------------------------------------------------------------------

print(f"Computing isNearStop (threshold: {NEAR_STOP_DISTANCE_M}m)...")

near_stop_pairs = []   # (flat_id, stop_id, distance_m)

for flat in flats:
    for stop in stops:
        dist = haversine_m(flat["lat"], flat["lon"], stop["lat"], stop["lon"])
        if dist <= NEAR_STOP_DISTANCE_M:
            near_stop_pairs.append((flat["id"], stop["id"], round(dist, 1)))

print(f"  {len(near_stop_pairs)} isNearStop triples generated")

# ---------------------------------------------------------------------------
# 4. WRITE TURTLE FILE
# ---------------------------------------------------------------------------

print(f"Writing Turtle to {OUTPUT_TTL}...")

ROUTE_TYPE_LABELS = {
    "0": "Tram", "1": "Metro", "2": "Rail",
    "3": "Bus",  "4": "Ferry", "5": "CableCar",
    "6": "Gondola", "7": "Funicular",
}

with open(OUTPUT_TTL, "w", encoding="utf-8") as out:

    out.write(PREFIXES)

    # --- Route nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Transit Lines (Routes)\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for route in routes_raw:
        rid = safe_id(route["route_id"])
        rtype = ROUTE_TYPE_LABELS.get(route.get("route_type", "3"), "Transit")
        rname = esc(route.get("route_short_name") or route.get("route_long_name", rid))
        out.write(f"ex:route_{rid}\n")
        out.write(f'    a ex:TransitLine ;\n')
        out.write(f'    rdfs:label "{rname}" ;\n')
        out.write(f'    ex:routeType "{rtype}" ;\n')
        out.write(f'    ex:routeId "{esc(route["route_id"])}" .\n\n')

    # --- Stop nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Transit Stops\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for stop in stops:
        sid = stop["id"]
        out.write(f"ex:stop_{sid}\n")
        out.write(f'    a ex:Stop ;\n')
        out.write(f'    rdfs:label "{esc(stop["name"])}" ;\n')
        out.write(f'    ex:latitude "{stop["lat"]}"^^xsd:decimal ;\n')
        out.write(f'    ex:longitude "{stop["lon"]}"^^xsd:decimal .\n\n')

    # --- Stop isOnLine Route ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Stop -> isOnLine -> Route\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for raw_stop_id, route_ids in stop_routes.items():
        sid = safe_id(raw_stop_id)
        for rid in route_ids:
            out.write(f"ex:stop_{sid} ex:isOnLine ex:route_{safe_id(rid)} .\n")
    out.write("\n")

    # --- Flat nodes ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Flat Listings\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for flat in flats:
        fid = flat["id"]
        out.write(f"ex:{fid}\n")
        out.write(f'    a ex:Flat ;\n')
        if flat["heading"]:
            out.write(f'    rdfs:label "{esc(flat["heading"])}" ;\n')
        out.write(f'    ex:price "{flat["price"]}"^^xsd:decimal ;\n')
        out.write(f'    ex:latitude "{flat["lat"]}"^^xsd:decimal ;\n')
        out.write(f'    ex:longitude "{flat["lon"]}"^^xsd:decimal ;\n')
        if flat["postcode"]:
            out.write(f'    ex:postcode "{esc(flat["postcode"])}" ;\n')
        if flat["rooms"]:
            out.write(f'    ex:numberOfRooms "{esc(flat["rooms"])}" ;\n')
        if flat["size"]:
            out.write(f'    ex:livingArea "{esc(flat["size"])}" ;\n')
        if flat["floor"]:
            out.write(f'    ex:floor "{esc(flat["floor"])}" ;\n')
        out.write(f'    schema:url "https://www.willhaben.at" .\n\n')

    # --- isNearStop ---
    out.write("# ---------------------------------------------------------------\n")
    out.write("# Flat -> isNearStop -> Stop\n")
    out.write("# ---------------------------------------------------------------\n\n")

    for flat_id, stop_id, dist in near_stop_pairs:
        out.write(f"ex:{flat_id} ex:isNearStop ex:stop_{stop_id} .\n")
    out.write("\n")

print("Done!")
print(f"\nSummary:")
print(f"  Routes  : {len(routes_raw)}")
print(f"  Stops   : {len(stops)}")
print(f"  Flats   : {len(flats)}")
print(f"  isNearStop triples: {len(near_stop_pairs)}")
print(f"\nOutput: {OUTPUT_TTL}")