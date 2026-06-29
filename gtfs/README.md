# Vienna GTFS Input

This directory stores the local GTFS source data used by
`conversion_v2.py`. The raw data files are deliberately excluded from Git.

The converter requires the following files from one consistent Vienna GTFS
dataset:

```text
stops.txt
routes.txt
trips.txt
stop_times.txt
```

`stop_times.txt` is approximately 402 MB in the dataset used for this
project. Including it would make the source repository and assignment
submission unnecessarily large. The other raw GTFS files are excluded with
it so that the repository does not contain an incomplete or mixed version of
the external dataset.

Place the downloaded files in this directory before running:

```bash
python conversion_v2.py
```

The repository already contains `vienna_kg_entities.ttl` and
`vienna_kg_attributes.json`, which were generated from the project data.
