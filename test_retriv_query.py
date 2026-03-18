"""Minimal script to load a nemotron-8b retriv index and issue a test query.

Run on sk3:
    python test_retriv_query.py
"""

import time
from pathlib import Path

import retriv
from retriv import DenseRetriever

# --- Config: pick any agency/year that has indexes ---
BULK_DIR = Path("/lfs/skampere3/0/alexspan/regulations-demo/data/bulk_downloads")
AGENCY_YEAR = "dhs/dhs_2017_2018"
LEVEL = "claims"
MODEL = "nvidia/llama-embed-nemotron-8b"
MAX_LENGTH = 4096

index_base = BULK_DIR / AGENCY_YEAR / ".retriv_indexes"
safe_suffix = MODEL.replace("/", "_")
dir_name = AGENCY_YEAR.split("/")[-1]
index_name = f"{dir_name}_{LEVEL}_{safe_suffix}"

print(f"Index base: {index_base}")
print(f"Index name: {index_name}")
print(f"Index dir:  {index_base / 'collections' / index_name}")
print()

# 1. Load the index
retriv.set_base_path(str(index_base))

t0 = time.time()
dr = DenseRetriever.load(index_name)
load_time = time.time() - t0
print(f"Loaded index in {load_time:.2f}s")

# Fix max_length if needed
if dr.encoder is not None and dr.encoder.max_length != MAX_LENGTH:
    print(f"Overriding max_length {dr.encoder.max_length} -> {MAX_LENGTH}")
    dr.encoder.max_length = MAX_LENGTH
    dr.encoder.tokenizer_kwargs["max_length"] = MAX_LENGTH

print(f"Encoder model: {dr.model}")
print(f"Num docs: {len(dr.id_mapping)}")
print()

# 2. Issue a test query
query = "The proposed rule fails to account for the economic impact on small businesses in rural communities."

print(f"Query: {query}")
print()

t0 = time.time()
results = dr.search(query, cutoff=10)
query_time = time.time() - t0

print(f"Query completed in {query_time:.2f}s")
print(f"Results ({len(results)}):")
for r in results:
    print(f"  score={r['score']:.4f}  id={r['id']}")
    # Optionally peek at text if available
    if "text" in r:
        print(f"    text={r['text'][:120]}...")
print()

# 3. Try a second query (model already warm)
query2 = "EPA should strengthen emissions standards for power plants to address climate change."
t0 = time.time()
results2 = dr.search(query2, cutoff=10)
query_time2 = time.time() - t0
print(f"Second query: {query2}")
print(f"Completed in {query_time2:.2f}s")
print(f"Top result: score={results2[0]['score']:.4f}  id={results2[0]['id']}")
