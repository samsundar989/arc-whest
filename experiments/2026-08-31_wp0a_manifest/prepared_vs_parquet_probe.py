"""Does whestbench 0.16.0's PREPARED-ARROW fast path materialise the same rows
as the `data/*.parquet` source objects the WP0A record compares?

Read-only. Downloads ONE prepared shard (~538 MB) plus the parquet shard whose
rows it should reproduce, and compares the columns whestbench actually consumes.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "aicrowd/arc-whestbench-public-2026"
REV = "v2-phase2"
OUT = os.path.join(r"c:\Users\samsu\Documents\arc-whest", "experiments",
                   "2026-08-31_wp0a_manifest", "prepared_vs_parquet.json")

print("downloading prepared/mini/data-00000-of-00014.arrow ...", flush=True)
arrow = hf_hub_download(REPO, "prepared/mini/data-00000-of-00014.arrow",
                        repo_type="dataset", revision=REV)
print("  ->", arrow, os.path.getsize(arrow), flush=True)

print("downloading data/mini-00000-of-00007.parquet ...", flush=True)
parq = hf_hub_download(REPO, "data/mini-00000-of-00007.parquet",
                       repo_type="dataset", revision=REV)
print("  ->", parq, os.path.getsize(parq), flush=True)

with pa.memory_map(arrow, "rb") as src:
    at = pa.ipc.open_stream(src).read_all()
print("arrow rows", at.num_rows, "cols", at.schema.names, flush=True)

pt = pq.read_table(parq)
print("parquet rows", pt.num_rows, "cols", pt.schema.names, flush=True)

n = min(at.num_rows, pt.num_rows)
cols = ["mlp_id", "mlp_name", "mlp_seed", "avg_variance"]
report = {
    "artifact": "wp0a_prepared_vs_parquet",
    "artifact_version": 1,
    "question": ("whestbench 0.16.0 prefers prepared/<split> Arrow over "
                 "data/*.parquet whenever metadata.json advertises "
                 "prepared_splits (dataset.py:479-491, dataset_io.py:61-68). "
                 "The v2-phase2 tag move ADDED that tree. Do the two paths "
                 "materialise the same rows, in the same order?"),
    "revision": REV,
    "arrow_shard": "prepared/mini/data-00000-of-00014.arrow",
    "parquet_shard": "data/mini-00000-of-00007.parquet",
    "arrow_rows": at.num_rows,
    "parquet_rows": pt.num_rows,
    "rows_compared": n,
    "arrow_columns": list(at.schema.names),
    "parquet_columns": list(pt.schema.names),
    "columns_added_by_prepared": sorted(set(at.schema.names) - set(pt.schema.names)),
    "columns_missing_from_prepared": sorted(set(pt.schema.names) - set(at.schema.names)),
    "scalar_columns": {},
    "array_columns": {},
}

for c in cols:
    if c not in at.schema.names or c not in pt.schema.names:
        report["scalar_columns"][c] = {"present": False}
        continue
    a = at.column(c).to_pylist()[:n]
    b = pt.column(c).to_pylist()[:n]
    report["scalar_columns"][c] = {
        "present": True,
        "identical_in_order": a == b,
        "identical_as_sets": sorted(map(str, a)) == sorted(map(str, b)),
        "first_arrow": str(a[0])[:60] if a else None,
        "first_parquet": str(b[0])[:60] if b else None,
    }
    print(f"  {c}: in-order={a == b} as-sets="
          f"{sorted(map(str, a)) == sorted(map(str, b))}", flush=True)


def _flat(v):
    """Flatten whatever nesting the column uses into a float64 numpy array."""
    out = []
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, (list, tuple)):
            stack.extend(reversed(x))
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif x is None:
            pass
        else:
            out.append(x)
    return np.asarray(out, dtype=np.float64)


for c in ("weights", "all_layer_means"):
    if c not in at.schema.names or c not in pt.schema.names:
        report["array_columns"][c] = {"present": False}
        continue
    worst = 0.0
    exact = 0
    digests = []
    for i in range(n):
        av = _flat(at.column(c)[i].as_py())
        bv = _flat(pt.column(c)[i].as_py())
        if av.shape != bv.shape:
            worst = float("inf")
            break
        d = float(np.max(np.abs(av - bv))) if av.size else 0.0
        worst = max(worst, d)
        if d == 0.0:
            exact += 1
        digests.append((hashlib.sha256(av.tobytes()).hexdigest()[:16],
                        hashlib.sha256(bv.tobytes()).hexdigest()[:16]))
    report["array_columns"][c] = {
        "present": True,
        "rows": n,
        "bitwise_identical_rows": exact,
        "max_abs_difference": worst,
        "first_row_digests_arrow_vs_parquet": digests[0] if digests else None,
    }
    print(f"  {c}: {exact}/{n} rows bitwise identical, max|diff|={worst}", flush=True)

report["verdict"] = (
    "PREPARED_MATCHES_PARQUET"
    if all(v.get("identical_in_order") for v in report["scalar_columns"].values()
           if v.get("present"))
    and all(v.get("bitwise_identical_rows") == v.get("rows")
            for v in report["array_columns"].values() if v.get("present"))
    else "PREPARED_DIFFERS"
)
report["scope"] = (
    "One of 14 prepared shards against one of 7 parquet shards. It settles "
    "whether the two paths agree on the rows they share; it does not verify "
    "the other 13 shards, nor the `full` split, nor that a future re-bake of "
    "the prepared tree would still agree."
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(report, indent=2) + "\n")
print("VERDICT:", report["verdict"])
print("written:", OUT)
