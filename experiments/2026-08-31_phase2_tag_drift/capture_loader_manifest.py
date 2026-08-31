"""Capture a read-only comparison of EVERY object in the v2-phase2 tree at both
commits -- not only `data/`.

WP0A's first record covered `data/*.parquet` on the premise that nothing else
feeds a measurement. An independent Skeptic falsified that premise against the
installed package: `whestbench/dataset.py:479-491` prefers a `prepared/<split>`
Arrow tree over the parquet path whenever `metadata.json` advertises
`prepared_splits`, and the tag move added exactly that tree.
"""
import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "aicrowd/arc-whestbench-public-2026"
OLD = "ce926b7d1ced65faa668c01b94a43f8c4f43783d"
NEW = "6a416fa5fb45cade1e9ad34320204853f5d4a5f6"
OUT = os.path.join(r"c:\Users\samsu\Documents\arc-whest", "experiments",
                   "2026-08-31_phase2_tag_drift", "loader_manifest.tsv")

api = HfApi()


def tree(rev):
    out = {}
    for f in api.list_repo_tree(REPO, repo_type="dataset", revision=rev,
                                recursive=True):
        if getattr(f, "tree_id", None) is not None and not hasattr(f, "size"):
            continue                      # a directory entry
        size = getattr(f, "size", None)
        if size is None:
            continue
        lfs = getattr(f, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        out[f.path] = (size, sha or "", getattr(f, "blob_id", "") or "")
    return out


print("listing", OLD[:12], flush=True)
a = tree(OLD)
print("  ", len(a), "objects", flush=True)
print("listing", NEW[:12], flush=True)
b = tree(NEW)
print("  ", len(b), "objects", flush=True)


def kind(p):
    if p.startswith("data/"):
        return "source"
    if p.startswith("prepared/"):
        return "prepared"
    if p == "metadata.json":
        return "metadata"
    if p == "README.md":
        return "readme"
    return "other"


paths = sorted(set(a) | set(b))
rows = []
for p in paths:
    ao, bo = a.get(p), b.get(p)
    if ao and bo:
        status = "unchanged" if ao == bo else "changed"
    elif bo:
        status = "added"
    else:
        status = "removed"
    def fmt(o):
        if o is None:
            return ("", "", "")
        size, sha, blob = o
        return (str(size), sha or "-", blob or "-")
    ao_s, ao_h, ao_b = fmt(ao)
    bo_s, bo_h, bo_b = fmt(bo)
    rows.append("\t".join([p, kind(p), status, ao_s, bo_s, ao_h, bo_h,
                           ao_b, bo_b]))

digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
counts = {}
for r in rows:
    k, st = r.split("\t")[1], r.split("\t")[2]
    counts[(k, st)] = counts.get((k, st), 0) + 1

hdr = [
    "# Read-only Hugging Face tree comparison of the FULL v2-phase2 tree.",
    "# Captured 2026-08-31 after an independent Skeptic falsified the premise",
    "# that only data/*.parquet feeds a measurement: whestbench 0.16.0",
    "# (dataset.py:479-491, dataset_io.py:61-68) PREFERS a prepared/<split>",
    "# Arrow tree over the parquet path whenever metadata.json advertises",
    "# prepared_splits, and this tag move added exactly that tree.",
    "# repo: %s" % REPO,
    "# old_commit: %s" % OLD,
    "# new_commit: %s" % NEW,
    "# old_object_count: %d" % len(a),
    "# new_object_count: %d" % len(b),
    "# row_count: %d" % len(rows),
    "# Canonical digest input is every row below in path order, joined by LF",
    "# with no trailing LF.",
    "# canonical_loader_manifest_sha256: %s" % digest,
]
for (k, st), n in sorted(counts.items()):
    hdr.append("# count_%s_%s: %d" % (k, st, n))
hdr.append("\t".join(["path", "kind", "status", "old_size", "new_size",
                      "old_lfs_sha256", "new_lfs_sha256",
                      "old_blob_id", "new_blob_id"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(hdr + rows) + "\n")

for (k, st), n in sorted(counts.items()):
    print("  %-10s %-10s %d" % (k, st, n))
print("digest", digest)
print("written", OUT, len(rows), "rows")
