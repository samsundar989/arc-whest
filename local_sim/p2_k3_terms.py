"""Enumerate upstream's SIMPLE-mode k_max=3 term list, which depends on nothing
but (k_max, d_max, augment). Freezing it turns an open-ended port into a bounded
one: the fnp side then implements a fixed, short list of contractions."""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\samsu\Documents\mlp_cumulant_propagation\src")

from mlp_kprop.kprop_harmonic import get_all_terms_iso, factored_keeps_term
from mlp_kprop.partitions import int_partitions, check_vec_partition

terms_iso = get_all_terms_iso(k_max=3, d_max=4, augment=False)

rows = []
for int_part, vec_part_dict in terms_iso.items():
    for vec_part, count in vec_part_dict.items():
        keep = (
            len(int_part) <= 3
            and int_part not in [(3, 1), (2, 1, 1)]
            and int_part != (1, 1, 1)
            and (int_part, set(vec_part)) != ((2, 1, 1), {(1, 1, 1)})
            and factored_keeps_term(3, int_part, vec_part)
        )
        rows.append({
            "int_part": list(int_part),
            "vec_part": [list(v) for v in vec_part],
            "count": count,
            "kept_by_factored_simple": bool(keep),
            "check_vec_partition": [int(x) for x in check_vec_partition(vec_part, len(int_part))],
        })

kept = [r for r in rows if r["kept_by_factored_simple"]]
print(f"total enumerated terms : {len(rows)}")
print(f"kept in SIMPLE mode    : {len(kept)}")
print()
by = {}
for r in kept:
    by.setdefault(tuple(r["int_part"]), []).append(r)
for ip in sorted(by, key=lambda t: (len(t), t)):
    print(f"  int_part {str(ip):<12} {len(by[ip])} vec_parts")
    for r in by[ip]:
        print(f"      vec_part {str([tuple(v) for v in r['vec_part']]):<34} "
              f"count {r['count']:<4} wick_sum {r['check_vec_partition']}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_terms_simple.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump({
        "source": "mlp_kprop.kprop_harmonic.get_all_terms_iso(k_max=3, d_max=4, augment=False)",
        "upstream_rev": "93d091a4",
        "note": "depends only on (k_max, d_max, augment); independent of n and of the weights",
        "int_partitions_d3": [list(p) for p in int_partitions(3)],
        "terms": rows,
    }, fh, indent=1)
print(f"\nwrote {out}")
