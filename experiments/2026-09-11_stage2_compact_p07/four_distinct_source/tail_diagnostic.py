"""Optional [0,10] truncation and 32/48 convergence diagnostics for the fixed fixture."""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "local_sim"))
from test_p2_k4_four_distinct_source import MULTI, direct_raw  # noqa: E402


def main() -> None:
    at10, nodes10 = direct_raw(64, 10.0)
    at12_32, nodes32 = direct_raw(32, 12.0)
    at12_48, nodes48 = direct_raw(48, 12.0)
    at12_64, nodes64 = direct_raw(64, 12.0)
    changes = {"".join(map(str, p)): abs(at10[p]-at12_64[p]) for p in MULTI}
    low_order = {"".join(map(str, p)): abs(at12_32[p]-at12_48[p]) for p in MULTI}
    result = {
        "scope": "optional diagnostics only; no new gate",
        "node_counts": {"32": nodes32, "48": nodes48, "64": nodes64,
                        "64_limit10": nodes10},
        "max_raw_abs_10_vs_12": max(changes.values()),
        "max_raw_abs_32_vs_48": max(low_order.values()),
        "selected_10_vs_12": {p: changes[p] for p in ("2000", "1010", "2011", "1111")},
        "selected_32_vs_48": {p: low_order[p] for p in ("2000", "1010", "2011", "1111")},
    }
    destination = Path(__file__).with_name("tail_diagnostic.json")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",
                           encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
