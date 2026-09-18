"""Tiny, label-free meter completeness and tamper checks."""
from __future__ import annotations

from copy import deepcopy

import s18_reference_meter as meter


def refuses(total, ledger):
    try:
        meter._assert_ledger_conserved(total, ledger)
    except ValueError:
        return
    raise AssertionError("incomplete operation ledger accepted")


def main():
    result = meter._one("unshared", width=8, depth=6)
    total = result["flops_used"]
    operations = result["operations"]
    assert meter._assert_ledger_conserved(total, operations) == result["operation_calls"]
    assert result["peak_process_rss_bytes"] > 0
    assert result["scalar_materializations_total"] == sum(
        result["scalar_materializations"].values())
    assert result["scalar_materializations_total"] == 0
    truncated = deepcopy(operations)
    truncated.pop(next(iter(truncated)))
    refuses(total, truncated)
    changed = deepcopy(operations)
    changed[next(iter(changed))]["flops"] += 1
    refuses(total, changed)
    refuses(total + 1, operations)
    refuses(0, operations)
    print("PASS: tiny instrumented reference meter and four incomplete-ledger refusals")


if __name__ == "__main__":
    main()
