"""Focused numerical/decision identity and arm-selection refusals."""

import p2_design_score as score


base = {"accuracy_only": True, "local_seconds_caps": "off",
        "weights_mode": "f64-readonly", "spec": {"lam": 0.0},
        "gate": 1.05, "ts": "first", "official_samples": 1e9}
changed_decision = dict(base, gate=1.25, ts="second", official_samples=2e9)
assert score._numerical_config(base) == score._numerical_config(changed_decision)
for change in ({"weights_mode": "f32"}, {"spec": {"lam": 1.0}},
               {"local_seconds_caps": "official"}):
    assert score._numerical_config(base) != score._numerical_config(dict(base, **change))

manifest = {"arms": {"cand0": {"candidate_sha256": "a" * 64,
                               "evidence_rows": [{}], "rows_hash": "b" * 64},
                     "control": {"candidate_sha256": "c" * 64,
                                 "evidence_rows": [{}], "rows_hash": "d" * 64}}}
assert score._reuse_arm(manifest, "a" * 64) == "cand0"
for digest in ("e" * 64,):
    try:
        score._reuse_arm(manifest, digest)
    except ValueError:
        pass
    else:
        raise AssertionError("missing source accepted")
manifest["arms"]["duplicate"] = dict(manifest["arms"]["cand0"])
try:
    score._reuse_arm(manifest, "a" * 64)
except ValueError:
    pass
else:
    raise AssertionError("ambiguous source role accepted")
print("control reuse: decision fields independent; numerical mutation and ambiguous source refused")
