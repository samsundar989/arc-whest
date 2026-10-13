# Patch atoms for `tools/build_v85.py`

Lanes S, K and O deliver estimator changes as atoms; Lane P (main) integrates, gates and submits.

## Contract
One module per atom family, `tools/atoms/<name>.py`:

```python
from tools.atoms import rep

def apply(src: str, z) -> str:        # used for --atoms <name>
    return rep(src, "<exact anchor, occurs once>", "<replacement>")

def make(arg: str):                   # optional; used for --atoms <name>:<arg>, e.g. k1:3.0
    return lambda src, z: ...
```

- `src` is the full estimator text after the base (v84 = `fold+age6`) and every earlier atom in the `--atoms` list.
- `z` is the loaded params npz. An atom that bakes learned constants reads its keys from it and must fail loudly
  when a key it needs is absent (never silently fall back to an unrefit value).
- Anchors are exact text and must occur exactly once (`rep` enforces it). Write anchors against the base text, and
  say in the docstring which other atoms the anchors assume (order matters when two atoms touch the same lines).
- The module docstring states: owner lane and handoff step; bit-identical or not; exact interpreter F/B and ops on
  the v84 base; the twin arm it corresponds to.

## Slots
| lane | atoms |
|---|---|
| S | `sb2`, `depth16`, `depth8`, `sbks` |
| K | `k0`, `k1`, `k2`, `k3`, `k4`, `k5` |
| O | `o1`, `o2`, `o3`, `o4` |

Check the base before anything else: `uv run python tools/build_v85.py --params experiments/2026-10-11_p6/r8/age6/params_best.npz --check-v84`.
