"""Concatenate the research modules into ONE self-contained candidate file.

`GAME_PLAN.md` § WP5: the integration *"must be convertible to a single-file
estimator without importing local modules"*. This performs the conversion, so
that requirement is **demonstrated rather than asserted** — and the product is
what `local_sim/certify_candidate.py` certifies, because certifying a file that
`import`s `local_sim` would prove nothing about the grader sandbox.

WHAT IT PRODUCES
----------------
A file whose only imports are `math`, `flopscope`, `flopscope.numpy` and
`whestbench.BaseEstimator` — the subset of hard rule 1 this architecture needs.
`BaseEstimator` is not optional: `whest validate` refuses any file without a
subclass of it. It exposes `Estimator`
with the grader's `setup(context)` / `predict(mlp, flop_budget)` lifecycle and a
frozen configuration baked in at build time.

WHAT IT IS NOT
--------------
**Not a promotion and not an archive.** It writes into `experiments/`, never to
`estimator.py` and never to an `estimator_v*.py`; `GAME_PLAN.md` § 6 makes both
of those a STRATEGIC step with its own contract, Samuel's prediction and a
Skeptic. The output is a research artifact for pricing and certification.

Run:
    uv run python local_sim/p2_build_candidate.py --config carrier \
        --out experiments/<slice>/candidate_carrier_k8.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import p2_candidate_architecture as ARCH  # noqa: E402

HEADER = '''"""GENERATED research candidate — do NOT edit, and do NOT promote from here.

Built by `local_sim/p2_build_candidate.py` from:
%(sources)s

Configuration frozen at build time: %(config)s
Build inputs (sha256 of each source, in order): %(digest)s

This is a RESEARCH artifact for pricing and certification. Promotion into
`estimator.py` and the creation of a frozen `estimator_v*.py` archive are
separate STRATEGIC steps (`docs/GAME_PLAN.md` section 6) requiring a contract,
Samuel's prediction, a full preflight, a CONFIRM and an independent Skeptic.

Imports are `math`, `flopscope`, `flopscope.numpy` and
`whestbench.BaseEstimator` only -- the subset of AGENTS.md hard rule 1 this
architecture needs. Nothing from `local_sim` is imported: that is the property
this file exists to demonstrate.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator
'''

FOOTER = '''

# --------------------------------------------------------------------------- #
# the grader's entry point
# --------------------------------------------------------------------------- #

_CONFIG = %(config)r
_KWARGS = %(kwargs)r


class Estimator(BaseEstimator):
    """The grader lifecycle: one instance, `setup()` once off-budget.

    Subclassing `whestbench.BaseEstimator` is a CONTRACT REQUIREMENT, not a
    convention: `whest validate` refuses a file with
    `SCORING_VALIDATION_ERROR: No BaseEstimator subclasses found`. The first
    build of this footer defined a bare class and was correctly rejected --
    while every other section of the certification passed, which is exactly why
    section 1 runs the official validator instead of trusting the shape of the
    code.
    """

    def __init__(self):
        self._inner = ResearchCandidate(config=_CONFIG, **_KWARGS)

    def setup(self, context):
        self._inner.setup(context)

    def predict(self, mlp, flop_budget):
        return self._inner.predict(mlp, flop_budget)
'''


#: Build-time-only helpers that must not travel into the shipped file. They are
#: legal under hard rule 1 (`os` is stdlib), but a candidate has no business
#: carrying a function that enumerates repository paths.
DROP_DEFS = ("single_file_sources",)


def _strip_module(path, drop_imports=(), drop_defs=DROP_DEFS):
    """A module's body, minus its docstring, imports, `main()` and entry guard.

    Mechanical rather than clever: the two sources are written knowing they will
    be concatenated, so nothing here has to understand Python.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    # drop the module docstring
    m = re.match(r'\s*(?:"""|\'\'\')', text)
    if m:
        q = text[m.end() - 3:m.end()]
        end = text.index(q, m.end())
        text = text[end + 3:]

    out = []
    skip_block = False
    for line in text.splitlines():
        s = line.strip()
        if (s.startswith("def main(") or s.startswith("if __name__")
                or any(s.startswith("def %s(" % d) for d in drop_defs)):
            skip_block = True
            continue
        if skip_block:
            # a top-level statement ends the skipped block
            if line and not line[0].isspace():
                skip_block = False
            else:
                continue
        if s.startswith("from __future__"):
            continue
        if s in ("import math", "import flopscope", "import flopscope.numpy as fnp"):
            continue
        if any(s.startswith(d) for d in drop_imports):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def build(config="carrier", out=None, **kwargs):
    sources = ARCH.single_file_sources()
    digests = [hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
               for p in sources]

    bodies = []
    for p in sources:
        body = _strip_module(p, drop_imports=("import p2_carrier_fnp",))
        # the architecture references the carrier through an alias that no
        # longer exists once both modules share one namespace
        body = body.replace("carrier_mod.", "")
        bodies.append("# " + "-" * 73 + "\n# from %s\n# " % os.path.basename(p)
                      + "-" * 73 + "\n" + body)

    text = (HEADER % {
        "sources": "\n".join("  %s" % os.path.relpath(p, _ROOT).replace("\\", "/")
                             for p in sources),
        "config": config,
        "digest": " ".join(digests),
    } + "\n\n" + "\n\n\n".join(bodies)
        + FOOTER % {"config": config, "kwargs": kwargs})

    # `_fnp()` / `_flops()` were indirections so the modules could be imported
    # without the packages present. In one file the packages are imported at the
    # top under the names `fnp` and `flops`, so the indirection must bind to
    # THOSE names.
    #
    # This rewrite used to match the pre-strip text, imports included -- and
    # `_strip_module` had already removed `import flopscope` from inside the
    # function, so the pattern matched nothing and the built file carried a bare
    # `return flopscope` that raised `NameError` on the first call. The
    # certifier caught it; the audit above could not, because a NameError is not
    # an import. Match the POST-strip form, and verify it.
    before = text
    text = text.replace("def _fnp():\n    return fnp",
                        "def _fnp():\n    return fnp        # module-level")
    text = text.replace("def _flops():\n    return flopscope",
                        "def _flops():\n    return flops     # module-level")
    if "return flopscope" in text:
        raise AssertionError(
            "the _flops() indirection still returns the undefined name "
            "`flopscope`; the concatenated file would raise NameError")
    if before == text:
        raise AssertionError(
            "neither indirection was rewritten -- the source shape changed and "
            "this build step silently stopped doing anything")

    if out:
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return text


def _audit(path):
    """Refuse to call a file self-contained without checking it."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    bad = []
    for i, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if not (s.startswith("import ") or s.startswith("from ")):
            continue
        if s in ("import math", "import flopscope as flops",
                 "import flopscope.numpy as fnp",
                 "from whestbench import BaseEstimator",
                 "from __future__ import annotations"):
            continue
        bad.append("%d: %s" % (i, s))
    for banned in ("numpy", "scipy", "torch", "local_sim", "p2_carrier_fnp",
                   "p2_candidate_architecture"):
        for i, line in enumerate(src.splitlines(), 1):
            s = line.strip()
            if (s.startswith("import ") or s.startswith("from ")) and banned in s:
                if "flopscope.numpy" in s:
                    continue
                bad.append("%d: BANNED %s" % (i, s))
    return sorted(set(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="carrier", choices=list(ARCH.CONFIGS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--blend", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kwargs = {"k": a.k, "chunk_rows": a.chunk_rows, "dtype": a.dtype}
    if a.blend is not None:
        kwargs["blend"] = a.blend
    text = build(config=a.config, out=a.out, **kwargs)

    rel = os.path.relpath(os.path.abspath(a.out), _ROOT).replace("\\", "/")
    print("built %s" % rel)
    print("  config    %s  %s" % (a.config, kwargs))
    print("  bytes     %d" % len(text.encode("utf-8")))
    print("  sha256    %s" % hashlib.sha256(
        open(a.out, "rb").read()).hexdigest())

    bad = _audit(os.path.abspath(a.out))
    if bad:
        print("  IMPORT AUDIT FAILED -- the file is not self-contained:")
        for b in bad:
            print("    %s" % b)
        return 1
    print("  import audit: only math / flopscope / flopscope.numpy / "
          "whestbench.BaseEstimator. No numpy, no local_sim.")
    print()
    print("  Certify it (this is NOT promotion):")
    print("    uv run python local_sim/certify_candidate.py --candidate %s" % rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
