"""No-label workerq snapshot byte check for the F21 source paths."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "estimator_k3_strassen_L4_f21.py":
        "d77dab0e2c740cc9f6cd7d9172adacc8da731cc5b40ca87798f3fb006e475a94",
    "experiments/2026-09-12_f21/estimator_L4_parent.py":
        "052dd768e970c4bae831cff26f8ff15b0487f3b95eadf2fd3578e542aec34d58",
    "estimator_k3_strassen_L2.py":
        "390bc226bbf517df0110facfc0dc91c78ef670e8a49368a536cabf630057d1db",
}


def main():
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        print(f"{name}: {actual} {'PASS' if actual == expected else 'FAIL'}")
        if actual != expected:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
