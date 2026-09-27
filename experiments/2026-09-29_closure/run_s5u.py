"""Step 3.3 twin run: identity check, untuned read, then stage S5 (absolute paths: the job
runs in a worker-q snapshot, and results must land in the live repo)."""
import subprocess, sys
R = "C:/Users/samsu/Documents/arc-whest"
D = R + "/experiments/2026-09-29_closure"
S3 = R + "/experiments/2026-09-25_twin/S3/params_best.npz"
py = [sys.executable, "tools/twin/twin.py"]
ev = lambda cand, out: subprocess.run(py + ["eval", "--cand", cand, "--params", S3, "--panel", "hold100",
                                              "--json", out], check=True)

ev("estimator_k3_v62_closu.py", D + "/hold_initu.json")               # offline init, untuned
subprocess.run(py + ["train", "--stage", "S5", "--cand", "estimator_k3_v62_closu.py", "--init", S3,
                     "--out", D + "/S5u", "--epochs", "2", "--accum", "8", "--lr", "3e-4",
                     "--zero-scale", "0.3", "--eval-every", "28", "--patience", "3"], check=True)
