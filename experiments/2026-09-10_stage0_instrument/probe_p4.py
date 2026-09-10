import sys, os, json
sys.path.append(os.path.join(os.getcwd(), "local_sim"))
import numpy as np, competition_score as cs
n = 24; rs = np.random.default_rng(11); B = 2.0**41
mse_c = rs.uniform(3e-8, 6e-8, n); mse_k = mse_c * rs.uniform(0.85, 1.05, n)
fl = rs.uniform(0.2, 0.6, n) * B
ctrl = [cs.MlpRow(mse=float(mse_c[i]), flops_used=float(fl[i]), budget=B, name="mlp%02d"%i) for i in range(n)]
cand = [cs.MlpRow(mse=float(mse_k[i]), flops_used=float(fl[i]), budget=B, name="mlp%02d"%i) for i in range(n)]
ok = cs.paired_compare_rows(ctrl, cand)
res = {"correct": {"mean_difference": ok.mean_difference, "se": ok.se_difference,
                   "t_ci": ok.t_ci_difference, "wins": ok.n_candidate_better,
                   "geometric": ok.geometric_gain}}
ses, excl = [], 0; T = 200
for t in range(T):
    p = rs.permutation(n)
    r = cs.paired_compare_rows(ctrl, [cand[i] for i in p])
    ses.append(r.se_difference)
    lo, hi = r.t_ci_difference
    if lo > 0 or hi < 0: excl += 1
lo, hi = ok.t_ci_difference
res["correct"]["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
res["shuffled"] = {"se_min": min(ses), "se_max": max(ses), "se_median": float(np.median(ses)),
                   "se_inflation_median_x": float(np.median(ses)) / ok.se_difference,
                   "ci_excludes_zero_rate": excl / T, "trials": T}
# the gate itself
v_ok = cs.decide([r.score() for r in ctrl], [r.score() for r in cand], gate=1.05)
res["gate_correct_pairing"] = {"passed": v_ok.passed, "gain": v_ok.gain,
                               "ci_excludes_zero": v_ok.ci_excludes_zero}
print(json.dumps(res, indent=2, default=str))
