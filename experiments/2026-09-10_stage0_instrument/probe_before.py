import sys, os, json
sys.path.append(os.path.join(os.getcwd(), "local_sim"))
import numpy as np
import competition_score as cs
out = {}
rng = np.random.default_rng(7)
d = rng.normal(1e-8, 3e-8, size=40)
ci95 = cs.paired_diff_ci(d, method="t", alpha=0.05)
ci99 = cs.paired_diff_ci(d, method="t", alpha=0.01)
ci80 = cs.paired_diff_ci(d, method="t", alpha=0.20)
out["P3_t_alpha_ignored"] = {"ci_alpha_0.05": ci95, "ci_alpha_0.01": ci99,
    "ci_alpha_0.20": ci80, "identical_95_vs_99": ci95 == ci99,
    "identical_95_vs_80": ci95 == ci80}
n = 24
rs = np.random.default_rng(11)
mse_c = rs.uniform(3e-8, 6e-8, n)
mse_k = mse_c * rs.uniform(0.85, 1.05, n)
B = 2.0**41
fl = rs.uniform(0.2, 0.6, n) * B
ctrl = [cs.MlpRow(mse=float(mse_c[i]), flops_used=float(fl[i]), budget=B, name="mlp%02d" % i) for i in range(n)]
cand = [cs.MlpRow(mse=float(mse_k[i]), flops_used=float(fl[i]), budget=B, name="mlp%02d" % i) for i in range(n)]
r_ok = cs.paired_compare_rows(ctrl, cand)
perm = list(rs.permutation(n))
cand_shuf = [cand[i] for i in perm]
try:
    r_bad = cs.paired_compare_rows(ctrl, cand_shuf); accepted = True; md_bad = r_bad.mean_difference
except Exception as e:
    accepted, md_bad = False, repr(e)
out["P4_pairs_by_order"] = {"accepted_shuffled_arm": accepted,
    "mean_difference_correct": r_ok.mean_difference,
    "mean_difference_shuffled": md_bad,
    "differs": (accepted and md_bad != r_ok.mean_difference)}
dup = [cand[0]] * n
try:
    cs.paired_compare_rows(ctrl, dup); dup_ok = True
except Exception:
    dup_ok = False
out["P4_accepts_duplicate_ids"] = dup_ok
N, M = 20000, 512
g = np.random.default_rng(3)
v = 1.1552e-9
true_mse_c = 4.0e-8; true_mse_k = 4.0e-8
q_c, q_k = 0.60, 0.25
sd_c = (true_mse_c + v) * 2 / np.sqrt(M); sd_k = (true_mse_k + v) * 2 / np.sqrt(M)
obs_c = true_mse_c + v + g.normal(0, sd_c, N)
obs_k = true_mse_k + v + g.normal(0, sd_k, N)
emp = float(np.mean(q_c * obs_c - q_k * obs_k))
true_part = q_c * true_mse_c - q_k * true_mse_k
closed = (q_c - q_k) * v + true_part
out["P8_unequal_multiplier_noise"] = {
    "empirical_mean_adjusted_difference": emp,
    "closed_form": closed, "true_part": true_part,
    "noise_term_(qc-qk)*v": (q_c - q_k) * v,
    "rel_error_vs_closed_form": abs(emp - closed) / abs(closed),
    "noise_share_of_measured_difference": (q_c - q_k) * v / emp}
print(json.dumps(out, indent=2, default=str))
