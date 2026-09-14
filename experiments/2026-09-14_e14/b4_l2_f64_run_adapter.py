def run():
    if sha(ORIGINAL) != ORIGINAL_SHA256:
        raise RuntimeError("Original failed B4 artifact changed")
    CHECKS.clear()
    ACTIVITY.clear()
    CONTEXTS.clear()
    frozen = b1.load_frozen()
    table = frozen._tables()  # coefficient data only; no frozen nonlinear helper
    weights = fixture_weights().astype(np.float32).astype(np.float64)
    assert weights.shape == (6, N, N)
    checks, activity = CHECKS, ACTIVITY
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        dtype = fnp.float64
        const = {"off_mask": fnp.ones((N, N), dtype=dtype) - fnp.eye(N, dtype=dtype),
                 "ones_n": fnp.ones(N, dtype=dtype),
                 "wick": frozen._wick_program(table, dtype)}
        reference = (fnp.zeros(N, dtype=dtype), fnp.eye(N, dtype=dtype), None, 0.0)
        compact_state = (np.zeros(N), np.eye(N), compact.Shared(), 0.0)
        for layer, fixed in enumerate(weights):
            final = layer == len(weights) - 1
            W = fnp.array(fixed, dtype=dtype)
            Wc = fixed.astype(np.float32).T.astype(np.float64)
            mu, cov, shared, c4 = compact_state
            context = {"layer": layer, "final": final,
                       "weights": snapshot(W),
                       "reference_incoming": snapshot(reference),
                       "compact_incoming": snapshot(compact_state)}
            CONTEXTS.append(context)
            record(checks, layer, "incoming.mu", mu, reference[0])
            record(checks, layer, "incoming.cov", cov, reference[1])
            record(checks, layer, "incoming.c4", c4, reference[3])
            if layer:
                compare_sources(checks, layer, shared, reference[2])
            advanced = compact.transport_shared(shared, Wc, np.ones(N))
            d3, d21_raw = compact.compact_slices(advanced, need_d21=not final)
            D3 = np.zeros(N) if d3 is None else d3
            off_mask = np.ones((N, N)) - np.eye(N)
            D21 = (None if final else
                   np.zeros((N, N)) if d21_raw is None else d21_raw * off_mask)
            pre_mu = Wc @ mu
            pre_cov = Wc @ cov @ Wc.T
            var = np.maximum(np.diag(pre_cov), 1e-12)
            sigma = np.sqrt(var)
            alpha = np.clip(pre_mu / sigma, -40.0, 40.0)
            w = independent_wick(pre_mu, sigma, alpha, table)
            D4 = c4 * 4.0 * np.ones(N)
            S = {(1,): pre_mu, (3,): D3, (4,): D4}
            if not final:
                C11 = pre_cov * off_mask
                D22 = (c4 * 4.0 / 3.0) * off_mask
                S.update({(2,): var, (1, 1): C11, (2, 1): D21,
                          (1, 2): D21.T, (2, 2): D22})
            pK, used_terms = independent_pk(S, w, table, final)
            K, used_center = independent_center(pK, table, final)
            ref_next, trace = observe_layer(frozen, W, reference, table, dtype,
                                            const, final)
            context["reference_checkpoints"] = {
                label: snapshot({key: value for key, value in frame.items()
                                 if key in {"mu_pre", "cov_pre", "var", "sigma", "alpha",
                                            "w", "S", "pK", "K", "p111_3", "p111_21",
                                            "S3", "S21", "adv", "out_tw", "c4"}})
                for label, frame in trace.items()}
            context["compact_pre"] = snapshot({
                "mu_pre": pre_mu, "cov_pre": pre_cov, "var": var,
                "sigma": sigma, "alpha": alpha, "w": w, "S": S,
                "pK": pK, "K": K, "advanced": advanced})
            observed = trace["before_K"]
            record(checks, layer, "pre.mu", pre_mu, trace["before_pk"]["mu_pre"])
            record(checks, layer, "pre.cov", pre_cov, trace["before_pk"]["cov_pre"])
            inventory(checks, layer, "S.keys", S, observed["S"])
            inventory(checks, layer, "wick.keys", w, observed["w"])
            inventory(checks, layer, "wick.table_keys", w,
                      [(int(spec["k"]), int(spec["p"]))
                       for spec in table["wick"].values()])
            for key in sorted(S):
                record(checks, layer, f"S.{key}", S[key], observed["S"][key])
            record(checks, layer, "D3.compact", D3, observed["S"][(3,)])
            if not final:
                record(checks, layer, "D21.compact_masked", D21,
                       observed["S"][(2, 1)])
                record(checks, layer, "D21.transpose", D21.T,
                       observed["S"][(1, 2)])
                record(checks, layer, "D21.diagonal_zero",
                       np.diag(D21), np.zeros(N))
            for key in sorted(w):
                record(checks, layer, f"wick.{key}", w[key], observed["w"][key])
            inventory(checks, layer, "pK.keys", pK, observed["pK"])
            for key in sorted(pK):
                record(checks, layer, f"pK.{key}", pK[key], observed["pK"][key])
            inventory(checks, layer, "K.keys", K, trace["after_K"]["K"])
            for key in sorted(K):
                record(checks, layer, f"K.{key}", K[key], trace["after_K"]["K"][key])
            inventory(checks, layer, "terms.row_inventory", used_terms,
                      [i for i, row in enumerate(table["terms"])
                       if not final or len(row["int_part"]) == 1])
            inventory(checks, layer, "center.row_inventory", used_center,
                      [(key, i) for key, rows in table["pK_to_K"].items()
                       if not final or key.replace(" ", "") in ("(1,)", "(1)")
                       for i in range(len(rows))])
            mu_out = K[(1,)]
            record(checks, layer, "post.mu", mu_out, ref_next[0])
            if final:
                record(checks, layer, "final.tower_absent", int(ref_next[2] is None), 1)
                record(checks, layer, "final.c4_zero", ref_next[3], 0)
                compact_state = (mu_out, None, compact.Shared(), 0.0)
                activity.append({"layer": layer, "ages": 0,
                                 "D3_max_abs": float(np.max(np.abs(D3))),
                                 "D21_max_abs": None, "qr_rank": None})
                reference = ref_next
                continue
            w1, w2 = w[(1, 1)], w[(2, 1)]
            fac1 = w1[:, None] * C11
            p111_3 = (w1 ** 3) * D3
            p111_21 = ((w1 ** 2)[:, None] * w1[None, :]) * D21
            p111_21 += fac1 ** 2 * w2[None, :] * off_mask
            S3 = K[(3,)] - p111_3
            S21 = K[(2, 1)] - p111_21
            context["compact_subtraction"] = snapshot({
                "p111_3": p111_3, "p111_21": p111_21, "S3": S3, "S21": S21})
            for name, value in (("p111_3", p111_3), ("p111_21", p111_21)):
                record(checks, layer, name, value, observed[name])
            record(checks, layer, "birth.S3", S3,
                   trace["after_subtraction"]["S3"])
            record(checks, layer, "birth.S21", S21,
                   trace["after_subtraction"]["S21"])
            old_scaled = scale_old(advanced, w1)
            compare_sources(checks, layer, old_scaled,
                            trace["after_birth"]["adv"],
                            prefix="old_after_wick", age_offset=1)
            newborn = compact.Source(
                A=fac1, P=np.eye(N), Z=np.diag(S3) + 3.0 * S21.T,
                L=np.eye(N), s=np.zeros(N), e=np.zeros(N), w2=w2,
                dA=np.ones(N), dP=np.ones(N), c1=None, c2=None, Y=None, age=0)
            sketch = newborn.A @ (newborn.dA[:, None]
                                   * (newborn.A.T @ np.eye(N)))
            sketch += newborn.P @ (newborn.dP[:, None]
                                    * (newborn.P.T @ np.eye(N)))
            if old_scaled.Q is not None:
                sketch += old_scaled.Q @ (old_scaled.gram
                                          @ (old_scaled.Q.T @ np.eye(N)))
            sketch_rank = int(np.linalg.matrix_rank(sketch))
            record(checks, layer, "qr.pre_sketch_rank", sketch_rank, N)
            next_shared = compact.weighted_gram_join(old_scaled, newborn,
                                                      RANK, np.eye(N))
            context["compact_birth"] = snapshot({
                "old_scaled": old_scaled, "newborn": newborn,
                "sketch": sketch, "next_shared": next_shared})
            next_cov = np.diag(K[(2,)]) + K[(1, 1)]
            lap = table["lap2_d4"]
            c4_out = projection_coefficient(N) * (
                float(lap["(4,)"]["terms"][0]["coef"])
                * float(lap["(4,)"]["int_partition_coef"])
                * float(np.sum(K[(4,)]))
                + float(lap["(2, 2)"]["terms"][0]["coef"])
                * float(lap["(2, 2)"]["int_partition_coef"])
                * float(np.sum(K[(2, 2)])))
            record(checks, layer, "post.cov", next_cov, ref_next[1])
            record(checks, layer, "post.c4", c4_out, ref_next[3])
            record(checks, layer, "birth.A_L", fac1,
                   b1.tower_sources(ref_next[2])[-1].A)
            record(checks, layer, "birth.A_D", compact.materialize(newborn),
                   b1.tower_sources(ref_next[2])[-1].Z)
            compare_sources(checks, layer, next_shared, ref_next[2])
            qr_rank = int(np.linalg.matrix_rank(next_shared.Q))
            record(checks, layer, "qr.rank", qr_rank, N)
            activity.append({"layer": layer, "ages": len(next_shared.factors),
                             "incoming_ages": len(advanced.factors),
                             "D3_max_abs": float(np.max(np.abs(D3))),
                             "D21_max_abs": float(np.max(np.abs(D21))),
                             "c4_max_abs": float(abs(c4_out)),
                             "p111_3_max_abs": float(np.max(np.abs(p111_3))),
                             "p111_21_max_abs": float(np.max(np.abs(p111_21))),
                             "pre_sketch_rank": sketch_rank,
                             "qr_rank": qr_rank})
            compact_state = (mu_out, next_cov, next_shared, c4_out)
            reference = ref_next
            context["compact_outgoing"] = snapshot(compact_state)
            context["reference_outgoing"] = snapshot(reference)
        positive(checks, len(weights) - 2, "activity.off_diagonal_D21", activity[len(weights) - 2]["D21_max_abs"])
        positive(checks, len(weights) - 2, "activity.scalar_c4", activity[len(weights) - 2]["c4_max_abs"])
        positive(checks, len(weights) - 2, "activity.p111_3", activity[len(weights) - 2]["p111_3_max_abs"])
        positive(checks, len(weights) - 2, "activity.p111_21", activity[len(weights) - 2]["p111_21_max_abs"])
        record(checks, len(weights) - 2, "activity.all_ages_nonfinal",
               int(activity[len(weights) - 2]["incoming_ages"] >= len(weights) - 2), 1)
        record(checks, len(weights) - 1, "final.output", compact_state[0], reference[0])
    return {"fixture": "B4 fixed asymmetric n8/depth6 formula; full shared/residual rank",
            "scope": "tiny full-rank nonlinear parity only; no compression/cost/score/truth",
            "tolerance": TOL,
            "normalization": "abs(got-want)/max(1,abs(want)) elementwise",
            "source_sha256": {"adapter": sha(__file__), "original_B4_probe": sha(ROOT / "local_sim" / "p2_compact_fullrank_deep_b4.py"), "frozen_L2": sha(b1.FROZEN),
                              "B1_probe": sha(b1.__file__),
                              "B3_probe": sha(ROOT / "local_sim" / "p2_compact_fullrank_nonlinear_b3.py"),
                              "compact_reference": sha(b1.COMPACT)},
            "weights": weights.tolist(), "activity": activity, "checks": checks,
            "check_count": len(checks),
            "failed_count": sum(not row["passed"] for row in checks),
            "max_normalized_error": max(row.get("normalized_error", 0)
                                        for row in checks)}
