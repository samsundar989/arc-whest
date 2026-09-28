# K111 rank screen (PLAN rev 15 B3a) — predictions, written before the run

Tool: `tools/k111_diag.py rank` (new subcommand). FIT nets 0–7, free-running twin (v62_s1 + S3),
source layers l ∈ {2, 5, 8, 11, 14}. E = label_AD − model_AD in the full D21 form: off-diagonal
D21 AD error + D3 AD error restored on the diagonal. Spectra of E, of label AD, of E orthogonal
to model AD, and of the rep1-vs-s00 noise matrix N = (AD_s00 − AD_rep1)/√2 (the K21 part of the
label noise only; the full1000 N = 1e9 part is not measured by rep1).
Downstream: every layer, D3 := model + D3 AD error (full), D21 := model + offdiag(rank-r(E)),
r ∈ {0, 8, 32, 128, full}, no refit.

## Expectations

- E is not compact: rank 64 captures 30–50% of ‖E‖² at l ≥ 8 (1024 × 1024, random-weight
  transport spreads the error); label AD itself is more compact (rank 64 ≥ 70%) since it is
  dominated by a few smooth directions; E ⊥ model AD is the least compact.
- Noise: ‖N‖² ≈ 0.1–1% of ‖E‖² at l ≥ 8, flat spectrum; E's singular values above N's top
  singular value for the first ~100–300 directions.
- Downstream: r = 0 ≈ the earlier AD_D3only arm (≈ −12%); r = 8 → −14 to −16%; r = 32 →
  −17 to −20%; r = 128 → −24 to −28%; full ≈ −33%.
- Most likely verdict: the rank-32 oracle recovers less than 1/3 of −33% beyond what the D3
  diagonal alone gives → "compact remainder rejected". (The stop rule is read on the total
  change including the diagonal; I will report both readings.)

## Resources

workerq, `--ram 16 --vram 9 --cpus 4`, normal priority, queued behind G1tg (2261), not
preempting. About 8 × (≈20 s spectrum run + 5 arms × ≈25 s) ≈ 20–25 min.

## Stop condition

As set by the coordinator: rank 64 < 50% of ‖E‖² at l ≥ 8, or rank-32 oracle < 1/3 of the
full-AD oracle → "compact remainder rejected". No expansion beyond nets 0–7 either way.
