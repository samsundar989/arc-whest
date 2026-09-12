# Tiny physical signed-law fixture execution record

The IMPORTANT LOG contract was committed at `12d0dad` before these runs. The
source was not committed before execution. An additional instruction to freeze
the source first arrived after the third attempt. No gates, matrices, or
reference laws changed between attempts; the two changes were execution fixes.
There is no separate terminal log file for the first two attempts. The error
lines below are transcribed from the tool responses, so no earlier source hash
or full stdout is claimed.

All three attempts used this command in the repository root:

```
uv run python local_sim/test_p2_k4_causal_signed_physical.py --output experiments/2026-09-12_causal_signed_physical/raw.json
```

1. Exit 1 before JSON publication. The independent numerical root finder used
   the whole prior-stage interval; a ReLU zero plateau could make both bracket
   endpoint values have the same sign. The tool traceback ended at
   `scipy.optimize.brentq` with `ValueError: f(a) and f(b) must have different signs`.
   The implementation fix narrowed the bracket to the interior of its already
   frozen piecewise-linear sector.
2. Exit 1 before JSON publication. All calculations returned, but JSON could
   not serialize a `numpy.bool_` in the gate table. The tool traceback ended at
   `TypeError: Object of type bool is not JSON serializable`. The implementation
   fix converted gate outcomes to Python `bool`.
3. Exit 0, published `raw.json` once, with 102 passed and 0 failed gates.
   Source SHA256: `dffd6aa6d4232827082f77f6d72da5f2781dfb93c374674192a88556b315416a`.
   First-publication raw bytes (CRLF) SHA256:
   `de0822109e54a27d04b58fff9f36b37dac2c0ea8eb82ea936d39bd8160d2cf5f`.
   Those exact bytes are preserved in `raw_first_publication.bin`. Because the
   repository requires LF for JSON, `raw.json` was subsequently normalized by
   replacing only CRLF with LF. Its SHA256, on disk and in Git, is
   `d838fed959e64ae20afa7114d0c7f8680c0d5697d160d08b55270f2889ede344`.

No further numerical run was made after the third attempt. The source hash
embedded in `raw.json` matches the final source bytes on disk.
