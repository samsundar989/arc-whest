# Step 4(b) fold-age arms — predictions before any run (2026-09-26 ~05:50 UTC)

Arms on v65_s2t: early-born lineages (born at layer <= U) fold into the shared rank-256 basis at
age A instead of 5. Folding earlier removes exact old-tier columns at layers A..8 (cost down) and
compresses lineages that are contracted x0.6-0.88/layer before reaching layer 16 (accuracy cost
small, per PLAN's argument).

Mine: FLOPs -1.5 to -3% (A3U3), -1 to -2% (A4U3), -3 to -5% (A3U5); raw +0.5 to +2% worse each.
Net score roughly neutral; I expect NO arm to clear the 2% kill line (PLAN §3 Step 4: "kill any
(b)/(c) arm at < 2% score"). D5's age 4 -> 5 (all lineages) cost +6% FLOPs for -11% raw, which says
the old tier is under-funded, not over-funded -- evidence against this arm.
