# R8 (P6.2): age 6 (M6 hand-over) vs a matched control, 1 epoch each from R7 fold's best, R4 recipe
cd /workspace/p5/arc
A=/workspace/p5/arc
K=$A/experiments/2026-10-06_mask/ktab/aggr_q50.json
RHO=$A/experiments/2026-10-10_p5/d2/rho.json
M="R=2.0,Hd=2.5,D2=2.5,rho=$RHO,J=2,K=$K"
C="--dress all --init /workspace/p6/r7/fold/params_best.npz --epochs 1 --accum 8 --lr 1e-4 --eval-every 28 --patience 99 --hold-panel hold100 --bank star,starall --mask $M --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf"
mkdir -p /workspace/p6/r8
nohup python3 -u tools/twin/twin.py train --cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py $C --out /workspace/p6/r8/ctl > /workspace/p6/r8/ctl.log 2>&1 &
nohup python3 -u tools/twin/twin.py train --cand /workspace/p6/m6/cand_age6.py $C --out /workspace/p6/r8/age6 > /workspace/p6/r8/age6.log 2>&1 &
