# R5 (P6.4): fold arm continued 2 epochs from R4 fold's best, same recipe (restart rule: stop after 2 restarts < 0.3 %)
cd /workspace/p5/arc
A=/workspace/p5/arc
K=$A/experiments/2026-10-06_mask/ktab/aggr_q50.json
RHO=$A/experiments/2026-10-10_p5/d2/rho.json
M="R=2.0,Hd=2.5,D2=2.5,rho=$RHO,J=2,K=$K"
mkdir -p /workspace/p6/r5
nohup python3 -u tools/twin/twin.py train --cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py --dress all --init /workspace/p6/r4/fold/params_best.npz --epochs 2 --accum 8 --lr 1e-4 --eval-every 28 --patience 99 --hold-panel hold100 --bank star,starall --mask "$M" --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf --out /workspace/p6/r5/fold > /workspace/p6/r5/fold.log 2>&1 &
