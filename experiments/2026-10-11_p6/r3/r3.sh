cd /workspace/p5/arc
A=/workspace/p5/arc
K=$A/experiments/2026-10-06_mask/ktab/aggr_q50.json
KS=$A/experiments/2026-10-11_p6/screen
RHO=$A/experiments/2026-10-10_p5/d2/rho.json
P=/workspace/p5/r2/v82b/params_best.npz
M="R=2.0,Hd=2.5,D2=2.5,rho=$RHO,J=2"
C="--cand experiments/2026-09-27_gates/estimator_k3_v62_gb.py --dress all --init $P --epochs 2 --accum 8 --lr 1e-4 --eval-every 28 --patience 99 --hold-panel hold100 --bank star,starall"
mkdir -p /workspace/p6/r3
for arm in "$@"; do
  case $arm in
    ctl)      X="--mask $M,K=$K --arm mask+dress+bank --stage S2ts";;
    fhabs)    X="--mask $M,K=$KS/K_habs25.json,Hrank=abs,Hfull=1 --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf";;
    fhasym2030) X="--mask $M,K=$KS/K_asym2030.json,Hrank=asym:2.0:3.0,Hfull=1 --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf";;
    fhasym2025) X="--mask $M,K=$KS/K_asym2025.json,Hrank=asym:2.0:2.5,Hfull=1 --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf";;
    fhabs20)  X="--mask $M,K=$KS/K_habs20.json,Hrank=abs,Hfull=1 --arm mask+dress+bank+fold --fold tau=2,mode=rec --stage S2tsf";;
  esac
  nohup python3 -u tools/twin/twin.py train $C $X --out /workspace/p6/r3/$arm > /workspace/p6/r3/$arm.log 2>&1 &
done
