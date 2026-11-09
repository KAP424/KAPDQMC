#!/bin/bash
# usage: bash run_all.sh [workers]
# 目录结构: beta<b>/L<L>/ 下每个 U 一个 csv；日志在 logs/
# Theta = TL * L（默认 TL=0.15），其余设置与 new 目录相同
ROOT=D:/JuliaDQMC/code/KAPDQMC/test/tU_Measure/tl=0.15
WORKERS=${1:-14}
BETAS=${BETAS:-"0.0"}
LS=${LS:-"6 9"}
TL=${TL:-0.15}
US="3.0 3.3 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.5"
declare -A WARM=([3]=50 [6]=50 [9]=50 [12]=50)
declare -A SWEEPS=([3]=500 [6]=500 [9]=500 [12]=500)
JOBS=$ROOT/jobs.txt
mkdir -p $ROOT/logs
: > $JOBS
for L in $LS; do
  THETA=$(awk -v c=$TL -v l=$L 'BEGIN{print c*l}')
  for b in $BETAS; do
    for U in $US; do
      echo "$ROOT/beta$b/L$L $b $L $U ${WARM[$L]} ${SWEEPS[$L]} $THETA" >> $JOBS
    done
  done
done
echo "jobs: $(wc -l < $JOBS)"
xargs -P $WORKERS -L 1 bash -c 'julia --startup-file=no '$ROOT'/run_one.jl "$@" > '$ROOT'/logs/b$2_L$3_U$4.log 2>&1' _ < $JOBS
echo "ALL DONE"
