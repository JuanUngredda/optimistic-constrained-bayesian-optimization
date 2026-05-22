#!/bin/sh

NRAND=40
EXPERIMENT="speedreducer"

mkdir -p log out

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json \
  --num-rand $NRAND \
  --acqfunc obsolete_aware_ucb_mixed \
  --plot 0 \
  > log/log_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt \
  2> log/err_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt
