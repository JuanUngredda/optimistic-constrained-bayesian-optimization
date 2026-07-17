#!/bin/sh

NRAND=40
EXPERIMENT="constrainedfunc3redundantc1"

mkdir -p log out

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json \
  --num-rand $NRAND \
  --acqfunc ucbc_decoupled \
  --plot 0 \
  > log/log_ucbc_decoupled_"$EXPERIMENT".txt \
  2> log/err_ucbc_decoupled_"$EXPERIMENT".txt

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json \
  --num-rand $NRAND \
  --acqfunc ucbc_coupled \
  --plot 0 \
  > log/log_ucbc_coupled_"$EXPERIMENT".txt \
  2> log/err_ucbc_coupled_"$EXPERIMENT".txt
