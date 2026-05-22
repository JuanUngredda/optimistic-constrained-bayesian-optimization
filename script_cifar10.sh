#!/usr/bin/sh

# ACQUISITION_FUNCTIONS = ["obsolete_aware_ucb", "obsolete_aware_ucb_mixed", "eic", "cmes_ibo"]

NRAND=40


EXPERIMENT="cifar10_cnn"

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json --num-rand $NRAND --acqfunc obsolete_aware_ucb_mixed --plot 0 > log/log_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt 2> log/err_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt