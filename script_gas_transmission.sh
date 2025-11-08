#!/usr/bin/sh

# ACQUISITION_FUNCTIONS = ["obsolete_aware_ucb", "obsolete_aware_ucb_mixed", "eic", "cmes_ibo"]

NRAND=10


EXPERIMENT="gas_transmission_2_3_15"

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json --num-rand $NRAND --acqfunc obsolete_aware_ucb_mixed --plot 0 > log/log_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt 2> log/err_obsolete_aware_ucb_mixed_"$EXPERIMENT".txt

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json --num-rand $NRAND --acqfunc eic --plot 0 > log/log_eic_"$EXPERIMENT".txt 2> log/err_eic_"$EXPERIMENT".txt

python run-multi-constraint-bo.py experiment_config/"$EXPERIMENT".json --num-rand $NRAND --acqfunc cmes_ibo --plot 0 > log/log_cmes_ibo_"$EXPERIMENT".txt 2> log/err_cmes_ibo_"$EXPERIMENT".txt


