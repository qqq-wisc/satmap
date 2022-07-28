mkdir -p aux_files plots results;
for EX in examples/jku_constraint_based/*; do VAR_ARGS="${EX} --mapper solveSwapsFF"; python3 src/experiment_runner.py $VAR_ARGS; done