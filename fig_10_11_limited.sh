mkdir -p aux_files plots results
./run_from_arg_file.sh arg_files/fig_10_11_limited.txt -c
python3 src/plotter.py -c