while read i ; do 
sem -j+0
    "python3" "src/experiment_runner.py" $i; "python3" "src/plotter.py" $2; done < $1
sem --wait