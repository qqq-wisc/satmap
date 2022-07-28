while read i ; do 
"python3" "src/experiment_runner.py" $i; "python3" "src/plotter.py" $2; done < $1
