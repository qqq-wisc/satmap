while read i ; do 
sem -j+0
    "python3" "src/experiment_runner.py" $i; done < $1
sem --wait
