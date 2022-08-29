# SATMap
   This is a MaxSAT-based tool for the Qubit Mapping and Routing Problem, implementing the algorithm described in [1]. This branch represents the latest version of SATMap and should be used for best results. To reproduce the experimental conditions of the paper, see the [micro22-artifact branch](https://github.com/qqq-wisc/satmap/tree/micro22-artifact).

# Dependencies

+ A C++ compiler, ``make``, and [GMP](https://gmplib.org/) to build the MaxSAT solver: [Open-WBO-Inc](https://github.com/sbjoshi/Open-WBO-Inc)
+ Python 3.8 or later with the third-party packages ``qiskit``, ``scipy``, and ``pysat``, which can all be installed via ``pip``

There is also a Docker image available [here](https://hub.docker.com/repository/docker/abtinm/qmapping) that provides a Ubuntu environment with the above preinstalled.

# Installation
1. Clone this repo *including the MaxSAT solver* 

   ```$ git clone https://github.com/qqq-wisc/satmap.git  --recurse-submodules```
 
    If you forget the ```--recurse-submodules``` option the first time around, you can fetch the submodule later with 

     ```$ git submodule update --init --recursive```

2. Build the MaxSAT solver. Make sure you have [GMP](https://gmplib.org/) installed. Then build with:
    ```
    $ cd satmap/lib/Open-WBO-Inc/
    $ make r
    ```
    
# Usage
To run SATMap on the file "circ.qasm" on the IBM Tokyo architecture
```
$ python3 src/satmap.py circ.qasm --arch tokyo [options]
```
The choices for options include:
+ ``--k <int>``: Sets the *slice size* for the local relaxation. Smaller values divide the problem into easier to solve subproblems with the trade-off of a degradation in solution quality and higher probability of backtracking. In our experience, the default value of 25 is generally a good choice.
+ ``--cyclic on``: Adds the constraint that the final mapping must be equal to the initial mapping, allowing for solution reuse in a circuit with repeating substructure.
+ ``--timeout <int>``: Sets a total budget (in seconds) for the MaxSAT solver."
+ ``--output_path <file_path>``: Sets a path for saving the output circuit. By default, SATMap writes the result of mapping and routing "fname.qasm" to a file in the home directory called "mapped_fname.qasm."

# Custom Architectures
SATMap includes the "brick-like" 20-qubit IBM Tokyo and heavy-hexagonal 27-qubit IBM Toronto connectivity graphs. It also provides functions for generating linear and nearest-neighbor connectivity graphs with arbitrary dimensions. 

To use a different connectivity graph, generate a text file consisting of an adjacency matrix for the desired graph and pass it as the ``--arch`` argument.
For example, to use a triangle connectivity graph:
 ```
    $ cat triangle.txt 
    [[0,1,0], [0,0,1], [1,0,0]]
    $ python3 src/satmap.py circ.qasm --arch triangle.txt
 ```

