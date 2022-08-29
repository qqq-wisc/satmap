# SATMap
   This is a MaxSAT-based tool for the Qubit Mapping and Routing Problem, implementing the algorithm described in [1]. This branch represents the latest version of SATMap and should be used for best results. To reproduce the experimental conditions of the paper, see the [micro22-artifact branch](https://github.com/qqq-wisc/satmap/tree/micro22-artifact).

# Dependencies

+ A C++ compiler, ``make``, and [GMP](https://gmplib.org/) to build the MaxSAT solver: [Open-WBO-Inc](https://github.com/sbjoshi/Open-WBO-Inc)
+ Python 3.8 or later with the third-party packages ``qiskit``, ``scipy``, and ``pysat``, which can all be installed via ``pip``

There is also a Docker image avaliable [here](https://hub.docker.com/repository/docker/abtinm/qmapping) that provides a Ubuntu environment with the above preinstalled.

# Installation
1. Clone this repo *including the MaxSAT solver* 

   ```$ git clone https://github.com/qqq-wisc/mapper.git  --recurse-submodules```
 
    If you forget the ```--recurse-submodules``` option the first time around, you can fetch the submodule later with 

     ```$ git submodule update --init --recursive```

2. Build the MaxSAT solver. Make sure you have [GMP](https://gmplib.org/) installed. Then build with:
    ```
    $ cd mapper/lib/Open-WBO-Inc/
    $ make r
    ```
    
# Usage
To run SATMap on the file "circ.qasm" on the IBM Tokyo architecture
```
$ python3 circ.qasm --arch tokyo [options]
```
    
