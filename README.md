# SATMap
This is a MaxSAT-based tool for the Qubit Mapping and Routing Problem
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
    
