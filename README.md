# SATMap
This is a MaxSAT-based tool for the Qubit Mapping and Routing Problem
# Installation

1. Setup the runtime environment

    Install docker (https://docs.docker.com/get-docker/) and
    then pull the docker image with the command:

    ```$ docker pull abtinm/qmapping:artifact```  

2. Clone this repo *including the MaxSAT solver* 

   ```$ git clone https://github.com/qqq-wisc/satmap.git  --recurse-submodules```
 
    If you forget the ```--recurse-submodules``` option the first time around, you can fetch the submodule later with 

     ```$ git submodule update --init --recursive```

3. Start container and mount the artifact

    From the satmap/ directory, mount the artifact in the
    Docker container as a volume with the command:

    ```$ docker run -itv "$(pwd):/home/" abtinm/qmapping:artifact```

4. Build the MaxSAT solver. Make sure you have [GMP](https://gmplib.org/) installed. Then build with:
    ```
    $ cd mapper/lib/Open-WBO-Inc/
    $ make r
    ```

5. Test the installation

    Return to the home/ directory and use the script ./functional.sh
    to test the installation. If everything is working, the
    script should run SATMAP on a few small benchmarks
    and save the output in the results/ directory
    

    
