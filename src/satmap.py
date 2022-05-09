import itertools
import math
import os
import re
import subprocess
import sys
import time
import numpy as np
import qiskit
import qiskit.circuit
import qiskit.dagcircuit
import architectures

# Controls whether debug is output (overwritten by Local if True)
DEBUG_GLOBAL = True


## OpenQASM parsing  ##


def extractCNOTs(fname):
    cnots = []
    with open(fname) as f:
        for line in f:
            match = re.match(r'cx\s+q\[(\d+)\],\s*q\[(\d+)\];', line)
            if match:
                cnots.append((int(match.group(1)), int(match.group(2))))
    return cnots


def extractQbits(fname):
    # Returns highest-value register used
    highest_register_value = 0
    with open(fname) as f:
        for line in f:
            if 'qreg' in line or 'creg' in line:  # these are not actual instructions
                continue  # skip adding qbits to the set
            match = re.findall(r'\[(\d+)\]', line)
            for num in match:
                num = int(num)
                if num > highest_register_value:
                    highest_register_value = num
    return highest_register_value + 1  # <--register values start at 0
 
## Topological layering ##

def getLayers(cnots):
    layers = [0]
    for i in range(len(cnots)):
        if inconsistent(cnots[:i], cnots[i]) and len(layers) == 1:
            layers.append(i)
        elif len(layers) >1 and  inconsistent(cnots[layers[-1]:i], cnots[i]):
            layers.append(i)
    return layers


def inconsistent(cnots, cnot):
    relevantQubits = [c for (c, _) in cnots] + [t for (_, t) in cnots]
    return (cnot[0] in relevantQubits or cnot[1] in relevantQubits)

def sortCnots(logNum, cnots):
    qc = qiskit.QuantumCircuit(logNum,0)
    for (c, t) in cnots:
        qc.cx(c,t)
    dag = qiskit.converters.circuit_to_dag(qc)
    sorted_cnots = []
    for layer in dag.layers():
       pairs = layer["partition"]
       sorted_cnots = sorted_cnots + list(map(lambda p: tuple(map(lambda q: q.index, p)), pairs))
    return sorted_cnots

## Constraint Generation ##


def generateAndWriteClauses(logNum, liveCnots, cnots, cm, swapNum, ffClauses, path, routing=True, layering=False):
    ''' 
        Writes the constraints corresponding to a particular MaxSat Instance to the given path as a wcnf file
    '''
    
    physNum = len(cm)
    numCnots = len(cnots)
    if layering:
        layers = getLayers(cnots)
    else: layers = list(range(len(cnots)))
    liveLog = set([c for (c,_) in liveCnots] + [t for (_,t) in liveCnots])
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    top = numS + numR + 1
    with open(path, "w") as f:
        f.write("p wcnf " + str(42) + " " + str(42) + " " + str(top) + "\n")
        writeFunConConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f)
        writeInjectivityConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f)
        writeCnotConstraint(cnots, cm, physNum, logNum, swapNum, top, f)
        if routing:
            writeSwapChoiceConstraint(swapNum, layers, cm, physNum, logNum, numCnots, top, f)
            writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, f)
        for clause in ffClauses:
            writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum)
        writeOptimizationConstraints(swapNum, physNum, numCnots, cm, logNum, routing, f)

# Mapping Constraints #

def writeFunConConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, path):
    for k in range(numCnots):
        for j in liveLog:
            atLeastOneJ = []
            for i in range(physNum):
                atLeastOneJ.append((False,"x", i,j,k))
                for i2 in range(i):
                    clause=[(True, "x", i2, j, k), (True,"x", i,j,k)]
                    writeHardClause(path, top, clause, physNum, logNum, numCnots, swapNum)
            writeHardClause(path, top, atLeastOneJ, physNum, logNum, numCnots, swapNum)


def writeInjectivityConstraint(numCnots, liveLog, physNum, logNum,  swapNum, top, path):
    for i in range(physNum):
        for j in liveLog:
            for k in range(numCnots):
                for j2 in range(j):
                   writeHardClause(path, top, [(True, "x", i, j2, k), (True,"x",i,j,k)], physNum, logNum, numCnots, swapNum)



def writeCnotConstraint(cnots, cm, physNum, logNum, swapNum, top, path):
    numCnots = len(cnots)
    for k in range(len(cnots)):
        (c,t) = cnots[k]
        edgeUsed = []
        nonzeroIndices = np.argwhere(cm>0)
        for edge in nonzeroIndices:
            [u,v] = edge
            edgeUsed.append((False, "p", u, v, k))
            edgeUsed.append((False, "r",  u, v, k))
            clauses =  [[(False, "x", u, c, k), (True, "p", u, v, k)],
                        [(False, "x", v, t, k), (True, "p", u, v, k)],
                        [(False, "x", u, t, k), (True, "r", u, v, k)],
                        [(False, "x", v, c, k), (True, "r", u, v, k)]]
            for clause in clauses:
                writeHardClause(path, top, clause, physNum, logNum, numCnots, swapNum)
        writeHardClause(path, top, edgeUsed, physNum, logNum, numCnots, swapNum)


# Routing Constraints #

def writeSwapChoiceConstraint(swapNum, layers, cm, physNum, logNum, numCnots,top, path):
    allowedSwaps = np.append(np.argwhere(cm>0), [[0,0]], axis=0)
    for k in layers:
        for t in range(swapNum):
            atLeastOne = []
            for (u,v) in allowedSwaps:
                i = 0
                atLeastOne.append((False, "s", u, v, t, k))
                if i != 0:
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (False,"b", i-1, t, k)], physNum, logNum, numCnots, swapNum)
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (True, "b", i, t, k)], physNum, logNum, numCnots, swapNum)
                    writeHardClause(path, top, [(True,"b", i-1, t, k), (False, "b", i, t, k), (False, "s", u, v, t, k)], physNum, logNum, numCnots, swapNum)
                writeHardClause(path, top, [(False, "b", i, t, k), (True,"b", i+1, t, k)], physNum, logNum, numCnots, swapNum)
                i += 1
            writeHardClause(path, top, atLeastOne, physNum, logNum, numCnots, swapNum)   



def writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, path):
    allowedSwaps = np.append(np.argwhere(cm>0), [[0,0]], axis=0) 
    swapSeqs = itertools.product(allowedSwaps, repeat=swapNum)
    for swapSeq in swapSeqs:
        indexed_swaps = list(enumerate(swapSeq))
        for k in range(1, len(layers)):
            swapLits = [(True, "s", u, v, t, layers[k]) for (t, [u,v]) in indexed_swaps]
            for i in range(physNum):
                for j in liveLog:
                    for prev in range(layers[k-1], layers[k]):
                        if k == len(layers)-1: currentRange = [layers[k]]
                        else: currentRange = range(layers[k], layers[ k+1])
                        for current in currentRange:
                            writeHardClause(path, top, swapLits + [(False, "x", i, j, prev), (True, "x", composeSwaps(swapSeq, physNum)[i], j, current)], physNum, logNum, numCnots, swapNum)
                            writeHardClause(path, top, swapLits + [(True, "x", i, j, prev), (False, "x", composeSwaps(swapSeq, physNum)[i], j, current)], physNum, logNum, numCnots, swapNum)



# Soft Constraints #

def writeOptimizationConstraints(swapNum, physNum, numCnots, cm, logNum, routing, path):
    if routing:
        for k in range(numCnots):
            for t in range(swapNum):
                for (u,v) in itertools.product(range(physNum), repeat=2):
                    if u != v:
                        writeSoftClause(path, (1, [(True, "s", u, v, t, k)]), physNum, logNum, numCnots, swapNum)
    else:
        for k in range(1, numCnots):
            for t in range(swapNum):
                for i in physNum:
                    for j in logNum:
                        writeSoftClause(path, (1, [(True, "x", i, j, k-1), (False, "x", i, j, k)]), physNum, logNum, numCnots, swapNum)
                        writeSoftClause(path, (1, [(False, "x", i, j, k-1), (True, "x", i, j, k)]), physNum, logNum, numCnots, swapNum)



def applySwap(swap, current):
    [u, v] = swap
    for i in current.keys():
        if current[i] == u:
            current[i] = v
        elif current[i] == v:
            current[i] = u


def composeSwaps(swapSeq, physNum):
    current = {phys : phys for phys in range(physNum)}
    for swap in swapSeq:
        applySwap(swap, current)
    return current



## Conversion to MaxSat solver input format ## 

def flattenedIndex(lit, physNum, logNum, numCnots, swapNum):
    '''
        Converts the tuple representation of literals into integers
    '''
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    indices = lit[2:]
    if lit[1] == "p":
        pos = np.ravel_multi_index(indices, (physNum, physNum, numCnots))
    elif lit[1] == "r":
        pos = (np.ravel_multi_index(indices, (physNum, physNum, numCnots)) + numP)
    elif lit[1] == "x":
        pos = (np.ravel_multi_index(
            indices, (physNum, logNum, numCnots)) + numP + numR)
    elif lit[1] == "s":
        pos = (np.ravel_multi_index(
            indices, (physNum, physNum, swapNum, numCnots)) + numP + numR + numX)
    elif lit[1] == "b":
        pos = (np.ravel_multi_index(indices, (physNum * physNum,
               swapNum, numCnots)) + numP + numR + numX+numS)
    pos = pos + 1
    if lit[0]:
        pos = -pos
    return pos

def flattenedWeightedClause(clause, physNum, logNum, numCnots, swapNum): return (clause[0], [flattenedIndex(lit, physNum, logNum, numCnots, swapNum) for lit in clause[1]])
def flattenedClause(clause, physNum, logNum, numCnots, swapNum): return [flattenedIndex(lit, physNum, logNum, numCnots, swapNum) for lit in clause]

def writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum):
        flatClause = flattenedClause(clause, physNum, logNum, numCnots, swapNum)
        f.write(str(top))
        f.write(" ")
        for lit in flatClause:
            f.write(str(lit))
            f.write(" ")
        f.write("0\n")

def writeSoftClause(f, clause, physNum, logNum, numCnots, swapNum):
    flattenedClause = flattenedWeightedClause(clause, physNum, logNum, numCnots, swapNum)
    f.write(str(clause[0]))
    f.write(" ")
    for lit in flattenedClause[1]:
        f.write(str(lit))
        f.write(" ")
    f.write("0\n")

## Reading MaxSat solver output ##

def unravel(flatLit, physNum, logNum, numCnots, swapNum):
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * swapNum * physNum * physNum
    flipped = flatLit < 0
    shifted = abs(flatLit) - 1
    if shifted < numP:
        return (flipped, "p", np.unravel_index(shifted, (physNum, physNum, numCnots)))
    elif shifted < (numP + numR):
        return (flipped, "r", np.unravel_index(shifted-numP, (physNum, physNum, numCnots)))
    elif shifted < (numX + numP + numR):
        return (flipped, "x", np.unravel_index(shifted-(numP+numR), (physNum, logNum, numCnots)))
    elif shifted < (numP+numR+numX+numS):
        return (flipped, "s", np.unravel_index(shifted-(numP+numR+numX), (physNum, physNum, swapNum, numCnots)))
    else:
        return (flipped, "b", np.unravel_index(shifted-(numP+numR+numX+numS), (physNum*physNum, swapNum, numCnots)))

def readMaxSatOutput(physNum, logNum, numCnots, swapNum, fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("v"):
                lits = line.split()[1:]      
                return [unravel(int(lit), physNum, logNum, numCnots, swapNum) for lit in lits]
    return []

def readCost(fname):
    best = math.inf
    with open(fname) as f:
        for line in f:
            if line.startswith("o") and int(line.split()[1]) < best:
                best = int(line.split()[1])
    return best


def mappingVars(parseFun, physNum, logNum, numCnots, permNum, fname):
    return map(lambda x: x[2], filter(lambda x: not x[0] and x[1] == "x", parseFun(physNum, logNum, numCnots, permNum, fname)))

## Solving ##

def solve(progName, cm, swapNum, chunks, iterations=100, time_wbo_max = 300, qaoa=False, _routing=True, pname="test", sname="out"):
    ''' The SAT-solving loop. Parses the program, generates corresponding MaxSat instances, and calls the MaxSat Solver '''
    # Controls whether this function's debug is printed (overwrites DEBUG_GLOBAL)
    DEBUG_LOCAL = False
    return_results = {}
    cost = 0  # <-- number of SWAPs added
    time_elapsed_wbo = 0
    logNum = extractQbits(progName)
    physNum = len(cm)
    return_results = {}
    hack = qiskit.QuantumCircuit.from_qasm_file(progName)
    (head, tail) = os.path.split(progName)
    with open(os.path.join(head, "qiskit-" + tail), "w") as f:
        f.write(hack.qasm())
    cnots = extractCNOTs(os.path.join(head, "qiskit-" + tail))
    sorted_cnots = sortCnots(logNum, cnots)
    numCnots = len(cnots)

    layers= range(len(cnots))
    chunkSize = len(layers)//chunks

    if(DEBUG_LOCAL and DEBUG_GLOBAL):
        print(f'logNum={logNum}, physNum={physNum}, cnots={cnots}, numCnots{numCnots}, layers={layers}, chunkSize={chunkSize}')

    currentChunk = 0
    addedSwaps = [0 for _ in range(chunks)]
    negatedModels = [[] for i in range(chunks)]
    time_elapsed_wbo = 0
    while currentChunk < chunks:
        print("current chunk is", currentChunk)
        print("negated", len(negatedModels[currentChunk]), "models")
        if currentChunk == chunks - 1: end = numCnots
        else: end = layers[chunkSize*(currentChunk+1)]
        currentSize = end - layers[chunkSize*(currentChunk)]
        print("current size:", currentSize)
        if negatedModels[currentChunk]:
            if(DEBUG_LOCAL and DEBUG_GLOBAL):
                print(set.intersection(*[set(l)
                      for l in negatedModels[currentChunk]]))
        if currentChunk == 0:
            swapBack = []
            if qaoa and currentChunk == chunks-1:
                swapBack = [[(False, "x", phys, log, currentSize-1), (True, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)] +  [[(True, "x", phys, log, currentSize-1), (False, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)]
            gen_write_s = time.process_time()
            generateAndWriteClauses(logNum, cnots[:end], cnots[:end], cm, swapNum+addedSwaps[0], negatedModels[0] + swapBack, pname+"-"+str(currentChunk)+".cnf", routing=_routing)
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max-time_elapsed_wbo 
            try:
               p = subprocess.Popen(["../lib/Open-WBO-Inc/open-wbo-inc_release", "-algorithm=8", "-iterations="+str(iterations), pname+"-"+str(currentChunk)+".cnf"],  stdout=open( sname + "-chnk0" + ".txt", "w"))
               p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.process_time()
            time_elapsed_wbo += t_f - t_s
        else:
            prevSize = layers[chunkSize*currentChunk] - layers[chunkSize*(currentChunk-1)]
            prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxSatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
            consistencyClauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prevAssignments]
            swapBack = []
            if qaoa and currentChunk == chunks-1:
                print("qaoa")
                initialMapping =  filter(lambda x : x[2] == 0, mappingVars(readMaxSatOutput, physNum, logNum, prevSize, swapNum, sname + "-chnk" + str(0) + ".txt")) 
                swapBack = [[(False, "x", phys, log, currentSize-1)] for (phys, log, _) in initialMapping]
            gen_write_s = time.process_time()
            print("start:", layers[chunkSize*(currentChunk)])
            print("end:", end)
            generateAndWriteClauses(logNum, cnots[:end], cnots[layers[chunkSize*(currentChunk)]:end], cm, swapNum+addedSwaps[currentChunk], consistencyClauses+negatedModels[currentChunk]+swapBack,  pname+"-"+str(currentChunk)+".cnf", routing=_routing)
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max-time_elapsed_wbo 
            try:
                p = subprocess.Popen(["../lib/Open-WBO-Inc/open-wbo-inc_release", "-algorithm=8", "-iterations="+str(iterations), pname+"-"+str(currentChunk)+".cnf"], stdout=open(sname + "-chnk" + str(currentChunk) + ".txt", "w"))
                p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.process_time()
            time_elapsed_wbo += t_f - t_s
        assignments = filter(lambda x : x[2] == currentSize-1, mappingVars(readMaxSatOutput, physNum, logNum, currentSize, swapNum+addedSwaps[currentChunk], sname + "-chnk" + str(currentChunk) + ".txt"))
        if list(assignments): 
            print("chunk", currentChunk, "solved")
            currentChunk = currentChunk+1
        else:
                if len(negatedModels[currentChunk-1]) < 50*(addedSwaps[currentChunk]+1): 
                    print("got stuck on chunk", currentChunk, "backtracking to chunk", currentChunk-1)
                    prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxSatOutput, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
                    negatedModel =  [(True, "x", phys, log, lastGate) for (phys, log, lastGate) in prevAssignments]
                    negatedModels[currentChunk-1].append(negatedModel)
                    currentChunk = currentChunk-1
                else:
                    print("got stuck on chunk", currentChunk, "repeatedly, increasing swap count")
                    addedSwaps[currentChunk] += 1 
    cost=0
    for i in range(chunks):
        with open(sname + "-chnk" + str(i) + ".txt") as f:
            for line in f:
                if line.startswith("o"):
                    count = int(line.split()[1])
        cost += count
    return_results['cost'] = cost
    return_results['time_wbo'] = time_elapsed_wbo
    return return_results




## Converting solutions to circuits, verifying correctness ##

def toQasm(physNum, logNum, numCnots, swapNum, fname, progPath, cm, prevMap, start=0):
    circ = qiskit.QuantumCircuit(16, 16)
    prog = qiskit.QuantumCircuit.from_qasm_file(progPath)
    edges = np.argwhere(cm > 0)
    i = start
    while circ.num_nonlocal_gates() < numCnots:
        circ.compose(*prog[i], inplace=True)
        i += 1
    lits = readMaxSatOutput(physNum, logNum, numCnots, swapNum, fname)

    swaps = [s[2] for s in filter(lambda x : not x[0] and x[1] == "s" and x[2][0] != x[2][1], lits)]
    mappingVars =  [x[2] for x in filter(lambda x : not x[0] and x[1] == "x", lits)]
    logToPhys = { (j,k) : i for (i,j,k) in mappingVars}
    physToLog = { (i,k) : j for (i,j,k) in mappingVars}
    
    swapIndices = [s[3] for s in swaps]
    for k in range(numCnots):
        mapKLog = list(filter(lambda x: x[0][1] == k, logToPhys.items()))
        assert(len(list(mapKLog)) == len(set(mapKLog))), "Invalid solution: non-injective"
        if k == 0 and prevMap: assert mapKLog == prevMap, "Invalid solution: slices aren't consistent"
        mapKPhys = list(filter(lambda x: x[0][1] == k, logToPhys.items()))
        assert(len(list(mapKPhys)) == len(set(mapKPhys))), "Invalid solution: non-function"
        swapsK = filter(lambda s: s[3] == k, swaps)
        justPhys = [s[:2] for s in swapsK]
        for (phys1,phys2) in justPhys:
            assert([phys1, phys2] in edges.tolist()), "Invalid solution: bad swap"
        if k>0:
            for l in range(logNum):
                physPrev = logToPhys[(l,k-1)]
                assert(logToPhys[(l,k)] == composeSwaps(justPhys,physNum)[physPrev]), "Invalid solution: unexpected SWAP"      
    mappedCirc = qiskit.QuantumCircuit(circ.num_qubits)
    prog2prog = {k : k for k in range(circ.num_qubits)}
    cnotCount = 0 
    for j in range(len(circ)):
        if circ[j][0].name == 'cx':
            if cnotCount in swapIndices:
                swapsK = filter(lambda s: s[3] == cnotCount, swaps)
                for s in swapsK:
                    mappedCirc.swap(prog2prog[physToLog[(s[0], cnotCount)]], prog2prog[physToLog[(s[1], cnotCount)]])
                    (prog2prog[physToLog[(s[0], cnotCount)]], prog2prog[physToLog[(s[1], cnotCount)]]) = (prog2prog[physToLog[(s[1], cnotCount)]], prog2prog[physToLog[(s[0], cnotCount)]])
            [c, t] = circ[j][1]
            logc, logt = c.index, t.index
            physc, physt = logToPhys[(logc, cnotCount)
                                     ], logToPhys[(logt, cnotCount)]
            assert([physc, physt] in edges.tolist()), "Invalid solution: unsatisfed cnot"
            cnotCount += 1
        qubits = list(map(lambda q : qiskit.circuit.Qubit(q.register, prog2prog[q.index]), circ[j][1]))
        mappedCirc.append(circ[j][0],qubits)
    finalMap = list(filter(lambda x: x[0][1] == numCnots, logToPhys.items()))
    return (mappedCirc, i, finalMap)
          
def toQasmFF(progName, cm, swapNum, chunks, fname):
    pointer = 0
    physNum = len(cm)
    cnots = extractCNOTs(progName)
    logNum = extractQbits(progName)
    numCnots = len(cnots)
    # layers = getLayers(cnots)
    layers = range(len(cnots))
    chunkSize = len(layers)//chunks
    prevMap = None
    circ = qiskit.QuantumCircuit(16, 16)
    for i in range(chunks):
        if i == chunks - 1: end = numCnots
        else: end = layers[chunkSize*(i+1)]
        currentSize = end - layers[chunkSize*(i)]
        (mapped_circ, gates, finalMap) = toQasm(physNum, logNum, currentSize, swapNum, fname + "-chnk" + str(i) + ".txt", progName, cm, prevMap, start=pointer)
        pointer = gates
        prevMap = finalMap
        circ.compose(mapped_circ, inplace=True)
    return circ.qasm()


def transpile(progname, cm, swapNum, chunks, cnfname, sname):
    solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname)
    return toQasmFF(os.path.join(os.path.split(progname)[0], "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, sname)


if __name__ == "__main__":
   transpile("../examples/4mod7-v0_94.qasm", architectures.ibmTokyo, 1, 1, "out", "test")