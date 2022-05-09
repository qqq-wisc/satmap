from curses.panel import top_panel
import itertools
import math
import os
import re
import subprocess
import sys
import time
from ast import Try
from cmath import log
from ctypes.wintypes import tagRECT

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.dagcircuit
from sympy import solve
# from memory_profiler import profile

import architectures

# Controls whether debug is output (overwritten by Local if True)
DEBUG_GLOBAL = True

## Architectures ##
triangle = np.array([[0,1,0], [0,0,1], [1,0,0]])
ibmqx4 = np.array([[0,0,0,0,0],[1,0,0,0,0], [1,1,0,0,0], [0,0,1,0,1],[0,0,1,0,0]])
k5 = np.array([[0,1,1,1,1], [1,0,1,1,1], [1,1,0,1,1], [1,1,1,0,1],[1,1,1,1,0]])
ring5 = np.array([[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],[1,0,0,0,0]])
ibmqx5 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0], [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]])

ibmtokyo = np.array([[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], 
                     [1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0], [0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0], [0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0], [0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0], [0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0], 
                     [0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0], [0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0], [0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1], [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1], 
                     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0]])

ibmtokyoSym = np.array([[0,1,0,0,0,1,0,0,0,0], [1,0,1,0,0,0,1,1,0,0], [0,1,0,1,0,0,1,1,0,0], [0,0,1,0,1,0,0,0,1,1], [0,0,0,1,0,0,0,0,1,1],
                        [1,0,0,0,0,0,1,0,0,0], [0,1,1,0,0,1,0,1,0,0], [0,1,1,0,0,0,1,0,1,0], [0,0,0,1,1,0,0,1,0,1], [0,0,0,1,1,0,0,0,1,0]])

ibmToronto = np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]] )

#ibmTorontoSym = 
advArch = np.array([[0,1,0,0,1], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,1,0]])


amandaXample = [(0,2),(1,2),(0,1), (3,5), (3,4), (4,5)]

def undirected(cm):
    (n, m) = np.shape(cm)
    undirected = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            if cm[i][j] == 1 or cm[j][i] == 1:
                undirected[i][j] = 1
    return undirected

## Getting info out of openQASM files ##


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

## Constructing Clauses ##


def genClauses(logNum, cnots, cm, opt):
    '''
        Returns a list of clauses ready to be printed to a DIMACS file and the # of vars within
        A clause is just a list of ints, each representing a literal.
        A negative value corresponds to the negation of a variable.

    '''
    physNum = len(cm)
    allPermutations = list(itertools.permutations(range(physNum)))
    permNum = len(allPermutations)
    numCnots = len(cnots)
    liveLog = [c for (c, _) in cnots] + [t for (_, t) in cnots]
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numY = numCnots * len(allPermutations)
    numB = numCnots * (len(allPermutations)-1)
    varNum = numX + numP + numR + numY + numB
    hardClauses = (
        funConConstraint(numCnots, liveLog, physNum) +
        injectivityConstraint(numCnots, liveLog, physNum) +
        cnotConstraint(logNum, cnots, cm) +
        permutationChoiceConstraint(allPermutations, numCnots, physNum) +
        permutationEffectConstraint(
            allPermutations, numCnots, liveLog, physNum)
    )
    flattenedClauses = [flattenedClause(
        clause, physNum, logNum, numCnots, permNum) for clause in hardClauses]
    if not opt:
        return (varNum, flattenedClauses)
    softClauses = optimizationConstraints(
        allPermutations, physNum, logNum, numCnots, cm)
    flattenedWeightedClauses = [flattenedWeightedClause(
        clause, physNum, logNum, numCnots, permNum) for clause in softClauses]
    # a loose upper bound on the sum of the weights of all soft clauses
    top = (7*physNum*physNum*permNum*numCnots)+(4*numR*numCnots)+1
    return (varNum, flattenedClauses, flattenedWeightedClauses, top)


def genClausesSwaps(logNum, cnots, cm, swapNum, opt):
    ''' 
        Returns a list of clauses ready to be printed to a DIMACS file and the # of vars within
        A clause is just a list of ints, each representing a literal.
        A negative value corresponds to the negation of a variable.

    '''
    physNum = len(cm)
    numCnots = len(cnots)
    layers = getLayers(cnots)
    liveLog = set([c for (c, _) in cnots] + [t for (_, t) in cnots])
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    numB = numCnots * (physNum * physNum * swapNum-1)
    varNum = numX + numP + numR + numS + numB
    hardClauses = (
        funConConstraint(numCnots, liveLog, physNum) +
        injectivityConstraint(numCnots, liveLog, physNum) +
        cnotConstraint(liveLog, cnots, cm) +
        swapChoiceConstraint(swapNum, layers, liveLog) +
        swapEffectConstraint(swapNum, layers, liveLog, physNum, cm) +
        validSwap(swapNum, layers, liveLog, cm)
    )
    flattenedClauses = [flattenedClauseSwap(
        clause, physNum, logNum, numCnots, swapNum) for clause in hardClauses]
    if not opt:
        return (varNum, flattenedClauses)
    softClauses = optimizationConstraintsSwapsUW(
        swapNum, physNum, liveLog, numCnots, cm)
    flattenedWeightedClauses = [flattenedWeightedClauseSwap(
        clause, physNum, logNum, numCnots, swapNum) for clause in softClauses]
    # a loose upper bound on the sum of the weights of all soft clauses
    top = numS + numR + 1
    return (varNum, flattenedClauses, flattenedWeightedClauses, top)


def genClausesSwapsG(logNum, cnots, cm, swapNum, opt):
    ''' 
        Returns a list of clauses ready to be printed to a DIMACS file and the # of vars within
        A clause is just a list of ints, each representing a literal.
        A negative value corresponds to the negation of a variable.

    '''
    physNum = len(cm)
    numCnots = len(cnots)
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = physNum * physNum * swapNum
    numB = (physNum * physNum * swapNum)
    numL = numCnots*(swapNum+1)
    varNum = numX + numP + numR + numS + numB + numL
    hardClauses = ( 
                    funConConstraint(numCnots, logNum, physNum) +
                    injectivityConstraint(numCnots, logNum, physNum) +
                    cnotConstraint(logNum, cnots, cm) +
                    swapChoiceConstraintG(swapNum, numCnots, physNum, cm) +
                    swapEffectConstraintG(swapNum, numCnots, liveLog, physNum, cm) +
                    partitionExactOne(numCnots, swapNum) +
                    partitionOrdered(numCnots, swapNum)
                   )
    print(len(hardClauses))
    flattenedClauses = [flattenedClauseSwapG(clause, physNum, logNum, numCnots, swapNum) for clause in hardClauses]
    if not opt: return (varNum, flattenedClauses)
    softClauses = optimizationConstraintsSwapsG(swapNum, physNum, logNum, numCnots, cm)
    flattenedWeightedClauses = [flattenedWeightedClauseSwapG(clause, physNum, logNum, numCnots, swapNum) for clause in softClauses]
    # a loose upper bound on the sum of the weights of all soft clauses
    top = numS + numR + 1
    return (varNum, flattenedClauses, flattenedWeightedClauses, top)


def genClausesSwapsFF(logNum, liveCnots, cnots, cm, swapNum, ffClauses, opt, sauc=False):
    ''' 
        Returns a list of clauses ready to be printed to a DIMACS file and the # of vars within
        A clause is just a list of ints, each representing a literal.
        A negative value corresponds to the negation of a variable.

    '''
    physNum = len(cm)
    numCnots = len(cnots)
    # layers = list(range(len(cnots)))
    layers = getLayers(cnots)
    liveLog = set([c for (c,_) in liveCnots] + [t for (_,t) in liveCnots])
    numX = numCnots * logNum * physNum 
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    numB = numCnots * (physNum * physNum * swapNum-1)
    #varNum = numX + numP + numR + numS + numB
    hardClauses = ( 
                    funConConstraint(numCnots, liveLog, physNum) +
                    injectivityConstraint(numCnots, liveLog, physNum) +
                    cnotConstraint(liveLog, cnots, cm) +
                    swapChoiceConstraint(swapNum, layers, liveLog) +
                    swapEffectConstraint(swapNum, layers, liveLog, physNum, cm) +
                    validSwap(swapNum, layers, liveLog, cm) +
                    ffClauses
                   )
    print("about", len(hardClauses)//1000, "thousand hard clauses")
    print("consuming", sys.getsizeof(hardClauses)/1000000, "megabytes of memory")
    flattenedClauses = [flattenedClauseSwap(clause, physNum, logNum, numCnots, swapNum) for clause in hardClauses]
    saucFiles = ["temp.cnf", "temp.cnf.g", "temp.cnf.Sym.cnf", "temp.cnf.SymOnly.cnf"]
    for fname in saucFiles:
        try: os.remove(fname)
        except OSError: pass
    varNum =  max(list(map(lambda l: abs(max(l, key=abs)), flattenedClauses)))
    symClauses = []
    if sauc:
        writeClauses((varNum, flattenedClauses), fname="temp.cnf")
        p = subprocess.Popen(["shatter.pl", "temp.cnf"])
        time.sleep(60)
        p.kill()
        time.sleep(6)
        subprocess.run(["gap2cnf", "-f", "temp.cnf", "-t", "temp.cnf.txt"])
        with open("temp.cnf.SymOnly.cnf") as f:
            clauseLines = f.readlines()[1:]
            symClauses = [ line.split()[:-1] for line in clauseLines ]
    if not opt: return (varNum, flattenedClauses)
    softClauses = optimizationConstraintsSwapsUW(swapNum, physNum, liveLog, numCnots, cm)
    flattenedWeightedClauses = [flattenedWeightedClauseSwap(clause, physNum, logNum, numCnots, swapNum) for clause in softClauses]
    # a loose upper bound on the sum of the weights of all soft clauses
    top = numS + numR + 1
    return (varNum, flattenedClauses+symClauses, flattenedWeightedClauses, top)


def genWriteClausesSwapsFF(logNum, liveCnots, cnots, cm, swapNum, ffClauses, opt, path, layering=False, sauc=False):
    ''' 
        Returns a list of clauses ready to be printed to a DIMACS file and the # of vars within
        A clause is just a list of ints, each representing a literal.
        A negative value corresponds to the negation of a variable.

    '''
    physNum = len(cm)
    numCnots = len(cnots)
    print( list(range(len(cnots))))
    if layering:
        layers = getLayers(cnots)
    else: layers = list(range(len(cnots)))
    liveLog = set([c for (c,_) in liveCnots] + [t for (_,t) in liveCnots])
    numX = numCnots * logNum * physNum 
    numP = physNum * physNum * numCnots
    numR = numP
    numS = numCnots * physNum * physNum * swapNum
    numB = numCnots * (physNum * physNum * swapNum-1)
    top = numS + numR + 1
    #varNum = numX + numP + numR + numS + numB
    with open(path, "w") as f:
        f.write("p wcnf " + str(42) + " " + str(42) + " " + str(top) + "\n")
        writeFunConConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f)
        writeInjectivityConstraint(numCnots, liveLog, physNum, logNum, swapNum, top, f)
        writeCnotConstraint(cnots, cm, physNum, logNum, swapNum, top, f)
        writeSwapChoiceConstraint(swapNum, layers, liveLog, physNum, logNum, numCnots, top, f)
        writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, f)
        writeValidSwap(swapNum, layers, liveLog, cm, physNum, logNum, numCnots, top, f)
        for clause in ffClauses:
            writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum)
        writeOptimizationConstraintsSwapsUW(swapNum, physNum, liveLog, numCnots, cm, logNum, f)
    # a loose upper bound on the sum of the weights of all soft clauses

def flattenedWeightedClause(clause, physNum, logNum, numCnots, swapNum): return (clause[0], [flattenedIndex(lit, physNum, logNum, numCnots, swapNum) for lit in clause[1]])

def flattenedClause(clause, physNum, logNum, numCnots, swapNum): return [
    flattenedIndex(lit, physNum, logNum, numCnots, swapNum) for lit in clause]

def flattenedWeightedClauseSwap(clause, physNum, logNum, numCnots, swapNum): return (clause[0], [flattenedIndexSwap(lit, physNum, logNum, numCnots, swapNum) for lit in clause[1]])


def flattenedClauseSwap(clause, physNum, logNum, numCnots, swapNum): return [flattenedIndexSwap(lit, physNum, logNum, numCnots, swapNum) for lit in clause]

def flattenedWeightedClauseSwapG(clause, physNum, logNum, numCnots, swapNum): return (
    clause[0], [flattenedIndexSwapG(lit, physNum, logNum, numCnots, swapNum) for lit in clause[1]])

def flattenedClauseSwapG(clause, physNum, logNum, numCnots, swapNum): return [
    flattenedIndexSwapG(lit, physNum, logNum, numCnots, swapNum) for lit in clause]


# def writeClause(clause, physNum, logNum, numCnots, permNum, handle, hard=True):
#     if hard:
#         flattened = flattenedClause(clause, physNum, logNum, numCnots, permNum)
#         f.write(str(top))
#         f.write(" ")
#         for lit in clause:
#             f.write(str(lit))
#             f.write(" ")
#         f.write("0\n")
#     else:
#         flattened = flattenedWeightedClause(clause, physNum, logNum, numCnots, permNum)


def funConConstraint(numCnots, liveLog, physNum):
    atLeastOne = []
    atMostOne = []
    for k in range(numCnots):
        for j in liveLog:
            atLeastOneJ = []
            for i in range(physNum):
                atLeastOneJ.append((False, "x", i, j, k))
                for i2 in range(i):
                    atMostOne.append(
                        [(True, "x", i2, j, k), (True, "x", i, j, k)])
            atLeastOne.append(atLeastOneJ)
    return atLeastOne + atMostOne

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

def injectivityConstraint(numCnots, liveLog, physNum):
    atMostOne = []
    for i in range(physNum):
        for j in liveLog:
            for k in range(numCnots):
                for j2 in range(j):
                    atMostOne.append(
                        [(True, "x", i, j2, k), (True, "x", i, j, k)])
    return atMostOne

def writeInjectivityConstraint(numCnots, liveLog, physNum, logNum,  swapNum, top, path):
    for i in range(physNum):
        for j in liveLog:
            for k in range(numCnots):
                for j2 in range(j):
                   writeHardClause(path, top, [(True, "x", i, j2, k), (True,"x",i,j,k)], physNum, logNum, numCnots, swapNum)


def cnotConstraint(liveLog, cnots, cm):
    clauses = []
    for k in range(len(cnots)):
        (c, t) = cnots[k]
        edgeUsed = []
        verticesUsed = []
        nonzeroIndices = np.argwhere(cm > 0)
        for edge in nonzeroIndices:
            [u, v] = edge
            edgeUsed.append((False, "p", u, v, k))
            edgeUsed.append((False, "r",  u, v, k))
            verticesUsed = verticesUsed + [[(False, "x", u, c, k), (True, "p", u, v, k)],
                                           [(False, "x", v, t, k),
                                            (True, "p", u, v, k)],
                                           [(False, "x", u, t, k),
                                            (True, "r", u, v, k)],
                                           [(False, "x", v, c, k), (True, "r", u, v, k)]]
        clauses = clauses + [edgeUsed] + verticesUsed
    return clauses

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



# def permutationChoiceConstraint(allPermutations, numCnots, physNum):
#     clauses = []
#     for k in range(numCnots):
#         atLeastOne = []
#         atMostOne = []
#         for i in range(len(allPermutations)):
#             atLeastOne.append((False, "y", i, k))
#             for j in range(math.ceil(math.log2(len(allPermutations)))):
#                     atMostOne.append([(True, "y", i, k), (not bin(i)[:-(k+1)] == 1,"b", i, k)])
#         clauses = clauses + [atLeastOne] + atMostOne
#     return clauses


def permutationChoiceConstraint(allPermutations, numCnots, physNum):
    clauses = []
    for k in range(numCnots):
        atLeastOne = []
        atMostOne = []
        for i in range(len(allPermutations)-1):
            atLeastOne.append((False, "y", i, k))
            if i != 0:
                atMostOne.append([(True, "y", i, k), (False, "b", i-1, k)])
                atMostOne.append([(True, "y", i, k), (True, "b", i, k)])
                atMostOne.append(
                    [(True, "b", i-1, k), (False, "b", i, k), (False, "y", i, k)])
            atMostOne.append([(False, "b", i, k), (True, "b", i+1, k)])
        clauses = clauses + [atLeastOne] + atMostOne
    return clauses


def swapChoiceConstraint(swapNum, layers, liveLog):
    clauses = []
    for k in layers:
        for t in range(swapNum):
            atLeastOne = []
            atMostOne = []
            for (u, v) in itertools.product(liveLog, repeat=2):
                i = 0
                atLeastOne.append((False, "s", u, v, t, k))
                if i != 0:
                    atMostOne.append(
                        [(True, "s", u, v, t, k), (False, "b", i-1, t, k)])
                    atMostOne.append(
                        [(True, "s", u, v, t, k), (True, "b", i, t, k)])
                    atMostOne.append(
                        [(True, "b", i-1, t, k), (False, "b", i, t, k), (False, "s", u, v, t, k)])
                atMostOne.append(
                    [(False, "b", i, t, k), (True, "b", i+1, t, k)])
                i += 1
            clauses = clauses + [atLeastOne] + atMostOne
    return clauses

def writeSwapChoiceConstraint(swapNum, layers, liveLog, physNum, logNum, numCnots,top, path):
    for k in layers:
        for t in range(swapNum):
            atLeastOne = []
            for (u,v) in itertools.product(liveLog, repeat=2):
                i = 0
                atLeastOne.append((False, "s", u, v, t, k))
                if i != 0:
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (False,"b", i-1, t, k)], physNum, logNum, numCnots, swapNum)
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (True, "b", i, t, k)], physNum, logNum, numCnots, swapNum)
                    writeHardClause(path, top, [(True,"b", i-1, t, k), (False, "b", i, t, k), (False, "s", u, v, t, k)], physNum, logNum, numCnots, swapNum)
                writeHardClause(path, top, [(False, "b", i, t, k), (True,"b", i+1, t, k)], physNum, logNum, numCnots, swapNum)
                i += 1
            writeHardClause(path, top, atLeastOne, physNum, logNum, numCnots, swapNum)   


def validSwap(swapNum, layers, liveLog, cm):
    clauses = []
    for (u, v) in itertools.product(liveLog, repeat=2):
        for k in layers:
            for t in range(swapNum):
                for (p1, p2) in np.argwhere(cm > 0):
                    
                    clauses.append(
                        [(True, "s", u, v, t, k), (False, "x", p1, u, k), (True, "x", p2, v, k)])
    return clauses

def writeValidSwap(swapNum, layers, liveLog, cm, physNum, logNum, numCnots, top, path):
    for (u,v) in itertools.product(liveLog, repeat=2):
        for k in layers:
            for t in range(swapNum):
                for p1 in range(physNum):
                    adjacents = [(False, "x", p2[0], v, k) for p2 in np.argwhere(cm[p1]>0)] + [(False, "x", p1, v, k)]
                    writeHardClause(path, top, [(True, "s", u, v, t, k), (True, "x", p1, u, k)] + adjacents, physNum, logNum, numCnots, swapNum)
                        

# def writeValidSwap(swapNum, layers, liveLog, cm, physNum, logNum, numCnots, top, path):
#     for (u,v) in itertools.product(liveLog, repeat=2):
#         for k in layers:
#             for t in range(swapNum):
#                 for (p1, p2) in np.argwhere(cm>0):
#                     writeHardClause(path, top, [(True, "s", u, v, t, k), (True, "x", p1, u, k), (True, "x", p2, v, k)], physNum, logNum, numCnots, swapNum)   


def swapChoiceConstraintG(swapNum, numCnots, physNum, cm):
    clauses = []
    for t in range(swapNum):
        atLeastOne = []
        atMostOne = []
        allowedSwaps = [[0, 0]] + list(np.argwhere(cm > 0))
        for i in range(len(allowedSwaps)):
            [u, v] = allowedSwaps[i]
            atLeastOne.append((False, "s", u, v, t))
            if i != 0:
                atMostOne.append([(True, "s", u, v, t), (False, "b", i-1, t)])
                atMostOne.append([(True, "s", u, v, t), (True, "b", i, t)])
                atMostOne.append(
                    [(True, "b", i-1, t), (False, "b", i, t), (False, "s", u, v, t)])
            atMostOne.append([(False, "b", i, t), (True, "b", i+1, t)])
        clauses = clauses + [atLeastOne] + atMostOne
    return clauses


def permutationEffectConstraint(allPermutations, numCnots, liveLog, physNum):
    effects = []
    for p in range(len(allPermutations)):
        for i in range(physNum):
            for j in liveLog:
                for k in range(1, numCnots):
                    effects.append([(True, "y", p, k), (False, "x", i, j, k-1),
                                   (True, "x", allPermutations[p][i], j, k)])
                    effects.append([(True, "y", p, k), (True, "x", i, j, k-1),
                                   (False, "x", allPermutations[p][i], j, k)])
    return effects


def swapEffectConstraint(swapNum, layers, liveLog, physNum, cm):
    effects = []
    allowedSwaps = itertools.product(liveLog, repeat=2)
    swapSeqs = itertools.product(allowedSwaps, repeat=swapNum)
    for swapSeq in swapSeqs:
        indexed_swaps = list(enumerate(swapSeq))
        for k in layers[1:]:
            swapLits = [(True, "s", u, v, t, k) for (t, [u,v]) in indexed_swaps]
            for i in range(physNum):
                for j in liveLog:
                    effects.append(swapLits + [(False, "x", i, j, k-1), (True, "x", i, composeSwaps(swapSeq, liveLog)[j], k)])
                    effects.append(swapLits + [(True, "x", i, j, k-1), (False, "x", i, composeSwaps(swapSeq, liveLog)[j], k)])
    return effects

def writeSwapEffectConstraint(swapNum, layers, liveLog, physNum, cm, logNum, numCnots, top, path):
    allowedSwaps = itertools.product(liveLog, repeat=2)
    #print(list(allowedSwaps))
    swapSeqs = itertools.product(allowedSwaps, repeat=swapNum)
    cnot_indices = range(numCnots)
    for swapSeq in swapSeqs:
        indexed_swaps = list(enumerate(swapSeq))
        #for k in layers[1:]:
        for k in range(1, len(layers)):
            swapLits = [(True, "s", u, v, t, layers[k]) for (t, [u,v]) in indexed_swaps]
           # swapLits = [(True, "s", u, v, t, k) for (t, [u,v]) in indexed_swaps]
            for i in range(physNum):
                for j in liveLog:
                    for prev in range(layers[k-1], layers[k]):
                        if k == len(layers)-1: currentRange = [layers[k]]
                        else: currentRange = range(layers[k], layers[ k+1])
                        for current in currentRange:
                            writeHardClause(path, top, swapLits + [(False, "x", i, j, prev), (True, "x", i, composeSwaps(swapSeq, liveLog)[j], current)], physNum, logNum, numCnots, swapNum)
                            writeHardClause(path, top, swapLits + [(True, "x", i, j, prev), (False, "x", i, composeSwaps(swapSeq, liveLog)[j], current)], physNum, logNum, numCnots, swapNum)

def swapEffectConstraintG(swapNum, numCnots, logNum, physNum, cm):
    effects = []
    allowedSwaps = [[i, i] for i in range(physNum)] + list(np.argwhere(cm > 0))
    for k in range(1, numCnots):
        for t in range(swapNum+1):
            for i in range(physNum):
                for j in range(logNum):
                    if k == 1:
                        swapSeqs = itertools.product(allowedSwaps, repeat=t)
                        for swapSeq in swapSeqs:
                            swapLits = [(True, "s", u, v, t)
                                        for (t, [u, v]) in enumerate(swapSeq)]
                            effects.append(
                                swapLits + [(True, "l", t, k), (False, "x", i, j, k-1), (True, "x", composeSwaps(swapSeq, physNum)[i], j, k)])
                            effects.append(swapLits + [(True, "l", t, k), (True, "x", i, j, k-1),
                                           (False, "x", composeSwaps(swapSeq, physNum)[i], j, k)])
                    else:
                        for t1 in range(t+1):
                            swapSeqs = itertools.product(
                                allowedSwaps, repeat=t-t1)
                            for swapSeq in swapSeqs:
                                swapLits = [(True, "s", u, v, t)
                                            for (t, [u, v]) in enumerate(swapSeq)]
                                effects.append(swapLits + [(True, "l", t1, k-1), (True, "l", t, k), (
                                    False, "x", i, j, k-1), (True, "x", composeSwaps(swapSeq, physNum)[i], j, k)])
                                effects.append(swapLits + [(True, "l", t1, k-1), (True, "l", t, k), (
                                    True, "x", i, j, k-1), (False, "x", composeSwaps(swapSeq, physNum)[i], j, k)])
    return effects


def partitionExactOne(numCnots, swapNum):
    atLeastOne = []
    atMostOne = []
    for k in range(numCnots):
        atLeastOnek = []
        for t in range(swapNum+1):
            atLeastOnek.append((False, "l", t, k))
            for t2 in range(t):
                atMostOne.append([(True, "l", t2, k), (True, "l", t, k)])
        atLeastOne.append(atLeastOnek)
    return atLeastOne + atMostOne


def partitionOrdered(numCnots, swapNum):
    ordered = []
    for k in range(1, numCnots):
        for t in range(swapNum+1):
            predecessorLits = [(False, "l", t1, k-1) for t1 in range(t+1)]
            constraint = predecessorLits + [(True, "l", t, k)]
            ordered.append(constraint)
    return ordered


def optimizationConstraints(allPermutations, physNum, logNum, numCnots, cm):
    clauses = []
    for k in range(numCnots):
        for i in range(len(allPermutations))[1:]:
            clauses.append(
                (7*minSwaps(cm, allPermutations[i]), [(True, "y", i, k)]))
        for a in range(physNum):
            for b in range(physNum):
                clauses.append((4, [(True, "r", a, b, k)]))
    return clauses


def optimizationConstraintsSwaps(swapNum, physNum, logNum, numCnots, cm):
    clauses = []
    for k in range(numCnots):
        for t in range(swapNum):
            for [u, v] in np.argwhere(cm > 0):
                clauses.append((7, [(True, "s", u, v, t, k)]))
        for a in range(physNum):
            for b in range(physNum):
                clauses.append((4, [(True, "r", a, b, k)]))
    return clauses


def optimizationConstraintsSwapsG(swapNum, physNum, logNum, numCnots, cm):
    clauses = []
    for t in range(swapNum):
        for [u, v] in np.argwhere(cm > 0):
            clauses.append((7, [(True, "s", u, v, t)]))
    for k in range(numCnots):
        for a in range(physNum):
            for b in range(physNum):
                clauses.append((4, [(True, "r", a, b, k)]))
    return clauses


def optimizationConstraintsSwapsFF(swapNum, physNum, logNum, numCnots, cm):
    clauses = []
    for k in range(1, numCnots):
        for t in range(swapNum):
            for [u, v] in np.argwhere(cm > 0):
                clauses.append((7, [(True, "s", u, v, t, k)]))
        for a in range(physNum):
            for b in range(physNum):
                clauses.append((4, [(True, "r", a, b, k)]))
    return clauses


def optimizationConstraintsSwapsUW(swapNum, physNum, liveLog, numCnots, cm):
    clauses = []
    for k in range(numCnots):
        for t in range(swapNum):
            for (u, v) in itertools.product(liveLog, repeat=2):
                if u != v:
                    clauses.append((1, [(True, "s", u, v, t, k)]))
    return clauses
def writeOptimizationConstraintsSwapsUW(swapNum, physNum, liveLog, numCnots, cm, logNum, path):
    for k in range(numCnots):
        for t in range(swapNum):
            for (u,v) in itertools.product(liveLog, repeat=2):
                if u != v:
                    writeSoftClause(path, (1, [(True, "s", u, v, t, k)]), physNum, logNum, numCnots, swapNum)

## Indexing ##

def flattenedIndex(lit, physNum, logNum, numCnots, permNum):
    '''
        Converts my ad-hoc internal tuple representation of literals into integers
    '''
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numY = numCnots * permNum
    indices = lit[2:]
    if lit[1] == "p":
        pos = np.ravel_multi_index(indices, (physNum, physNum, numCnots))
    elif lit[1] == "r":
        pos = (np.ravel_multi_index(indices, (physNum, physNum, numCnots)) + numP)
    elif lit[1] == "x":
        pos = (np.ravel_multi_index(
            indices, (physNum, logNum, numCnots)) + numP + numR)
    elif lit[1] == "y":
        pos = (np.ravel_multi_index(
            indices, (permNum, numCnots)) + numP + numR + numX)
    elif lit[1] == "b":
        pos = (np.ravel_multi_index(indices, (permNum, numCnots)) +
               numP + numR + numX+numY)
    pos = pos + 1
    if lit[0]:
        pos = -pos
    return pos


def flattenedIndexSwap(lit, physNum, logNum, numCnots, swapNum):
    '''
        Converts my ad-hoc internal tuple representation of literals into integers
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


def flattenedIndexSwapG(lit, physNum, logNum, numCnots, swapNum):
    '''
        Converts my ad-hoc internal tuple representation of literals into integers
    '''
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = physNum * physNum * swapNum
    numB = numS
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
            indices, (physNum, physNum, swapNum)) + numP + numR + numX)
    elif lit[1] == "b":
        pos = (np.ravel_multi_index(
            indices, (physNum * physNum, swapNum)) + numP + numR + numX+numS)
    elif lit[1] == "l":
        pos = (np.ravel_multi_index(indices, (swapNum+1, numCnots)) +
               numP + numR + numX+numS + numB)
    pos = pos + 1
    if lit[0]:
        pos = -pos
    return pos


def unravel(flatLit, physNum, logNum, numCnots, permNum):
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numY = numCnots * permNum
    flipped = flatLit < 0
    shifted = abs(flatLit) - 1
    if shifted < numP:
        return (flipped, "p", np.unravel_index(shifted, (physNum, physNum, numCnots)))
    elif shifted < (numP + numR):
        return (flipped, "r", np.unravel_index(shifted-numP, (physNum, physNum, numCnots)))
    elif shifted < (numX + numP + numR):
        return (flipped, "x", np.unravel_index(shifted-(numP+numR), (physNum, logNum, numCnots)))
    elif shifted < (numP+numR+numX+numY):
        return (flipped, "y", np.unravel_index(shifted-(numP+numR+numX), (permNum, numCnots)))
    else:
        return (flipped, "b", np.unravel_index(shifted-(numP+numR+numX+numY), (permNum, math.ceil(math.log2(numCnots)))))


def unravelSwaps(flatLit, physNum, logNum, numCnots, swapNum):
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


def unravelSwapsG(flatLit, physNum, logNum, numCnots, swapNum):
    numX = numCnots * logNum * physNum
    numP = physNum * physNum * numCnots
    numR = numP
    numS = swapNum * physNum * physNum
    numB = numS
    flipped = flatLit < 0
    shifted = abs(flatLit) - 1
    if shifted < numP:
        return (flipped, "p", np.unravel_index(shifted, (physNum, physNum, numCnots)))
    elif shifted < (numP + numR):
        return (flipped, "r", np.unravel_index(shifted-numP, (physNum, physNum, numCnots)))
    elif shifted < (numX + numP + numR):
        return (flipped, "x", np.unravel_index(shifted-(numP+numR), (physNum, logNum, numCnots)))
    elif shifted < (numP+numR+numX+numS):
        return (flipped, "s", np.unravel_index(shifted-(numP+numR+numX), (physNum, physNum, swapNum)))
    elif shifted < (numP+numR+numX+numS+numB):
        return (flipped, "b", np.unravel_index(shifted-(numP+numR+numX+numS), (physNum*physNum, swapNum)))
    else:
        return (flipped, "l", np.unravel_index(shifted-(numP+numR+numX+numS+numB), (swapNum+1, numCnots)))


def minSwaps(cm, permutation, init=None):
    for swapNum in range((len(cm))**2):
        edges = np.argwhere(cm > 0)
        swapSeqs = itertools.product(edges, repeat=swapNum)
        for swapSeq in swapSeqs:
            if init: current = init
            else: current = list((range(len(cm))))
            for swap in swapSeq:
                applySwap(swap, current)
            print(current)
            submap = True
            for key in current:
                if permutation[key] != current[key]:
                    submap = False
            if submap:
                return swapNum
    return "error: couldn't realize permutation"


def applySwap(swap, current):
    [u, v] = swap
    for i in current.keys():
        if current[i] == u:
            current[i] = v
        elif current[i] == v:
            current[i] = u


def composeSwaps(swapSeq, liveLog):
    current = {log: log for log in liveLog}
    for swap in swapSeq:
        applySwap(swap, current)
    return current


## Tests ##
simp = (2, [(0,1),(1,0)], triangle)

ex_1_166 = (3, "examples/ex-1_166.qasm", ibmqx4) 
ex_ham3_102 = (3, "examples/ham3_102.qasm", ibmqx4)
ex_3_17_13 = (3, "examples/3_17_13.qasm", ibmqx4) #
ex_miller_11 = (3, "examples/miller_11.qasm", ibmqx4) #
ex_4gt11_84 = (4, "examples/4gt11_84.qasm", ibmqx4) #
ex_rd32_v0_66 = (4, "examples/rd32-v0_66.qasm", ibmqx4) #
ex_rd32_v1_68 =  (4, "examples/rd32-v1_68.qasm", ibmqx4) #2
ex_4gt11_82 =  (5, "examples/4gt11_82.qasm", ibmqx4) #
ex_4gt11_83 =  (5, "examples/4gt11_83.qasm", ibmqx4) #
ex_4gt13_92 =  (5, "examples/4gt13_92.qasm", ibmqx4) #
ex_4mod5_v0_19 =  (5, "examples/4mod5-v0_19.qasm", ibmqx4) #
ex_4mod5_v0_20 =  (5, "examples/4mod5-v0_20.qasm", ibmqx4) #
ex_4mod5_v1_22 =  (5, "examples/4mod5-v1_22.qasm", ibmqx4) #
ex_4mod5_v1_24 =  (5, "examples/4mod5-v1_24.qasm", ibmqx4) #
ex_alu_v0_27 =  (5, "examples/alu-v0_27.qasm", ibmqx4) #
ex_alu_v1_28 =  (5, "examples/alu-v1_28.qasm", ibmqx4) #
ex_alu_v1_29 =  (5, "examples/alu-v1_29.qasm", ibmqx4) #
ex_alu_v2_33 =  (5, "examples/alu-v2_33.qasm", ibmqx4) #
ex_alu_v3_34 =  (5, "examples/alu-v3_34.qasm", ibmqx4) #
ex_alu_v3_35 =  (5, "examples/alu-v3_35.qasm", ibmqx4) #
ex_alu_v4_37 =  (5, "examples/alu-v4_37.qasm", ibmqx4) #
ex_mod5d1_63 =  (5, "examples/mod5d1_63.qasm", ibmqx4) #
ex_mod5mils_65 = (5, "examples/mod5mils_65.qasm", ibmqx4) #
ex_qe_qft_4 =  (5, "examples/qe_qft_4.qasm", ibmqx4)  #
ex_qe_qft_5 =  (5, "examples/qe_qft_5.qasm", ibmqx4) #
ex_qe_qft_5_8 = (5, "examples/qe_qft_5.qasm", k5)

ex_ising_model_10 = (10, "examples/ising_model_10.qasm", ibmqx5)
ex_qft_10 = (10, "examples/qft_10.qasm", ibmqx5)
ex_sys6_v0 = (10, "examples/sys6-v0_111.qasm", ibmqx5)
ex_sys6_v0_t = (10, "examples/sys6-v0_111.qasm", ibmtokyoSym)
ex3_229 = (6, "examples/ex3_229.qasm", ibmtokyoSym)
mini_alu_305 = (10, "examples/mini_alu_305.qasm", ibmtokyoSym)
rd73_140 = (10, "examples/rd73_140.qasm", ibmtokyoSym)
ex_qft_10_t = (10, "examples/qft_10.qasm", ibmtokyoSym)
ex_ising_model_10 = (10, "examples/ising_model_10.qasm", ibmtokyoSym)
mod5adder_127 = (6, "examples/mod5adder_127.qasm", ibmtokyoSym)
dc1_220 = (10, "examples/dc1_220.qasm", ibmtokyoSym)
adr4_197 = (13, "examples/adr4_197.qasm", ibmtokyoSym)
advCirc = (4, [(0, 1), (1, 2), (2, 3), (0, 3)], advArch)

ibmQx4Tests = [ex_1_166, ex_3_17_13, ex_4gt11_82, ex_4gt11_83, ex_4gt11_84, ex_4gt13_92, ex_4mod5_v0_19,
               ex_4mod5_v0_20, ex_4mod5_v1_22, ex_4mod5_v1_24, ex_alu_v0_27, ex_alu_v1_28, ex_alu_v1_29,
               ex_alu_v2_33, ex_alu_v3_34, ex_alu_v3_35, ex_alu_v4_37, ex_ham3_102, ex_miller_11,
               ex_mod5d1_63, ex_mod5mils_65, ex_qe_qft_4, ex_qe_qft_5, ex_rd32_v0_66, ex_rd32_v1_68]

ibmqx5Tests = [ex_ising_model_10, ex_qft_10, ex_sys6_v0]

ibmTokyoTests = [ex3_229, mini_alu_305, rd73_140]

## Printing to file ##


def writeHardClause(f, top, clause, physNum, logNum, numCnots, swapNum):
        flattenedClause = flattenedClauseSwap(clause, physNum, logNum, numCnots, swapNum)
        f.write(str(top))
        f.write(" ")
        for lit in flattenedClause:
            f.write(str(lit))
            f.write(" ")
        f.write("0\n")

def writeSoftClause(f, clause, physNum, logNum, numCnots, swapNum):
    flattenedClause = flattenedWeightedClauseSwap(clause, physNum, logNum, numCnots, swapNum)
    f.write(str(clause[0]))
    f.write(" ")
    for lit in flattenedClause[1]:
        f.write(str(lit))
        f.write(" ")
    f.write("0\n")


def writeClauses(vcpair, fname="test.cnf"):
    (vars, clauseList) = vcpair
    with open(fname, "w") as f:
        f.write("p cnf " + str(vars) + " " + str(len(clauseList)) + "\n")
        for clause in clauseList:
            for lit in clause:
                f.write(str(lit))
                f.write(" ")
            f.write("0\n")


def writeClausesWeighted(specTup, fname="test.cnf"):
    (vars, hardClauses, softClauses, top) = specTup
    with open(fname, "w") as f:
        f.write("p wcnf " + str(vars) + " " + str(42) + " " + str(top) + "\n")
        for clause in hardClauses:
            f.write(str(top))
            f.write(" ")
            for lit in clause:
                f.write(str(lit))
                f.write(" ")
            f.write("0\n")
        for clause in softClauses:
            f.write(str(clause[0]))
            f.write(" ")
            for lit in clause[1]:
                f.write(str(lit))
                f.write(" ")
            f.write("0\n")

## Read result ##


def readMiniSatFile(physNum, logNum, numCnots, permNum, fname):
    with open(fname) as f:
        lits = f.read().split()[1:-1]
        return [unravel(int(lit), physNum, logNum, numCnots, permNum) for lit in lits]


def readMaxHSFile(physNum, logNum, numCnots, permNum, fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("v"):
                lits = line.split()[1:]
                return [unravel(int(lit), physNum, logNum, numCnots, permNum) for lit in lits]


def readMaxHSFileSwap(physNum, logNum, numCnots, swapNum, fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("v"):
                lits = line.split()[1:]      
                return [unravelSwaps(int(lit), physNum, logNum, numCnots, swapNum) for lit in lits]
    return []


def readMaxHSFileSwapG(physNum, logNum, numCnots, swapNum, fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("v"):
                lits = line.split()[1:]
                for lit in lits:
                    print(unravelSwapsG(int(lit), physNum, logNum, numCnots, swapNum))
                return [unravelSwapsG(int(lit), physNum, logNum, numCnots, swapNum) for lit in lits]


def readCost(fname):
    best = math.inf
    with open(fname) as f:
        for line in f:
            if line.startswith("o") and int(line.split()[1]) < best:
                best = int(line.split()[1])
    return best


def mappingVars(parseFun, physNum, logNum, numCnots, permNum, fname):
    return map(lambda x: x[2], filter(lambda x: not x[0] and x[1] == "x", parseFun(physNum, logNum, numCnots, permNum, fname)))


def solveInstance(logNum, cnots, cm, pname="test.cnf", sname="out.txt", opt=True):
    ''' Constructs a SAT instance, writes it to a file, calls the SAT solver, and returns some of the variables set to true '''
    physNum = len(cm)
    permNum = len(list((itertools.permutations(range(physNum)))))
    numCnots = len(cnots)
    if opt:
        specTup = genClauses(logNum, cnots, cm, True)
        writeClausesWeighted(specTup, pname)
        subprocess.run(["./open-wbo-inc_static", pname], stdout=open(sname, "w"))
        # return mappingVars(readMaxHSFile, physNum, logNum, numCnots, permNum, sname)
    else:
        vcPair = genClauses(logNum, cnots, cm, False)
        writeClauses(vcPair, pname)
        subprocess.run(["minisat", pname, sname])
        return mappingVars(readMiniSatFile, physNum, logNum, numCnots, permNum, sname)


def solveSwaps(logNum, cnots, cm, swapNum, pname="test.cnf", sname="out.txt"):
    ''' Constructs a SAT instance, writes it to a file, calls the SAT solver, and returns some of the variables set to true '''
    physNum = len(cm)
    numCnots = len(cnots)
    specTup = genClausesSwaps(logNum, cnots, cm, swapNum, True)
    writeClausesWeighted(specTup, pname)
    p = subprocess.Popen(["./open-wbo-inc_static", pname],
                         stdout=open(sname, "w"))
    return list(mappingVars(readMaxHSFileSwap, physNum, logNum, numCnots, swapNum, sname))


def solveSwapsG(logNum, cnots, cm, swapNum, pname="test.cnf", sname="out.txt"):
    ''' Constructs a SAT instance, writes it to a file, calls the SAT solver, and returns some of the variables set to true '''
    physNum = len(cm)
    numCnots = len(cnots)
    specTup = genClausesSwapsG(logNum, cnots, cm, swapNum, True)
    writeClausesWeighted(specTup, pname)
    p = subprocess.Popen(["./open-wbo-inc_static", pname],
                         stdout=open(sname, "w"))

    return mappingVars(readMaxHSFileSwapG, physNum, logNum, numCnots, swapNum, sname)

# def solveSwapsFF(logNum, cnots, cm, swapNum, chunks, opt=True, pname="test.cnf", sname="out.txt"):
#     ''' Constructs a SAT instance, writes it to a file, calls the SAT solver, and returns some of the variables set to true '''
#     physNum = len(cm)
#     numCnots = len(cnots)
#     layers = getLayers(cnots)
#     print(layers)
#     chunkSize = len(layers) // chunks
#     initSpecTup = genClausesSwaps(logNum, cnots[:chunkSize], cm, swapNum, True)
#     writeClausesWeighted(initSpecTup, pname)
#     p = subprocess.Popen(["open-wbo-inc_static", pname], stdout=open( sname + "-chnk0" + ".txt", "w"))
#     #time.sleep(chunkSize)
#     #p.terminate()
#     totalCost = readCost(sname + "-chnk0" + ".txt")
#     for i in range(1,chunks):
#         mapped = False
#         prevAddedSwaps = 0
#         if i > 1: prevAddedSwaps = addedSwaps
#         addedSwaps = 0
#         while not mapped:
#             prevAssignments = filter(lambda x : x[2] == chunkSize-1, mappingVars(readMaxHSFileSwap, physNum, logNum, chunkSize, swapNum+prevAddedSwaps, sname + "-chnk" + str(i-1) + ".txt"))
#             consistencyClauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prevAssignments]
#             if i == chunks - 1: end = numCnots
#             else: end = layers[chunkSize*(i+1)]
#             specTup = genClausesSwapsFF(logNum, cnots[layers[chunkSize*(i)-1]:end], cm, swapNum+addedSwaps, consistencyClauses, True)
#             writeClausesWeighted(specTup, pname)
#             p = subprocess.Popen(["open-wbo-inc_static", pname], stdout=open(sname + "-chnk" + str(i) + ".txt", "w"))
#             # time.sleep(20*chunkSize*(swapNum+addedSwaps))
#             # p.terminate()
#             assignments = filter(lambda x : x[2] == chunkSize-1, mappingVars(readMaxHSFileSwap, physNum, logNum, chunkSize, swapNum+addedSwaps, sname + "-chnk" + str(i) + ".txt"))
#             if list(assignments):
#                 mapped = True
#             else:
#                 addedSwaps = addedSwaps + 1
#                 print("got stuck, increasing swap count to", swapNum + addedSwaps, "on chunk", i)
#         totalCost = totalCost + readCost(sname + "-chnk" + str(i) + ".txt")
#         print("chunk", i, "solved, current cost:", totalCost)
#     return totalCost


def solveSwapsFF(logNum, progName, cm, swapNum, chunks, iterations=100, time_wbo_max = 10, qaoa=False, pname="test", sname="out"):
    ''' Constructs a SAT instance, writes it to a file, calls the SAT solver, and returns some of the variables set to true '''

    # Controls whether this function's debug is printed (overwrites DEBUG_GLOBAL)
    DEBUG_LOCAL = False

    # TODO sandwich open-wbo calls in time() so we can measure how much time is spent in wbo instances
    return_results = {}
    cost = 0  # <-- number of SWAPs added
    time_elapsed_wbo = 0

    physNum = len(cm)
    return_results = {}
    hack = qiskit.QuantumCircuit.from_qasm_file(progName)
    (head, tail) = os.path.split(progName)
    with open(os.path.join(head, "qiskit-" + tail), "w") as f:
        f.write(hack.qasm())
    cnots = extractCNOTs(os.path.join(head, "qiskit-" + tail))
    sorted_cnots = sortCnots(logNum, cnots)
    numCnots = len(cnots)
    # layers = getLayers(sorted_cnots)
    # print(layers)
    # print(len(layers))
    layers= range(len(cnots))
    #print(layers)
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
                #print("qaoa")
                swapBack = [[(False, "x", phys, log, currentSize-1), (True, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)] +  [[(True, "x", phys, log, currentSize-1), (False, "x", phys, log, 0) ] for phys in range(physNum) for log in range(logNum)]
            gen_write_s = time.process_time()
            # initSpecTup = genClausesSwapsFF(logNum, cnots[:end], cnots[:end], cm, swapNum+addedSwaps[0], negatedModels[0], True)
            # writeClausesWeighted(initSpecTup, pname+"-"+str(currentChunk)+".cnf")
            genWriteClausesSwapsFF(logNum, cnots[:end], cnots[:end], cm, swapNum+addedSwaps[0], negatedModels[0] + swapBack, True,  pname+"-"+str(currentChunk)+".cnf")
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max-time_elapsed_wbo 
            try:
               p = subprocess.Popen(["./open-wbo-inc_static", "-algorithm=8", "-iterations="+str(iterations), pname+"-"+str(currentChunk)+".cnf"],  stdout=open( sname + "-chnk0" + ".txt", "w"))
               p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.process_time()
            time_elapsed_wbo += t_f - t_s
        else:
            prevSize = layers[chunkSize*currentChunk] - layers[chunkSize*(currentChunk-1)]
            prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxHSFileSwap, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
            consistencyClauses = [[(False, "x", phys, log, 0)] for (phys, log, _) in prevAssignments]
            swapBack = []
            if qaoa and currentChunk == chunks-1:
                print("qaoa")
                initialMapping =  filter(lambda x : x[2] == 0, mappingVars(readMaxHSFileSwap, physNum, logNum, prevSize, swapNum, sname + "-chnk" + str(0) + ".txt")) 
                swapBack = [[(False, "x", phys, log, currentSize-1)] for (phys, log, _) in initialMapping]
            gen_write_s = time.process_time()
            print("start:", layers[chunkSize*(currentChunk)])
            print("end:", end)
            genWriteClausesSwapsFF(logNum, cnots[:end], cnots[layers[chunkSize*(currentChunk)]:end], cm, swapNum+addedSwaps[currentChunk], consistencyClauses+negatedModels[currentChunk]+swapBack, True, pname+"-"+str(currentChunk)+".cnf")
            #specTup = genClausesSwapsFF(logNum, cnots[:end], cnots[layers[chunkSize*(currentChunk)]:end], cm, swapNum+addedSwaps[currentChunk], consistencyClauses+negatedModels[currentChunk]+swapBack, True)
            #writeClausesWeighted(specTup, pname+"-"+str(currentChunk)+".cnf")
            gen_write_f = time.process_time()
            print("generation and write time:", gen_write_f - gen_write_s)
            t_s = time.process_time()
            if time_wbo_max:
                solve_time_rem = time_wbo_max-time_elapsed_wbo 
            try:
                p = subprocess.Popen(["./open-wbo-inc_static", "-algorithm=8", "-iterations="+str(iterations), pname+"-"+str(currentChunk)+".cnf"], stdout=open(sname + "-chnk" + str(currentChunk) + ".txt", "w"))
                p.wait(timeout=solve_time_rem/(chunks-currentChunk))
            except subprocess.TimeoutExpired:
                print("exiting open-wbo because of solve time alloted...")
                p.terminate()
                time.sleep(10)
            t_f = time.process_time()
            time_elapsed_wbo += t_f - t_s
        assignments = filter(lambda x : x[2] == currentSize-1, mappingVars(readMaxHSFileSwap, physNum, logNum, currentSize, swapNum+addedSwaps[currentChunk], sname + "-chnk" + str(currentChunk) + ".txt"))
        if list(assignments): 
            print("chunk", currentChunk, "solved")
            currentChunk = currentChunk+1
        else:
                if len(negatedModels[currentChunk-1]) < 50*(addedSwaps[currentChunk]+1): 
                    print("got stuck on chunk", currentChunk, "backtracking to chunk", currentChunk-1)
                    prevAssignments = filter(lambda x : x[2] == prevSize-1, mappingVars(readMaxHSFileSwap, physNum, logNum, prevSize, swapNum+addedSwaps[currentChunk-1], sname + "-chnk" + str(currentChunk-1) + ".txt"))
                    negatedModel =  [(True, "x", phys, log, lastGate) for (phys, log, lastGate) in prevAssignments]
                    negatedModels[currentChunk-1].append(negatedModel)
                    currentChunk = currentChunk-1
                else:
                    print("got stuck on chunk", currentChunk, "repeatedly, increasing swap count")
                    addedSwaps[currentChunk] += 1 
    cost = 0
    for i in range(chunks):
        with open(sname + "-chnk" + str(i) + ".txt") as f:
            for line in f:
                if line.startswith("o"):
                    count = int(line.split()[1])
        cost += count
    return_results['cost'] = cost
    return_results['time_wbo'] = time_elapsed_wbo
    return return_results

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

def testScript():
    for i in range(len(ibmQx4Tests)):
        solveInstance(*ibmQx4Tests[i], "test/small" +
                      i + ".cnf", "test/SOL-small" + i + ".txt")
    for i in range(len(ibmqx5Tests)):
        solveSwaps(*ibmQx4Tests[i], "test/med" + i +
                   ".cnf", "test/SOL-med" + i + ".txt", 1)
    for i in range(len(ibmTokyoTests)):
        for j in range(1, 21, 5):
            solveSwapsFF(*ibmQx4Tests[i], "test/large" +
                         i + ".cnf", "test/SOL-large" + i + ".txt", 1, j)


def toQasm(physNum, logNum, numCnots, swapNum, fname, progPath, cm, prevMap, start=0):
    circ = qiskit.QuantumCircuit(16, 16)
    prog = qiskit.QuantumCircuit.from_qasm_file(progPath)
    edges = np.argwhere(cm > 0)
    i = start
    while circ.num_nonlocal_gates() < numCnots:
        circ.compose(*prog[i], inplace=True)
        i += 1
    lits = readMaxHSFileSwap(physNum, logNum, numCnots, swapNum, fname)

    swaps = [s[2] for s in filter(lambda x : not x[0] and x[1] == "s" and x[2][0] != x[2][1], lits)]
    mappingVars =  [x[2] for x in filter(lambda x : not x[0] and x[1] == "x", lits)]
    logToPhys = { (j,k) : i for (i,j,k) in mappingVars}
    physToLog = { (i,k) : j for (i,j,k) in mappingVars}
    
    swapIndices = [s[3] for s in swaps]
    # print([s[2] for s in filter(lambda x : not x[0] and x[1] == "s", lits)])
    

    for k in range(numCnots):
        mapKLog = list(filter(lambda x: x[0][1] == k, logToPhys.items()))
        assert(len(list(mapKLog)) == len(set(mapKLog))), "Invalid solution: non-injective"
        if k == 0 and prevMap: assert mapKLog == prevMap, "Invalid solution: slices aren't consistent"
        mapKPhys = list(filter(lambda x: x[0][1] == k, logToPhys.items()))
        assert(len(list(mapKPhys)) == len(set(mapKPhys))), "Invalid solution: non-function"
        swapsK = filter(lambda s: s[3] == k, swaps)
        interMap = dict(mapKLog)
        justLog = [s[:2] for s in swapsK]
        for (q1,q2) in justLog:
            phys1 =interMap[(q1,k)] 
            phys2 =interMap[(q2,k)] 
            assert([phys1, phys2] in edges.tolist()), "Invalid solution: bad swap"
            newLocations = composeSwaps([(q1,q2)], range(logNum))
            interMap = {(newLocations[j], k) : i for ((j,k), i) in interMap.items() }
        if k>0:
            for p in range(physNum):
                if (p,k) in physToLog.keys():
                    log1 = physToLog[(p,k-1)]
                    log2 = physToLog[(p,k)]
                    assert(log2 == composeSwaps(justLog, range(logNum))[log1]), "Invalid solution: unexpected SWAP"      
    mappedCirc = qiskit.QuantumCircuit(circ.num_qubits)
    prog2prog = {k : k for k in range(circ.num_qubits)}
    cnotCount = 0 
    for j in range(len(circ)):
        if circ[j][0].name == 'cx':
            if cnotCount in swapIndices:
                swapsK = filter(lambda s: s[3] == cnotCount, swaps)
                for s in swapsK:
                    mappedCirc.swap(prog2prog[s[0]], prog2prog[s[1]])
                    (prog2prog[s[0]], prog2prog[s[1]]) = (prog2prog[s[1]], prog2prog[s[0]])
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
          
def toQasmFF(logNum, progName, cm, swapNum, chunks, fname):
    pointer = 0
    physNum = len(cm)
    cnots = extractCNOTs(progName)
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


def transpile(logNum, progname, cm, swapNum, chunks, cnfname, sname):
    solveSwapsFF(logNum, progname, cm, swapNum, chunks, cnfname, sname)
    return toQasmFF(logNum, progname, cm, swapNum, chunks, sname)


if __name__ == "__main__":
    print(solveSwapsFF(10, "examples/mini.qasm", ibmtokyo, 1, 3, iterations=10,time_wbo_max=1800))
