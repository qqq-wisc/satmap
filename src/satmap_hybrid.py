import itertools
import numpy as np
import qiskit
import qiskit.circuit
import qiskit.dagcircuit
import scipy.sparse.csgraph
from pysat.solvers import Solver
from pysat.card import *
import sabre_interface
import architectures

# Constraints
def add_functional_consistency_constraint(cnot_count, log_num, phys_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    for k in range(cnot_count):
        for i in range(log_num):
            lits = [to_int(sem_vars, (1, "m", i, j, k)) for j in range(phys_num)]
            for clause in CardEnc.equals(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)

def add_injectivity_constraint(cnot_count, log_num, phys_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    for k in range(cnot_count):
        for j in range(phys_num):
            lits  = [to_int(sem_vars, (1, "m", i, j, k)) for i in range(log_num)]
            for clause in CardEnc.atmost(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)

def add_cnot_adjacency_constraint(cnots, coupling_map, s:Solver, sem_vars):
    nonzeroIndices = np.argwhere(coupling_map>0)
    for k in range(len(cnots)):
        edge_clause = []
        (c,t) = cnots[k]
        for edge in nonzeroIndices:
            [u,v] = edge
            edge_clause.append(to_int(sem_vars, (1, "e", u, v, k)))
            control_lits = [(-1, "e", u, v, k), (1, "m", c, u, k), (1, "m", c, v, k)]
            control_clause = [to_int(sem_vars, lit) for lit in control_lits]
            s.add_clause(control_clause)
            target_lits = [(-1, "e", u, v, k), (1, "m", t, u, k), (1, "m", t, v, k)]
            target_clause = [to_int(sem_vars, lit) for lit in target_lits]
            s.add_clause(target_clause)
        s.add_clause(edge_clause)

def add_swap_choice_constraint(cnot_count, coupling_map, swap_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    allowed_swaps = np.append(np.argwhere(coupling_map>0), [[0,0]], axis=0)
    for k in range(cnot_count):
        for t in range(swap_num):
            lits = [to_int(sem_vars, (1, "s", u, v, k, t)) for [u,v] in allowed_swaps]
            for clause in CardEnc.equals(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)
            
def add_swap_effect_constraint(cnot_count, coupling_map, log_num, phys_num, swap_num, s:Solver, sem_vars:IDPool):
    allowed_swaps = np.append(np.argwhere(coupling_map>0), [[0,0]], axis=0) 
    swap_seqs = itertools.product(allowed_swaps, repeat=swap_num)
    for swap_seq in swap_seqs:
        indexed_swaps = list(enumerate(swap_seq))
        perm = compose_swaps(swap_seq, phys_num)
        for k in range(1, cnot_count):
            swap_lits = [(-1, "s", u, v, k, t) for (t, [u,v]) in indexed_swaps]
            for i in range(log_num):
                for j in range(phys_num):
                    lits = swap_lits + [(-1, "m", i, j, k-1),(1, "m", i, perm[j], k)]
                    clause = [to_int(sem_vars, lit) for lit in lits]
                    s.add_clause(clause)
                
def add_optimization_constraint(cnot_count, coupling_map, swap_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool, upper_bound):
    real_swaps = np.argwhere(coupling_map>0)
    lits = [to_int(sem_vars, (1, "s", u, v, k, t)) for [u,v] in real_swaps for k in range(cnot_count) for t in range(swap_num)]
    for clause in CardEnc.atmost(lits=lits, bound=upper_bound, vpool=aux_vars).clauses:
        s.add_clause(clause)
 
# utility helper functions 
def to_int(vpool, lit):
    return lit[0]*vpool.id(lit[1:])

def compose_swaps(swap_seq, phys_num):
    current = {phys : phys for phys in range(phys_num)}
    for swap in swap_seq:
        apply_swap(swap, current)
    return current

def apply_swap(swap, current):
    [u, v] = swap
    for i in current.keys():
        if current[i] == u:
            current[i] = v
        elif current[i] == v:
            current[i] = u

def extract_qubits(gate_list):
    qubits = set()
    for gate in gate_list:
        for qubit in gate:
            qubits.add(qubit)
    return qubits

def extract2qubit(fname):
    gates = []
    circ = qiskit.QuantumCircuit.from_qasm_file(fname)
    for j in range(len(circ)):
        qubits = circ[j][1]
        
        if len(qubits) == 2:
            gates.append([circ.find_bit(q)[0] for q in qubits])
        elif len(qubits) > 2:
            print('Warning: ignoring gate with more than 2 qubits')
    return gates

def unpack_model(model, sem_vars):
    return [sem_vars.obj(v) for v in model if sem_vars.obj(v)]

def swap_count(model, sem_vars):
    semantic = unpack_model(model, sem_vars)
    swaps = [var for var in semantic if var[0] == 's' and var[1] != var[2]]
    return len(swaps)

def get_mapping(unpacked_model):
    return {(var[2], var[3]): var[1] for var in unpacked_model if var[0] == 'm'}

def check_model(cnots, coupling_map, unpacked_model):
    cnot_count = len(cnots)
    phys_num = len(coupling_map)
    mapping_vars = [var[1:] for var in unpacked_model if var[0] == 'm']
    swaps = [var[1:] for var in unpacked_model if var[0] == 's']
    log_to_phys = { (i,k) : j for (i,j,k) in mapping_vars}
    for k in range(cnot_count):
        mapping_at_k = {i : j for ((i,kp),j) in log_to_phys.items() if kp == k}
        assert(len(set(mapping_at_k.values())) == len(list(mapping_at_k.values()))), "Invalid solution: non-injective"
        assert(len(set(mapping_at_k.keys())) == len(list(mapping_at_k.keys()))), "Invalid solution: non-function"
        (c,t) = cnots[k]
        assert([mapping_at_k[c], mapping_at_k[t]] in np.argwhere(coupling_map>0).tolist()), f"Invalid solution: unsatisfied cnot between {c} and {t} mapped to {mapping_at_k[c]} and {mapping_at_k[t]}"
        if k>0:
            for l in mapping_at_k.keys():
                swaps_k = filter(lambda s: s[3] == k, swaps)
                prev_phys = log_to_phys[(l, k-1)]
                assert(log_to_phys[(l,k)] == compose_swaps(swaps_k, phys_num)[prev_phys]), 'Invalid solution: mapping not consistent with swaps'
        edges = [var for var in unpacked_model if var[0] == 'e' and var[1] != var[2]]

        

# solving 
def solve(cnots, coupling_map, swap_num, upper_bound, mapping):
    cnot_count = len(cnots)
    phys_num = len(coupling_map)
    log_num = max(extract_qubits(cnots)) + 1
    num_m = phys_num * log_num * cnot_count
    num_e = phys_num * phys_num * cnot_count
    num_s = phys_num * phys_num * cnot_count * swap_num
    top = num_m + num_s + num_e 
    sem_vars = IDPool()
    aux_vars = IDPool(start_from=top+1)
    with Solver('cd15') as s:
        add_functional_consistency_constraint(cnot_count=cnot_count, log_num=log_num, phys_num=phys_num, s=s, sem_vars=sem_vars, aux_vars=aux_vars)
        add_injectivity_constraint(cnot_count=cnot_count, log_num=log_num, phys_num=phys_num, s=s, sem_vars=sem_vars,aux_vars=aux_vars)
        add_cnot_adjacency_constraint(cnots=cnots, coupling_map=coupling_map, s=s, sem_vars=sem_vars)
        add_swap_choice_constraint(cnot_count=cnot_count, coupling_map=coupling_map, swap_num=swap_num, s=s, sem_vars=sem_vars, aux_vars=aux_vars)
        add_swap_effect_constraint(cnot_count=cnot_count, coupling_map=coupling_map, log_num=log_num, phys_num=phys_num, swap_num=swap_num, s=s, sem_vars=sem_vars)
        add_optimization_constraint(cnot_count=cnot_count,coupling_map=coupling_map,swap_num=swap_num, s=s, sem_vars=sem_vars, aux_vars=aux_vars, upper_bound=upper_bound)
        log_mapping = {q : (p,k) for ((p,k),q) in mapping.items() if q < log_num}
        submaps = itertools.chain.from_iterable(itertools.combinations(range(log_num), r) for r in range(log_num+1))
        for submap in submaps:
            print(submap)
            fixed_qubits = log_mapping.keys()
            for entry in submap:
                fixed_qubits.remove(entry)
            assump = [to_int(sem_vars, (1, 'm', l, p, k)) for l in fixed_qubits for p,k in log_mapping[l]]
            if s.solve(assumptions = assump): 
                m = s.get_model()
                unpacked = unpack_model(m, sem_vars)
                check_model(cnots, coupling_map, unpacked)
                print(f"found solution with {swap_count(m, sem_vars)}")
                return ('sat', unpack_model(m, sem_vars), swap_count(m, sem_vars))        
        return ("unsat", None, None)


def solve_with_sabre(file_name, coupling_map, swap_num, use_sabre_sol=True):
   mapping, sabre_swaps =  sabre_interface.get_sabre_initial_map_and_swap_count(file_name=file_name, coupling_map=coupling_map)
   if not use_sabre_sol: mapping = {}
   print(f"sabre used {sabre_swaps} swaps")
   cnots = extract2qubit(file_name)
   upper_bound = sabre_swaps-1
   while upper_bound >= 0:
        print(f"trying to find a solution with {upper_bound} swaps")
        result, sol, cost = solve(cnots, coupling_map=coupling_map, swap_num=swap_num, mapping=mapping, upper_bound=upper_bound)
        if result == 'unsat':
            return upper_bound+1
        else:
            upper_bound = cost-1
            mapping = get_mapping(sol)





solve_with_sabre(file_name="../examples/4gt10-v1_81.qasm", coupling_map=architectures.ibmToronto, swap_num=1, use_sabre_sol=False)
