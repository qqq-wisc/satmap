import itertools
import os
import numpy as np
import qiskit
import qiskit.circuit
from pysat.solvers import Solver
from pysat.card import *
import signal
from common import compose_swaps, extract_qubits
import sabre_interface

# Constraints
def add_functional_consistency_constraint(cnot_count:int, log_num:int, phys_qubits:'set[int]', s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    for k in range(cnot_count):
        for i in range(log_num):
            lits = [to_int(sem_vars, (1, "m", i, j, k)) for j in phys_qubits]
            for clause in CardEnc.equals(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)

def add_injectivity_constraint(cnot_count:int, log_num:int, phys_qubits:'set[int]', s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    for k in range(cnot_count):
        for j in phys_qubits:
            lits  = [to_int(sem_vars, (1, "m", i, j, k)) for i in range(log_num)]
            for clause in CardEnc.atmost(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)

def add_cnot_adjacency_constraint(cnots:'tuple[int, int]', coupling_map, phys_qubits, s:Solver, sem_vars):
    arr = np.array(list(phys_qubits))
    restricted_coupling_map = np.zeros_like(coupling_map)
    restricted_coupling_map[np.ix_(arr, arr)] = coupling_map[np.ix_(arr, arr)]
    nonzeroIndices = np.argwhere(restricted_coupling_map>0)
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

def add_swap_choice_constraint(cnot_count, phys_qubits, coupling_map, swap_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool):
    arr = np.array(list(phys_qubits))
    restricted_coupling_map = np.zeros_like(coupling_map)
    restricted_coupling_map[np.ix_(arr, arr)] = coupling_map[np.ix_(arr, arr)]
    allowed_swaps = np.append(np.argwhere(restricted_coupling_map>0), [[0,0]], axis=0)
    for k in range(cnot_count):
        for t in range(swap_num):
            lits = [to_int(sem_vars, (1, "s", u, v, k, t)) for [u,v] in allowed_swaps]
            for clause in CardEnc.equals(lits=lits, bound=1, vpool=aux_vars).clauses:
                s.add_clause(clause)
            
def add_swap_effect_constraint(cnot_count, coupling_map, log_num, phys_qubits, swap_num, s:Solver, sem_vars:IDPool):
    arr = np.array(list(phys_qubits))
    restricted_coupling_map = np.zeros_like(coupling_map)
    restricted_coupling_map[np.ix_(arr, arr)] = coupling_map[np.ix_(arr, arr)]
    allowed_swaps = np.append(np.argwhere(restricted_coupling_map>0), [[0,0]], axis=0) 
    swap_seqs = itertools.product(allowed_swaps, repeat=swap_num)
    for swap_seq in swap_seqs:
        indexed_swaps = list(enumerate(swap_seq))
        perm = compose_swaps(swap_seq, phys_qubits)
        for k in range(1, cnot_count):
            swap_lits = [(-1, "s", u, v, k, t) for (t, [u,v]) in indexed_swaps]
            for i in range(log_num):
                for j in phys_qubits:
                    lits = swap_lits + [(-1, "m", i, j, k-1),(1, "m", i, perm[j], k)]
                    clause = [to_int(sem_vars, lit) for lit in lits]
                    s.add_clause(clause)
                
def add_optimization_constraint(cnot_count, coupling_map, phys_qubits, swap_num, s:Solver, sem_vars:IDPool, aux_vars:IDPool, upper_bound):
    arr = np.array(list(phys_qubits))
    restricted_coupling_map = np.zeros_like(coupling_map)
    restricted_coupling_map[np.ix_(arr, arr)] = coupling_map[np.ix_(arr, arr)]
    real_swaps = np.argwhere(restricted_coupling_map>0)
    lits = [to_int(sem_vars, (1, "s", u, v, k, t)) for [u,v] in real_swaps for k in range(cnot_count) for t in range(swap_num)]
    for clause in CardEnc.atmost(lits=lits, bound=upper_bound, vpool=aux_vars).clauses:
        s.add_clause(clause)
 
# utility helper functions 
def to_int(vpool, lit):
    return lit[0]*vpool.id(lit[1:])

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
    phys_qubits = range(len(coupling_map))
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
                swaps_k = [[s[0], s[1]] for s in filter(lambda s: s[2] == k, swaps)]
                prev_phys = log_to_phys[(l, k-1)]
                assert(log_to_phys[(l,k)] == compose_swaps(swaps_k, phys_qubits)[prev_phys]), 'Invalid solution: mapping not consistent with swaps'
        edges = [var for var in unpacked_model if var[0] == 'e' and var[1] != var[2]]

def get_circ_from_model(input_circ, phys_num, unpacked_model):
    mapping_vars = [var[1:] for var in unpacked_model if var[0] == 'm']
    nontrivial_swaps = [var[1:] for var in unpacked_model if var[0] == 's' and var[1] != var[2]]
    final_circ = qiskit.QuantumCircuit(phys_num, phys_num)
    temp =  qiskit.QuantumCircuit(phys_num, phys_num)
    temp.compose(input_circ)
    log_to_phys = { (i,k) : j for (i,j,k) in mapping_vars}
    cnot_counter = 0
    for ins, qubits, clbits in input_circ:
        if ins.name == 'cx': 
            swaps_at_current = [s for s in nontrivial_swaps if s[2] == cnot_counter]
            for (u, v,  _, _) in swaps_at_current:
                final_circ.swap(u, v)
            ctrl = input_circ.find_bit(qubits[0])[0] 
            tar = input_circ.find_bit(qubits[1])[0] 
            final_circ.cx(log_to_phys[(ctrl, cnot_counter)], log_to_phys[(tar, cnot_counter)])
            cnot_counter +=1 
    return final_circ

# solving

def vertical_iterator(log_num, log_mapping):
    submaps = itertools.chain.from_iterable(itertools.combinations(range(log_num), r) for r in range(log_num+1))
    return ([(1, 'm', l, log_mapping[(l,k)], k) for l,k in log_mapping.keys() if l not in submap] for submap in submaps)

def empty_iterator(): return ([],) 

def horizontal_sliding_iterator(cnot_count, log_mapping):
    windows = (((i, i+window_size) for window_size in range(0,cnot_count+1,min(10, cnot_count)) for i in range(cnot_count - window_size + 1)) )
    return ([ (1, 'm', l, log_mapping[(l,k)], k) for l,k in log_mapping.keys() if k not in range(lb, ub)] for (lb, ub) in windows)

def horizontal_iterator(cnot_count, log_mapping, sabre_swaps):
    return ([(1, 'm', l, log_mapping[(l,k)], k) for l,k in log_mapping.keys() if k <= i] + \
                         [(1, 's', u, v, k,t) for u,v,k,t in sabre_swaps if k <= i]  for i in range(cnot_count, -1, -1))

def solve(cnots, coupling_map, swap_num, upper_bound, mapping, sabre_swaps, explore):
    cnot_count = len(cnots)
    phys_num = len(coupling_map)
    log_num = max(extract_qubits(cnots)) + 1
    num_m = phys_num * log_num * cnot_count
    num_e = phys_num * phys_num * cnot_count
    num_s = phys_num * phys_num * cnot_count * swap_num
    top = num_m + num_s + num_e 
    sem_vars = IDPool()
    aux_vars = IDPool(start_from=top+1)
    log_mapping = {(q,k) : p for ((p,k),q) in mapping.items() if -1 < q < log_num}
    if mapping == {}:
        phys_qubits = range(len(coupling_map))
    else:
        #phys_qubits = range(len(coupling_map)
        phys_qubits = set(log_mapping.values())
    with Solver('cd15', use_timer=True) as s:
        add_functional_consistency_constraint(cnot_count=cnot_count, log_num=log_num, phys_qubits=phys_qubits, s=s, sem_vars=sem_vars, aux_vars=aux_vars)
        add_injectivity_constraint(cnot_count=cnot_count, log_num=log_num, phys_qubits=phys_qubits, s=s, sem_vars=sem_vars,aux_vars=aux_vars)
        add_cnot_adjacency_constraint(cnots=cnots, phys_qubits=phys_qubits, coupling_map=coupling_map, s=s, sem_vars=sem_vars)
        add_swap_choice_constraint(cnot_count=cnot_count, phys_qubits=phys_qubits,coupling_map=coupling_map, swap_num=swap_num, s=s, sem_vars=sem_vars, aux_vars=aux_vars)
        add_swap_effect_constraint(cnot_count=cnot_count, coupling_map=coupling_map, log_num=log_num, phys_qubits=phys_qubits, swap_num=swap_num, s=s, sem_vars=sem_vars)
        add_optimization_constraint(cnot_count=cnot_count,phys_qubits=phys_qubits, coupling_map=coupling_map,swap_num=swap_num, s=s, sem_vars=sem_vars, aux_vars=aux_vars, upper_bound=upper_bound)
        if explore == "free": assumps = empty_iterator()
        elif explore == 'vertically': assumps = vertical_iterator(log_num, log_mapping)
        elif explore == 'horizontally': assumps = horizontal_iterator(cnot_count, log_mapping, sabre_swaps)
        elif explore == 'horizontal_sliding_window': assumps = horizontal_sliding_iterator(cnot_count, log_mapping)        
        for assump in assumps:
            lits = map(lambda x : to_int(sem_vars, x), assump)
            if s.solve(assumptions = lits): 
                    m = s.get_model()
                    unpacked = unpack_model(m, sem_vars)
                    check_model(cnots, coupling_map, unpacked)
                    print(f"found solution with {swap_count(m, sem_vars)}")
                    return ('sat', unpack_model(m, sem_vars), swap_count(m, sem_vars), s.time_accum())  
        return ("unsat", None, None, s.time_accum())      

class TimeoutError(Exception):
    pass

def _sig_alarm(sig, tb):
    raise TimeoutError("timeout")

def solve_with_sabre(file_name, coupling_map, swap_num, explore='vertically', output_file='data.txt', use_sabre_swap_num=False, timeout=1800):
   mapping, sabre_swaps, cnots, sabre_swap_num =  sabre_interface.get_sabre_initial_map_and_swap_count(file_name=file_name, coupling_map=coupling_map)
   if explore=='free': mapping = {}
   if use_sabre_swap_num:
       swap_num = sabre_swap_num  
   print(f"sabre used {len(sabre_swaps)} swaps")
   print(f"sabre used at most {sabre_swap_num} swaps between cnots")
   upper_bound = len(sabre_swaps)
   total_solve_time = 0
   min_found = False
   signal.signal(signal.SIGALRM, _sig_alarm)
   signal.signal(signal.SIGINT, _sig_alarm)
   signal.signal(signal.SIGTERM, _sig_alarm)
   best_so_far = sabre_interface.run_sabre(file_name, coupling_map)
   try:
    signal.alarm(timeout)
    while upper_bound >= 0 and not min_found:
            print(f"trying to find a solution with {upper_bound} swaps")
            result, sol, cost, solve_time = solve(cnots, coupling_map=coupling_map, swap_num=swap_num, mapping=mapping, sabre_swaps=sabre_swaps, upper_bound=upper_bound, explore=explore) 
            total_solve_time += solve_time
            if result == 'unsat':
                with open(output_file, 'a') as f:
                    output = ({'circ' : os.path.basename(file_name), 'qubits' : len(extract_qubits(cnots)), 'cnots' : len(cnots), 'algorithm' : explore, "swaps" : cost, "solve_time" : total_solve_time, 'search_terminated' : True })
                    f.write(str(output) + "\n")
                    min_found = True
            else:
                upper_bound = cost-1
                mapping = get_mapping(sol)
                with open(output_file, 'a') as f:
                    output = {'circ' : os.path.basename(file_name), 'qubits' : len(extract_qubits(cnots)), 'cnots' : len(cnots), 'algorithm' : explore, "swaps" : cost, "solve_time" : total_solve_time, 'search_terminated' : False}
                    mapped_circ = (get_circ_from_model(qiskit.QuantumCircuit.from_qasm_file(file_name), len(coupling_map), sol))
                    best_so_far = output, mapped_circ.qasm()
                    f.write(str(output) + "\n")
    
    best_so_far[0]['search_terminated'] = True
    return best_so_far
   except:
       print('timed out/interrupted... returning best so far')
       return best_so_far
   