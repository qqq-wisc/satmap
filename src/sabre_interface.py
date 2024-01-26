
import os
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap, SabreLayout, ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla
from qiskit.test.mock import FakeToronto


def get_sabre_initial_map_and_swap_count(file_name, coupling_map):
    circ = QuantumCircuit.from_qasm_file(file_name)
    cm = CouplingMap(np.argwhere(coupling_map > 0))
    sabre_layout = SabreLayout(cm, seed=42)
    pm = PassManager([sabre_layout])
    mapped_circ = pm.run(circ)
    i_map =  ({circ.find_bit(q)[0] : p for (q,p) in pm.property_set['layout'].get_virtual_bits().items() if q.register.name == 'q'})
    #rolling_map  ={(p,0) : q for q,p in i_map.items()}
    rolling_map = {(p, 0) : -1 for p in range(len(coupling_map))}
    rolling_map.update({(p,0) : q for q,p in i_map.items()} )
    if 'swap' not in mapped_circ.count_ops().keys():
        swap_count = 0
    else: 
        swap_count = mapped_circ.count_ops()['swap']
    swaps = []
    cnot_counter = 0
    cnots = []
    count_dict = {0 : 0}
    for ins, qubits, clbits in mapped_circ:
        if ins.name=='cx':
            q1 = mapped_circ.find_bit(qubits[0])[0] 
            q2 = mapped_circ.find_bit(qubits[1])[0]
            cnots.append([rolling_map[q1, cnot_counter],  rolling_map[q2, cnot_counter]])
            cnot_counter += 1
            count_dict[cnot_counter] = 0
            rolling_map.update({ (p,cnot_counter) : q for (p, k),q in rolling_map.items() if k == cnot_counter-1 })
        elif ins.name=='swap':
            q1 = mapped_circ.find_bit(qubits[0])[0] 
            q2 = mapped_circ.find_bit(qubits[1])[0]
            log1 = rolling_map[q1, cnot_counter]  
            rolling_map[q1, cnot_counter]  = rolling_map[q2, cnot_counter]  
            rolling_map[q2, cnot_counter]  = log1
            
            swaps.append((q1, q2, cnot_counter, count_dict[cnot_counter]))
            count_dict[cnot_counter]  += 1
    max_per = max(count_dict.values())
    return rolling_map, swaps, cnots, max_per


def run_sabre(file_name, coupling_map):
    circ = QuantumCircuit.from_qasm_file(file_name)
    cm = CouplingMap(np.argwhere(coupling_map > 0))
    sabre_layout = SabreLayout(cm,seed=42)
    pm = PassManager([sabre_layout])
    t_s = time.time()
    mapped_circ = pm.run(circ)
    t_f = time.time()
    if 'swap' not in mapped_circ.count_ops().keys():
        swap_count = 0
    else: 
        swap_count = mapped_circ.count_ops()['swap']
    return ({'circ' : os.path.basename(file_name), 'cnots' : circ.num_nonlocal_gates(), 'swaps' : swap_count, "solve_time" : t_f - t_s, 'algorithm ': 'sabre'}, mapped_circ.qasm())
