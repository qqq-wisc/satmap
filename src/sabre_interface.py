
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap, SabreLayout, ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla
from qiskit.test.mock import FakeToronto


def get_sabre_initial_map_and_swap_count(file_name, coupling_map):
    circ = QuantumCircuit.from_qasm_file(file_name)
    cm = CouplingMap(np.argwhere(coupling_map > 0))
    sabre_layout = SabreLayout(cm, layout_trials=1, swap_trials=1)
    pm = PassManager([sabre_layout])
    mapped_circ = pm.run(circ)
    i_map =  ({circ.find_bit(q)[0] : p for (q,p) in pm.property_set['layout'].get_virtual_bits().items() if q.register.name == 'q'})
    rolling_map = {(p,k) : q for q,p in i_map.items() for k in range(circ.count_ops()['cx'])}
    if 'swap' not in mapped_circ.count_ops().keys():
        swap_count = 0
    else: 
        swap_count = mapped_circ.count_ops()['swap']
    swaps = []
    cnot_counter = 0
    for ins, qubits, clbits in mapped_circ:
        if ins.name=='cx':
            cnot_counter += 1
        elif ins.name=='swap':
            q1 = mapped_circ.find_bit(qubits[0])[0] 
            q2 = mapped_circ.find_bit(qubits[1])[0]
            log1 = rolling_map[q1, cnot_counter]  
            rolling_map[q1, cnot_counter]  = rolling_map[q2, cnot_counter]  
            rolling_map[q2, cnot_counter]  = log1
            swaps.append((q1, q2, cnot_counter))
    return rolling_map, swap_count


