import qiskit

def compose_swaps(swap_seq, phys_qubits):
    current = {phys : phys for phys in phys_qubits}
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
