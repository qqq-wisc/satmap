import os
import resource
from pytket.qasm import circuit_from_qasm, circuit_from_qasm_str, circuit_to_qasm, circuit_to_qasm_str
from pytket.passes import PlacementPass, RoutingPass
# Below is used for pytket 0.19.2 (CURRENT)
#from pytket.routing import GraphPlacement, Architecture
# Below is used for pytket >= 1.0.0 (BREAKS)
from pytket.architecture import Architecture
from pytket.placement import GraphPlacement
from pytket.transform import Transform
from pytket.circuit.display import render_circuit_jupyter
from pytket.transform import Transform
import mqt.qmap
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap, SabreLayout, ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla
from qiskit.test.mock import FakeTokyo
from satmap import computeFidelity, solve, extractCNOTs,  extractQbits, transpile, toQasmFF, getLayers, sortCnots
import numpy as np
import json
from time import time
import math
import re
import itertools
import architectures
import sys
from os import listdir, mkdir
from os.path import basename
import subprocess
from olsq import OLSQ
from olsq.device import qcdevice
from architectures import generateEnfFile, tokyo_error_list, tokyo_error_map
import argparse
import signal

is_mapping = False

def count_atomic_gates(gate_counts):
    # This function maps all gates to the number of atomic gates they represent
    atomic_gates = {'U': 1, 'X': 1, 'H': 1, 'T': 1, 'TDG': 1, 'CX': 1}
    composite_gates = {'SWAP': 3, 'BRIDGE': 4}
    gate_num_total = 0

    print(gate_counts)

    for gate_name, gate_num in gate_counts.items():
        gate_name = gate_name.upper()

        if(gate_name in atomic_gates):
            gate_num_total += gate_num*atomic_gates[gate_name]
        elif(gate_name in composite_gates):
            gate_num_total += gate_num*composite_gates[gate_name]
        elif(gate_name[0] == 'U'):  # for custom unary gates (Enfield output)
            gate_num_total += gate_num*atomic_gates['U']
        elif(gate_name == "MEASURE"):
            pass
        else:
            print(
                f'Error, gate {gate_name} not recognized by count_atomic_gates')
            gate_num_total += gate_num*1
    return gate_num_total


def tket_count_gates(commands):

    gates = {}
    for gate in commands:
        op = str(gate.op)
        if op in gates:
            gates[op] += 1
        else:
            gates[op] = 1

    return gates


def mqt_count_gates(qasm_text):

    match = re.findall(
        r'(?:(cx|swap) q\[\d+\], q\[\d+\])|(?:(x|h|t|tdg) q\[\d+\]);', qasm_text)
    gates = {}
    for op in match:
        op = op[0] if op[0] != '' else op[1]
        if op in gates:
            gates[op] += 1
        else:
            gates[op] = 1

    return gates


def obtain_circuit_metrics(qasm_circuit_str):
    # qasm_circuit <-- string qasm circuit
    # This function obtains desired metrics from a qasm circuit
    # and returns them in the form of a Dictionary
    # It should help reduce the clutter in run_all_methods()

    circ = circuit_from_qasm_str(qasm_circuit_str)
    circuit_metrics = {}

    circuit_metrics['depth'] = circ.depth()
    circuit_metrics['g_tot'] = count_atomic_gates(
        tket_count_gates(circ.get_commands()))
    # circuit_metrics['q_tot'] =

    return circuit_metrics


def obtain_circuit_metrics_from_file(qasm_file_path):
    program = QuantumCircuit.from_qasm_file(qasm_file_path).qasm()
    return obtain_circuit_metrics(program)


# Should have functions return metrics in a Dict (or OrderedDict)
# That way we can return a bunch of metrics if we want to and use only the ones we're interested in examining


def run_tket(file_path, arch, timeout=None):
    # tket
    circ = circuit_from_qasm(file_path)
    gates_init = circ.get_commands()

    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    PlacementPass(GraphPlacement(Architecture(
        np.argwhere(arch > 0)))).apply(circ)
    RoutingPass(Architecture(np.argwhere(arch > 0))).apply(circ)
    is_mapping = False
    time_final = time()

    # print('Displaying tket commands (pre-BRIDGE-decomposition')
    # for command in circ.get_commands():
    #     print(command)
    # print('Ended displaying tket commands')

    # tket makes BRIDGES, which are somehow not supported by pytket
    # thus, we transform them into CNOTs
    Transform.DecomposeBRIDGE().apply(circ)

    results = {}
    results['time'] = time_final - time_init
    results['output_qasm'] = circuit_to_qasm_str(circ)
    results['gate_counts'] = tket_count_gates(circ.get_commands())
    results['gates_added'] = -1

    return results


def run_mqt(file_path, arch, timeout=None, method_mqt = None):
    # mqt
    architectures.generateMQTFile(arch, "jku_arch_file.txt")
    results = {}

    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    if method_mqt:
        output = mqt.qmap.compile(file_path, "jku_arch_file.txt", method=method_mqt)
        results['method'] = method_mqt
    else:
        output = mqt.qmap.compile(file_path, "jku_arch_file.txt")
        results['method'] = "default"
    is_mapping = False
    time_final = time()

    results_dict = json.loads(str(output))
    statistics = results_dict['statistics']
    mqt_qasm_text = results_dict['mapped_circuit']['qasm']

    results['time'] = time_final - time_init
    results['output_qasm'] = mqt_qasm_text
    results['gate_counts'] = mqt_count_gates(mqt_qasm_text)
    results['gates_added'] = statistics['additional_gates']
    results['time_reported'] = statistics['mapping_time']
    return results


def run_sabre(file_path, arch, timeout=None):
    # sabre
    cm = CouplingMap(np.argwhere(arch > 0))
    sabre_swap = SabreSwap(cm)
    sabre_layout = SabreLayout(cm, routing_pass=sabre_swap)
    pm = PassManager([sabre_layout, FullAncillaAllocation(cm),
                      EnlargeWithAncilla(), ApplyLayout(), sabre_swap])
    input_circ = QuantumCircuit.from_qasm_file(file_path)
    
    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    mapped_circ = pm.run(input_circ)
    is_mapping = False
    time_final = time()

    output_qasm = mapped_circ.qasm()

    results = {}
    results['time'] = time_final - time_init
    results['output_qasm'] = output_qasm
    results['gate_counts'] = mapped_circ.count_ops()
    results['gates_added'] = -1
    return results


def run_enfield(file_path, arch, timeout=None):
    # Enfield

    # obtain architecture in Enfield's special format
    arch_file_path = 'arch_enf.json'
    generateEnfFile(arch, arch_file_path)

    # compile circuit by calling subprocess on ./efd
    intermediate_file_path = f'file_path.enfield.temp'
    
    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    p = subprocess.run(["./efd", "-i", file_path, "-alloc", "Q_wpm",
                       "-arch-file", arch_file_path, "-o", intermediate_file_path, "-stats", "--inline"],
                       # stdout=subprocess.DEVNULL
                       )
    is_mapping = False
    time_final = time()

    # read output file back
    qc = QuantumCircuit.from_qasm_file(intermediate_file_path)
    # Using --inline decomposed all gates to atomic gates
    # this can also be achieved through:
    # new_qc = new_qc.decompose(gates_to_decompose='intrinsic_swap__')
    # print(f'depth: {new_qc.depth()}, g_tot:{new_qc.size()}, ops:{new_qc.count_ops()}')

    results = {}
    results['time'] = time_final - time_init
    results['output_qasm'] = qc.qasm()
    results['gate_counts'] = qc.count_ops()
    results['gates_added'] = -1
    return results

def run_olsq(file_path, arch, objectiveFunction = "swap", calibrationData=None, timeout=None):
    results = {}
    edgeList= [[int(u), int(v)] for[u,v] in np.argwhere(arch>0)]
    lsqc_solver = OLSQ(objectiveFunction, "transition")
    circuit_file = open(file_path, "r").read()
    success_rate = None
    if calibrationData:
        success_rate = [1-p for p in calibrationData] 
    lsqc_solver.setdevice(qcdevice(name="dev", nqubits=len(arch), connection=edgeList, ftwo=success_rate, swap_duration=3))
    lsqc_solver.setprogram(circuit_file)
    
    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    result_circuit, final_mapping, objective_value  = lsqc_solver.solve()
    is_mapping = False
    time_final = time()

    results['time'] = time_final - time_init
    results['output_qasm'] = result_circuit
    # with open('benchmarking_outputs/output_olsq.qasm', "w") as f:
    #     f.write(result_circuit)
    results['gate_counts'] = None
    results['objective value'] = math.exp(objective_value/1000)
    return results

def run_triq(file_path, arch, timeout=None):
    results = {}
    fname = os.path.basename(file_path)
    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    print(fname.split(".")[0]+".in")
    print(file_path)
    subprocess.run(["python3",  "../lib/TriQ/ir2dag.py", file_path, "triq_int/"+fname.split(".")[0]+".in" ])
    subprocess.run(["./triq", "triq_int/"+fname.split(".")[0]+".in", "triq_out/triq-"+fname, "ibmtokyo", "0" ])
    is_mapping = False
    time_final = time()
    results['time'] = time_final - time_init
    results['output_qasm'] = open("triq_out/triq-"+fname, "r").read()
    return results

def run_solveSwapsFF_old(file_path, arch, timeout=None):
    qbits_num_logical = extractQbits(file_path)
    swaps_num_max = 1
    cnots = extractCNOTs(file_path)
    chunks_num = 1
    
    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    output = solve(qbits_num_logical, file_path, arch, swaps_num_max, chunks_num)
    is_mapping = False
    time_final = time()

    output_qasm = toQasmFF(qbits_num_logical, os.path.join(os.path.split(file_path)[0], "qiskit-"+os.path.split(file_path)[1]),  arch, swaps_num_max, chunks_num, 'out')

    results = {}
    results['time'] = time_final - time_init
    results['output_qasm'] = output_qasm
    results['time_wbo'] = output['time_wbo']
    results['gate_counts'] = None
    results['gates_added'] = output['cost'] * 3
    return results

def run_solveSwapsFF(file_path, arch, timeout=None, routing=True, weighted=False, calibrationData = None, time_open_wbo_max=1800, layersPerChunk=20):
    qbits_num_logical = extractQbits(file_path)
    swaps_num_max = 1
    cnots = extractCNOTs(file_path)
    sorted_cnots = sortCnots(qbits_num_logical, cnots)
    layers = getLayers(sorted_cnots)

    if len(layers) < layersPerChunk:
        chunks_num = 1
    else:
        chunks_num = len(layers)//layersPerChunk

    time_init = time()
    global is_mapping 
    is_mapping = True
    if timeout:
        signal.alarm(args.timeout) 
    output = solve(file_path,
                          arch, swaps_num_max, chunks_num, pname="aux_files/" + basename(file_path)+str(layersPerChunk), sname="aux_files/"+basename(file_path)+str(layersPerChunk), time_wbo_max=time_open_wbo_max)
    is_mapping = False
    time_final = time()
    swaps = None
    if "swaps" in output.keys():
        swaps = output["swaps"]
    output_qasm = toQasmFF( os.path.join("aux_files/"+ "qiskit-"+os.path.split(file_path)[1]),  arch, swaps_num_max, chunks_num,"aux_files/"+basename(file_path)+str(layersPerChunk), swaps=swaps)

    results = {}
    results['time'] = time_final - time_init
    results['output_qasm'] = output_qasm
    results['time_wbo'] = output['time_wbo']
    results['gate_counts'] = None
    results['gates_added'] = output['cost'] * 3
    results['k'] = layersPerChunk
    if "a_star_time" in output.keys():
        results["a_star_time"] = output["a_star_time"]
    return results

def handle_timeout(signum, frame):
    # if True:
    if is_mapping:
        print("Mapper ran over the maximum allowed time")
        # print(f"Value of is_mapping: {is_mapping}")
        raise Exception("Error, mapper exceeded maximum runtime")

def run_mapper(input_program_path, mapper, arch, args=None):
    # executes a single run of mapping a program to a qc architecture

    # input_program_path - path to input program file
    # mapper - name of mapper used to map input program
    # arch - qc architecture
    # args - optional command-line args that may contain mapper-specific args
    
    # invididual methods should take inputs and return outputs and not have
    # other significant side-effects

    # make directory for results
    program_name = input_program_path.split(sep="/")[-1]
    output_path = args.output_dir if args and args.output_dir else f'results/results.{program_name}'
    try:
        mkdir(output_path)
    except:
        pass #dir already exists
    rid = f'RID-{args.run_id}.' if args and args.run_id else ""

    # Obtain specs of input program to compare with results
    print(input_program_path)
    circ_init = circuit_from_qasm(input_program_path)
    qbits_init_num = extractQbits(input_program_path)
    gates_init_num = count_atomic_gates(
        tket_count_gates(circ_init.get_commands()))

    
    mappers = {
        'tket': run_tket, 
        'mqt': run_mqt, 
        'sabre': run_sabre, 
        'olsq': run_olsq,
        'enfield': run_enfield,
        'solveSwapsFF': run_solveSwapsFF, 
        'solveSwapsFF_old': run_solveSwapsFF_old,
        'triq' : run_triq, 
        }

    archs =  {
        "tokyo" : architectures.ibmTokyo,
        "toronto" : architectures.ibmToronto,
        "4x4_mesh" : architectures.meshArch(4,4),
        'small_linear' : architectures.linearArch(6),
        "16_linear" : architectures.linearArch(16),
        "tokyo_full_diags" : architectures.tokyo_all_diags(),
        "tokyo_no_diags" : architectures.tokyo_no_diags(),
        'tokyo_drop_2' : architectures.tokyo_drop_worst_n(2, architectures.tokyo_error_map()),
        'tokyo_drop_6' : architectures.tokyo_drop_worst_n(6, architectures.tokyo_error_map()),
        'tokyo_drop_10' : architectures.tokyo_drop_worst_n(10, architectures.tokyo_error_map()),
        'tokyo_drop_14' : architectures.tokyo_drop_worst_n(14, architectures.tokyo_error_map()),
    }
    error_rates = {
        'fake_tokyo' : architectures.tokyo_error_list(),
        'fake_linear' : architectures.fake_linear_error_list()
    }
    #try:        
    if args:
            if mapper == 'solveSwapsFF':
                results = mappers[mapper](input_program_path, archs[arch], timeout=args.timeout, routing= not args.no_route, weighted=args.weighted, calibrationData=error_rates[args.err] if args.err else None, time_open_wbo_max = args.towbo if args.towbo else 1800, layersPerChunk=args.k)
            elif mapper == 'mqt' and args.mqt_method:
                results = mappers[mapper](input_program_path, archs[arch], timeout=args.timeout, method_mqt=args.mqt_method)
            elif mapper == 'olsq' and args.olsq_obj:
                results = mappers[mapper](input_program_path, archs[arch], timeout=args.timeout, objectiveFunction=args.olsq_obj, calibrationData=error_rates[args.err])
            else:
                results = mappers[mapper](input_program_path, archs[arch], timeout=args.timeout)
    else:
            results = mappers[mapper](input_program_path, archs[arch])

    # except Exception as e:
        
    #     print(e)
    #     # An exception was raised when time was done and mapper was still running
    #     # Write a .txt informing the user of a timeout and return
    #     with open(f'{output_path}/output.{mapper}.{arch}.{rid}txt', "w+") as f:
    #         timeout_metrics = {'timeout': args.timeout}
    #         f.write(str(timeout_metrics))
    #     return

    is_mapping = False

    # Once we've run our mapper, we store results
    circuit_metrics = {'time': results['time']}

    # Flush output qasm circuit to disk
    if 'output_qasm' in results:
        output_qasm = results['output_qasm']
        with open(f'{output_path}/output.{mapper}.{arch}.{rid}qasm', "w+") as f:
            f.write(output_qasm)

        circuit_metrics = obtain_circuit_metrics(output_qasm)
        circuit_metrics['time'] = results['time']
        circuit_metrics['g_add'] = circuit_metrics['g_tot'] - \
            gates_init_num

        if mapper == 'olsq' and args.olsq_obj == 'fidelity':
            circuit_metrics["reported fidelity"] = results['objective value']
        if arch == "tokyo":
            circuit_metrics["computed fidelity"] = computeFidelity(QuantumCircuit.from_qasm_str(output_qasm), tokyo_error_map())        
        # mapper-specific metrics
        if mapper == 'solveSwapsFF':
            circuit_metrics['time_wbo'] = results['time_wbo']
            circuit_metrics['k'] = results['k'] 
            if 'a_star_time' in results.keys():
                circuit_metrics['a_star_time'] = results["a_star_time"]    

    # Flush metric results as another file to disk.
    with open(f'{output_path}/output.{mapper}.{arch}.{rid}txt', "w+") as f:
        f.write(str(circuit_metrics))



def run_all_mappers(input_program_path, arch, args=None):
    mappers = ['tket', 'mqt', 'sabre', 
    #'enfield', #TEMPORARILY DISABLED
               'olsq', 'solveSwapsFF']

    for mapper in mappers:
        run_mapper(input_program_path, mapper, arch, args)
                

if __name__ == '__main__':
    # Please type `python3 rq2script.py -h` for help with args :)
    parser = argparse.ArgumentParser()
    parser.add_argument("prog", help="path to input program file")
    parser.add_argument("--mapper", help="name of mapper used to map input program", choices=['tket', 'mqt', 'sabre', 'olsq', 'solveSwapsFF', 'solveSwapsFF_old']) #enfield currently disabled for CHTC
    parser.add_argument("-o_d", "--output_dir", help="directory for output files")
    parser.add_argument("-a", "--arch", choices=["tokyo", "toronto", "4x4_mesh", "16_linear", 'small_linear', "tokyo_full_diags", "tokyo_no_diags", 'tokyo_drop_2', 'tokyo_drop_6', 'tokyo_drop_10', 'tokyo_drop_14'], help="name of qc architecture")
    parser.add_argument("-to", "--timeout", type=int, help="maximum run time for a mapper in seconds")
    parser.add_argument("-rid", "--run_id", help="ID to uniquely identify this run (if necessary)")
    parser.add_argument("--k", type=int, default=25, help="SolveSwapsFF: k-value")
    parser.add_argument("--towbo", type=int, help="maximum run time for maxsat solver in seconds")
    parser.add_argument("--cyclic", choices=["on", "off"], default="off", help="cyclic mapping")
    parser.add_argument("--no_route",  action="store_true", help="SolveSwapsFF routing")
    parser.add_argument("--weighted",  action="store_true", help="SolveSwapsFF weighting on dist")
    parser.add_argument("--mqt_method", choices=['exact'], help="mqt: method for mapper")
    parser.add_argument("--olsq_obj", choices=['swap', 'fidelity'], help="olsq: optimization objective")
    parser.add_argument("--err", choices=['fake_tokyo', 'fake_linear'], help="olsq: 2 qubit gate error rates")
    args = parser.parse_args()

    arch = None
    if args.arch == None:
        # arch = architectures.ibmToronto
        arch = "tokyo"
    else:
        arch = args.arch

    if args.mapper:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (4096*10**6, hard))
        try: 
            run_mapper(args.prog, args.mapper, arch, args=args)
        except MemoryError:
            print("out of memory")
    else:
        run_all_mappers(args.prog, arch, args=args)
