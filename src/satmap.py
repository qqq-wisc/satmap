import argparse
import ast
import datetime
import os
import numpy as np
import architectures
from common import extract2qubit
import satmap_core
import satmap_hybrid

def transpile(progname, cm, swapNum=1, cnfname='test', sname='out', slice_size=25, max_sat_time=600, routing=True, weighted=False, calibrationData = None, bounded_above=True, hybrid=None):
    chunks = -(len(extract2qubit(progname)) // -slice_size)
    if len(extract2qubit(progname)) == 0:
        print('Exiting... circuit contains no two qubit gates')
        with open(progname) as f:
            return (None, f.read())
    elif hybrid:
        return satmap_hybrid.solve_with_sabre(progname, coupling_map=cm, swap_num=swapNum, explore=hybrid, timeout=max_sat_time)
    elif routing:
        stats = satmap_core.solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname, time_wbo_max=max_sat_time, _calibrationData=calibrationData)
        return (stats, satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, sname))
    elif bounded_above:
        results = satmap_core.solve_bounded_above(progname, cm, swapNum, chunks, pname=cnfname, sname=sname)
        return ((results['cost'], results['a_star_time']), satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, results['solvers'], swaps=results['swaps']))
    else: 
      results = satmap_core.solve(progname, cm, swapNum, chunks, pname=cnfname, sname=sname, _routing=False, _weighted=weighted)
      return ((results['cost'], results['time_wbo'], results['a_star_time']), satmap_core.toQasmFF(os.path.join("aux_files", "qiskit-"+os.path.split(progname)[1]),  cm, swapNum, chunks, sname, swaps=results['swaps']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prog", help="path to input program file")
    parser.add_argument("-o_p", "--output_path", help="where to write the resulting qasm")
    parser.add_argument("-a", "--arch", help="name of qc architecture")
    parser.add_argument("-to", "--timeout", type=int, default=1800,help="maximum run time for a mapper in seconds")
    parser.add_argument("--k", type=int, default=25, help="SolveSwapsFF: k-value")
    parser.add_argument("--cyclic", choices=["on", "off"], default="off", help="cyclic mapping")
    parser.add_argument("--no_route",  action="store_true", help="SolveSwapsFF routing")
    parser.add_argument("--weighted",  action="store_true", help="SolveSwapsFF weighting on dist")
    parser.add_argument("--err", choices=['fake_tokyo', 'fake_linear'], help="olsq: 2 qubit gate error rates")
    parser.add_argument('--hybrid', choices=['vertically', 'horizontally', 'horizontal_sliding_window'])
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
    args = parser.parse_args()
    if args.arch in archs:
        arch = archs[args.arch]
    else:
        with open(args.arch) as f:
            arch = np.array(ast.literal_eval(f.read()))
    hybrid = args.hybrid
    base, _ = os.path.splitext(os.path.basename(args.prog))
    #print(transpile(args.prog, arch, 1, "prob_"+base, "sol_"+base, slice_size=args.k, max_sat_time=args.timeout, routing= not args.no_route, weighted= args.weighted, calibrationData=error_rates[args.err] if args.err else None, bounded_above=False ))
    (stats, qasm) = transpile(args.prog, arch, 1, os.path.join("aux_files", "prob_"+base), os.path.join("aux_files", "sol_"+base), slice_size=args.k, max_sat_time=args.timeout, routing= not args.no_route, weighted= args.weighted, calibrationData=error_rates[args.err] if args.err else None, bounded_above=True, hybrid=hybrid)
    print(stats)
    if args.arch in archs:
        stats["arch"] = args.arch
    else:
        stats['arch'] = f"custom arch with {len(arch)} qubits"
    post_fix = str(datetime.datetime.now()).replace(" ", "_")
    os.makedirs(f"results_{post_fix}", exist_ok =True)
    with open(f"results_{post_fix}/data.txt", "w") as f:
        f.write(str(stats))
    out_file = args.output_path if args.output_path else "mapped_"+os.path.basename(args.prog)
    if out_file != "no_qasm":
        with open(out_file, "w") as f:
            f.write(qasm) 
    