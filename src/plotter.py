import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import numpy as np
import shutil
import qiskit
import sys

SATMAP = "SATMAP"
MQT = "EX-MQT"
OLSQ = "TB-OLSQ"

def build_dicts_heuristic():
    [res_dicts_us_cost, res_dicts_tket, res_dicts_sabre, res_dicts_mqt] = [{"tokyo" : {}, "toronto" : {}, "16_linear" : {}, "4x4_mesh" : {}, "tokyo_full_diags" : {}, "tokyo_no_diags" : {}} for _ in range(4)]
    directory_parent  ='results/'
    for directory_program in os.scandir(directory_parent):
        entry_name = directory_program.name
        entry_name_split = entry_name.split('.')
        if entry_name_split[0] == 'results' and entry_name_split[-1] == 'qasm':
            # #Add 'header' information for program
            name_no_qasm = entry_name_split[1]
            #go into directory, compile results for each solver
            for output_file in os.scandir(directory_program):
                output_file_split = output_file.name.split('.')
                if not output_file_split[-1] == 'txt':
                    continue
                arch = output_file_split[2]
                solver = output_file_split[1]
                with open(output_file.path) as f:
                    data = f.read()
                output_file_dict = ast.literal_eval(data) 
                if 'timeout' in output_file_dict.keys():
                    continue
                else:
                    if solver == 'tket':
                        res_dicts_tket[arch][name_no_qasm] = output_file_dict["g_add"]
                    elif solver == 'sabre':
                        res_dicts_sabre[arch][name_no_qasm] = output_file_dict["g_add"]
                    elif solver == "mqt":
                        res_dicts_mqt[arch][name_no_qasm] = output_file_dict["g_add"]
                    elif solver == "solveSwapsFF":
                        if name_no_qasm not in res_dicts_us_cost[arch] or output_file_dict["g_add"] < res_dicts_us_cost[arch][name_no_qasm]:
                                res_dicts_us_cost[arch][name_no_qasm] = output_file_dict["g_add"]
    return (res_dicts_us_cost, res_dicts_tket, res_dicts_sabre, res_dicts_mqt)

def build_dicts_constraint_based():
    res_dicts_us, res_dicts_mqt_ex, res_dicts_olsq = [{"tokyo" : {}, "toronto" : {}, "16_linear" : {}, "4x4_mesh" : {}, "tokyo_full_diags" : {}, "tokyo_no_diags" : {}} for _ in range(3)]
    # xaxis = list(range())
    directory_parent  ='results'
    for directory_program in os.scandir(directory_parent):
        entry_name = directory_program.name
        entry_name_split = entry_name.split('.')
        # print(entry_name)
        if entry_name_split[0] == 'results' and entry_name_split[-1] == 'qasm':
            # print(entry_name)
            # #Add 'header' information for program
            name_no_qasm = entry_name_split[1]

            #go into directory, compile results for each solver
            for output_file in os.scandir(directory_program):
                output_file_split = output_file.name.split('.')
                if not output_file_split[-1] == 'txt':
                    continue
                solver = output_file_split[1]
                arch = output_file_split[2]
                with open(output_file.path) as f:
                    data = f.read()
                output_file_dict = ast.literal_eval(data) 
                
                if 'timeout' in output_file_dict.keys() or output_file_dict["g_add"] < -100:
                    continue
                else:
                    if solver == 'solveSwapsFF':
                        if name_no_qasm not in res_dicts_us[arch] or output_file_dict["time"] < res_dicts_us[arch][name_no_qasm]:
                                res_dicts_us[arch][name_no_qasm] = output_file_dict["time"]
                    elif solver == "olsq":
                        res_dicts_olsq[arch][name_no_qasm] = output_file_dict["time"]
                    elif solver == "mqt_exact":
                        res_dicts_mqt_ex[arch][name_no_qasm] = output_file_dict["time"]
    return (res_dicts_us, res_dicts_mqt_ex, res_dicts_olsq)

def plot_against_heuristic(res_dicts_us_cost, res_dicts_heuristic, name_str):
    for key in res_dicts_us_cost["tokyo"].keys():
        if res_dicts_us_cost["tokyo"][key] < 0:
            res_dicts_us_cost["tokyo"][key] = 0

    diff_heuristic = { key: (res_dicts_heuristic["tokyo"][key])/ (res_dicts_us_cost["tokyo"][key]) for key in res_dicts_heuristic["tokyo"].keys() & res_dicts_us_cost["tokyo"].keys() if res_dicts_us_cost["tokyo"][key] > 0}
    diff_heuristic.update({ key: 1 for key in res_dicts_heuristic["tokyo"].keys() & res_dicts_us_cost["tokyo"].keys() if res_dicts_us_cost["tokyo"][key] == 0 and res_dicts_heuristic["tokyo"][key] == 0})
    zeros = { key: 26  for key in res_dicts_heuristic["tokyo"].keys() & res_dicts_us_cost["tokyo"].keys() if res_dicts_us_cost["tokyo"][key] == 0 and res_dicts_heuristic["tokyo"][key] != 0}
    data = pd.DataFrame({"program" : sorted(list(diff_heuristic.keys()), key=diff_heuristic.get), "difference" : sorted(list(diff_heuristic.values()))})
    zData = pd.DataFrame({"program" : zeros.keys(), "difference" : sorted(list(zeros.values()))})
    sns.set_theme(style='whitegrid', palette="colorblind", font_scale=1.25)

    points = sns.scatterplot(data=data,
        x = "program", y = "difference", edgecolor=None)
    points.grid(False, axis="x")
    points.set( xlabel = None, xticklabels=[])

    points2 = sns.scatterplot(data=zData,
        x = "program", y = "difference", edgecolor=None)
    points2.set(xlabel=None, xticklabels=[])
    x = np.arange(0, len(diff_heuristic.keys()) + len(zeros), 1)
    avg = [statistics.mean(diff_heuristic.values())]*len(x)
    plt.plot(x,avg, "black", label="Mean: " + str(round(statistics.mean(diff_heuristic.values()),2)))
    plt.plot(x,[1]*len(x), "black", linestyle="dashed", label='SATMAP')
    plt.ylim(0, 26)
    plt.ylabel("Cost ratio")
    plt.title(name_str)
    plt.tight_layout()
    leg = plt.legend()
    points.figure.savefig(f"fig11_{name_str}.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close() 

def plot_olsq(res_dicts_us, res_dicts_olsq):
    intersection = [key  for key in res_dicts_olsq["tokyo"].keys() & res_dicts_us["tokyo"].keys()]
    data = pd.DataFrame({"circuit" : intersection+intersection, "method": [SATMAP for _ in range(len(intersection))] + [OLSQ for _ in range(len(intersection))]  , "time": [res_dicts_us["tokyo"][key] for key in intersection]+[res_dicts_olsq["tokyo"][key] for key in intersection]})
    sns.set_theme(style='whitegrid', palette="colorblind", font_scale=2.5)

    barchart = sns.catplot(
        data=data,
        x = "circuit",
        y= "time",
        hue = "method",
        kind="bar", aspect=8/1
    )
    plt.yscale('log')
    plt.xticks(rotation=45, horizontalalignment='right')
    barchart.set(ylabel = "time (s)")
    barchart.savefig("rq1_barchart_all.pdf")
    plt.close() 

def plot_jku(res_dicts_us_cost, res_dicts_jku_exact, res_dicts_olsq):
    intersection = [key  for key in res_dicts_olsq["tokyo"].keys() & res_dicts_us["tokyo"].keys() & res_dicts_jku_exact["tokyo"].keys()]
    data = pd.DataFrame({"circuit" : intersection+intersection+intersection, "method": [SATMAP for _ in range(len(intersection))] + [OLSQ for _ in range(len(intersection))] + [MQT for _ in range(len(intersection))]  , "time": [res_dicts_us["tokyo"][key] for key in intersection]+[res_dicts_olsq["tokyo"][key] for key in intersection] +[res_dicts_jku_exact["tokyo"][key] for key in intersection] })
    #data = pd.DataFrame(np.array([[res_dicts_us["tokyo"][key], res_dicts_olsq["tokyo"][key]] for key in intersection]), columns = ["SatMap", "OLSQ"], index=intersection)
    sns.set_theme(style='whitegrid', palette="colorblind", font_scale=3)

    barchart = sns.catplot(
        data=data,
        x = "circuit",
        y= "time",
        hue = "method",
        kind="bar", aspect=3/1
    )
    plt.yscale('log')
    barchart.set(ylabel = "time (s)")

    barchart.savefig("rq1_barchart_jku.pdf")
    plt.close()     

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == "-h":
        (res_dicts_us_cost, res_dicts_tket, res_dicts_sabre, res_dicts_mqt) = build_dicts_heuristic()   
        plot_against_heuristic(res_dicts_us_cost, res_dicts_tket, "tket")
        plot_against_heuristic(res_dicts_us_cost, res_dicts_sabre, "sabre")
        plot_against_heuristic(res_dicts_us_cost, res_dicts_mqt, "mqt")
    elif arg == "-c":
        (res_dicts_us, res_dicts_mqt_ex, res_dicts_olsq) = build_dicts_constraint_based()   
        plot_olsq(res_dicts_us, res_dicts_olsq)
        plot_jku(res_dicts_us, res_dicts_mqt_ex, res_dicts_olsq)
