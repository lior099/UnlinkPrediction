import csv
import os
import random
import time

from memory_profiler import memory_usage

import matplotlib.pyplot as plt
import networkx as nx

from Code.graphs import random_graph, Params
from Code.plots import plot_all_results
from Code.tasks import LineTask
from copy import copy

SEP = '/'


def save_to_file(lines_list, path):
    with open(path, 'w', newline='') as file:
        wr = csv.writer(file)
        for line in lines_list:
            wr.writerow(line)


def get_params_from_pickles(dir, data_name):
    graphs_params = {}
    graphs_feats = {}
    for filename in os.listdir(dir):
        number = int(filename.rstrip('.pkl').split('_')[-1])
        if 'graph' in filename:
            name = str(number) + '_' + data_name
            params = Params(name=name, id=number, path=SEP.join([dir, filename]), pkl=True)
            graphs_params[number] = params
        elif 'mx_line' in filename and '.pkl' in filename:
            graphs_feats[number] = SEP.join([dir, filename])
    graphs_params_list = []
    for number, params in graphs_params.items():
        if number + 1 in graphs_params.keys():
            params.future = copy(graphs_params[number + 1])
            params.feats = graphs_feats.get(number)
            graphs_params_list.append(params)
    graphs_params_list = sorted(graphs_params_list, key=lambda x: x.id)
    return graphs_params_list

def run_task(task, data_name, graphs_params):
    print('Running task',task, 'on data', data_name)

    memory_list = []

    for graph_params in graphs_params:
        memory = memory_usage((task.run, (graph_params, data_name)), max_iterations=1)
        memory_list.append(max(memory))
        task.save_attributes(memory_list)



if __name__ == "__main__":
    random.seed(0)
    if 'Code' not in os.listdir(os.getcwd()):
        raise Exception("Bad pathing, use the command os.chdir() to make sure you work on UnlinkPrediction directory")
    results_dir = "Results"
    data_dir = SEP.join(["Data", "DBLP"])
    data_name = 'dblp'

    # graphs_params = [Params(name='0_dblp', id=0, path='C:/Users/shifmal2/Downloads/for_lior/DBLP/graph_0.pkl', pkl=True),
    #                  Params(name='1000_10_random', id=1000, size=1000, rank=10),
    #                  Params(name='2000_10_random', id=2000, size=2000, rank=10),
    #                  Params(name='3000_10_random', id=3000, size=3000, rank=10),
    #                  Params(name='4000_10_random', id=4000, size=4000, rank=10),
    #                  Params(name='5000_10_random', id=5000, size=5000, rank=10)]
    graphs_params = get_params_from_pickles(data_dir, data_name)
    task_params = {}
    task = LineTask(results_dir, task_params)
    run_task(task=task, data_name=data_name, graphs_params=graphs_params)
    # plot_all_results()

    print()
