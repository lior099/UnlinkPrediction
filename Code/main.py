import sys
import os
sys.path.append(os.path.abspath('..'))
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import argparse
import csv
import os
import random
import time

import nni
from memory_profiler import memory_usage
import numpy as np

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


def get_params_from_pickles(dir, data_name, multi_train):
    graphs_params = {}
    graphs_feats = {}
    for filename in os.listdir(dir):
        number = int(filename.rstrip('.pkl').split('_')[-1])
        if 'graph' in filename:
            name = str(number) + '_' + data_name
            params = Params(data_name=data_name, name=name, id=number, path=SEP.join([dir, filename]), pkl=True)
            graphs_params[number] = params
        elif 'features' in filename and '.pkl' in filename:
            graphs_feats[number] = SEP.join([dir, filename])
    graphs_params_list = []
    graphs_params = {key: graphs_params[key] for key in sorted(graphs_params)}
    for number, params in graphs_params.items():
        params.feats = graphs_feats.get(number)
        if not multi_train:
            params.mode = "train_test"
        elif number + 2 in graphs_params.keys():
            params.mode = "train"
        else:
            params.mode = "test"

        if graphs_params_list and (number + 2 in graphs_params.keys() or number + 1 not in graphs_params.keys()):
            graphs_params_list[-1].future = params
        graphs_params_list.append(params)

    graphs_params_list = sorted(graphs_params_list, key=lambda x: x.id)
    return graphs_params_list

def run_task(task, graphs_params):
    print('Running task',task, 'on data', graphs_params[0].data_name)
    cross_validation = 5 if task.task_params['nni'] else 1
    list_of_scores = []
    for cross_validation_run in range(cross_validation):
        print("Running cross-validation number", cross_validation_run + 1, "out of", cross_validation)
        for graph_params in graphs_params:
            task.run(graph_params)
        print("FINAL AUC of cross-validation number", cross_validation_run + 1, ":", task.scores)
        list_of_scores.append(task.scores)
        task.reset()
    avg_scores = task.avg_scores(list_of_scores)
    final_scores = {role: round(sum(scores) / len(scores), 3) for role, scores in avg_scores.items()}
    if task.task_params['nni']:
        for score in avg_scores['val']:
            nni.report_intermediate_result(score)
        score_dict = {'default': min(final_scores['train'], final_scores['val']), 'train': final_scores['train'], 'test': final_scores['test']}
        nni.report_final_result(score_dict)
    print("Average Results from all cross-validation runs:", avg_scores)
    print("FINAL RESULTS (average on snapshots):", final_scores)
    task.save_attributes(avg_scores)
        # memory_list = []
        # for graph_params in graphs_params:
        #     memory = memory_usage((task.run, (graph_params,)), max_iterations=1)
        #     memory_list.append(max(memory))
        #     task.save_attributes(memory_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='enron', type=str)
    parser.add_argument("--nni", default=0, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(0)
    if 'Code' not in os.listdir(os.getcwd()):
        os.chdir("..")
        if 'Code' not in os.listdir(os.getcwd()):
            raise Exception("Bad pathing, use the command os.chdir() to make sure you work on UnlinkPrediction directory")
    results_dir = "Results"
    # dblp, imdb, enron, radoslaw, facebook
    args = get_args()
    data_name = args.data
    data_dir = SEP.join(["Data", data_name])


    task_params = {'transfer_model': True, 'multi_train': True, 'nni': args.nni, 'gpu': args.gpu}
    graphs_params = get_params_from_pickles(data_dir, data_name, task_params['multi_train'])
    if task_params['nni']:
        task_params['train_params'] = nni.get_next_parameter()
        task_params['train_params']['dim'] = int(task_params['train_params']['dim'])
        # task_params['train_params']['end'] = 10
    else:
        # task_params['train_params'] = {'end': 10}
        # task_params['train_params'] = {'end': 10, "dim": 1024, "arch": "1-1-0", "lr": 0.22954203138090948, "dropout": 0.1, "sample_coverage": 41.0, "sampler": "mrw", "size_subg_edge": 4400.0}
        task_params['train_params'] = {'end': 10, "dim": 128, "arch": "1-0-1-0", "lr": 0.09300589780635062, "dropout": 0.2, "sample_coverage": 10.0, "sampler": "node", "size_subg_edge": 1500.0}
    task = LineTask(results_dir, task_params)
    run_task(task=task, graphs_params=graphs_params)
    print('Finished run!')
    # if task_params['nni']:
    #     nni.report_final_result(round(np.average(task.scores['val']), 3))
    # plot_all_results()

    print()


