import datetime
import itertools
import sys
import os

from OtherPapers.XGBoost.xgboost_task import XGBoostTask
from OtherPapers.logistic.logistic_main import LogisticTask

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
from Code.plots import plot_all_results, plot_categories_auc
from Code.tasks import LineTask, MulTask, EmbedTask, ConcatTask
from copy import copy

SEP = '/'


def save_to_file(lines_list, path):
    with open(path, 'w', newline='') as file:
        wr = csv.writer(file)
        for line in lines_list:
            wr.writerow(line)


def get_params_from_pickles(dir, data_name, task_params, line=False):
    mode = task_params['mode']
    # edge_features = not task_params.get('nodes_features', True)
    # features_type = task_params.get('features_type', 'node')
    graphs_params = {}
    graphs_feats = {}
    graph_edge_feats = {}
    for filename in os.listdir(dir):
        number = int(filename.rstrip('.pkl').split('_')[-1])
        if 'graph' in filename:
            name = str(number) + '_' + data_name
            params = Params(data_name=data_name, name=name, id=number, path=SEP.join([dir, filename]), pkl=True)
            graphs_params[number] = params
        elif '.pkl' in filename and \
                (line and filename.startswith('line_features') or
                 not line and filename.startswith('features')):
            graphs_feats[number] = SEP.join([dir, filename])
        elif '.pkl' in filename and filename.startswith('edge_features'):
            graph_edge_feats[number] = SEP.join([dir, filename])
    graphs_params_list = []
    graphs_params = {key: graphs_params[key] for key in sorted(graphs_params)}
    for number, params in graphs_params.items():
        params.feats = graphs_feats.get(number)
        params.edge_feats = graph_edge_feats.get(number)
        if mode is None:
            params.mode = "train_test"
        elif mode == 'const_train':
            params.mode = "const_train_test"
        else:
            assert mode == 'multi_train'
            if number + 2 in graphs_params.keys():
                params.mode = "train"
            else:
                params.mode = "test"

        if graphs_params_list:
            graphs_params_list[-1].future = params
        graphs_params_list.append(params)

    graphs_params_list = sorted(graphs_params_list, key=lambda x: x.id)
    return graphs_params_list


def run_task(task, graphs_params):
    print('Running task', task, 'on data', graphs_params[0].data_name)
    cross_validation = 5 if task.task_params['nni'] else task.task_params['cross_validation_iters']
    list_of_scores = []
    list_of_categories = []
    for cross_validation_run in range(cross_validation):
        print("Running cross-validation number", cross_validation_run + 1, "out of", cross_validation)
        task.run_on_snapshots(graphs_params)
        print("FINAL AUC of cross-validation number", cross_validation_run + 1, ":", task.scores)
        list_of_scores.append(task.scores)
        if task.task_params.get('categories'):
            list_of_categories.append(task.categories)
        task.reset()
    avg_scores = task.avg_scores(list_of_scores)
    final_scores = task.get_final_scores(avg_scores)
    if task.task_params.get('categories'):
        final_categories = task.get_final_scores(task.avg_scores(list_of_categories))
        plot_categories_auc(final_categories, task.categories_image, task.data_name, str(task))
    if task.task_params['nni']:
        for score in avg_scores['val']:
            nni.report_intermediate_result(score)
        score_dict = {'default': final_scores['val'], 'train': final_scores['train'],
                      'test': final_scores['test']}
        nni.report_final_result(score_dict)
    print("Average Results from all cross-validation runs:", avg_scores)
    # print(f"FINAL RESULTS of task {task} on data {graphs_params[0].data_name} (average on snapshots):", final_scores)
    task.save_attributes(avg_scores, final_scores)
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


def prepare_run(run_type='line', eval_future=False, data=None, feats_idx=None):
    results_dir = "Results"
    args = get_args()
    data_name = (data if data is not None else args.data).lower()
    data_dir = SEP.join(["Data", data_name])

    # if feats_idx is None:
    #     feats_idx = [0, 1, 2, 3, 5, 6]

    if run_type in ['line', 'mul', 'concat']:
        task_params = {'transfer_model': True, 'categories': True}
    elif run_type in ['logistic', 'xgboost']:
        # features_type can be: 'node', 'edge', 'node_edge'
        task_params = {'transfer_model': True, 'features_type': 'node'}
    else:
        assert run_type in ['embed']
        task_params = { 'transfer_model': False, 'eval_future': eval_future}
    task_params['nni'] = args.nni
    task_params['gpu'] = args.gpu
    # Modes are: None, 'const_train', 'multi_train'
    task_params['mode'] = 'const_train'
    task_params['create_feats'] = False
    task_params['cross_validation_iters'] = 1
    task_params['feats_idx'] = feats_idx
    graphs_params = get_params_from_pickles(data_dir, data_name, task_params, line=(run_type == 'line'))

    if task_params['nni']:
        task_params['train_params'] = nni.get_next_parameter()
        task_params['train_params']['dim'] = int(task_params['train_params']['dim'])
        # task_params['train_params']['end'] = 10
    else:
        task_params['train_params'] = {'end': 1000}
        # task_params['train_params'] = {'end': 10, "dim": 1024, "arch": "1-1-0", "lr": 0.22954203138090948, "dropout": 0.1, "sample_coverage": 41.0, "sampler": "mrw", "size_subg_edge": 4400.0}
        # task_params['train_params'] = {'end': 1000, 'loss_type': 'node', "dim": 128, "arch": "1-0-1-0", "lr": 0.00009300589780635062,
        #                                "dropout": 0.2, "sample_coverage": 10.0, "sampler": "node",
        #                                "size_subg_edge": 1500.0}
    if run_type == 'line':
        task = LineTask(results_dir, task_params)
    elif run_type == 'mul':
        task = MulTask(results_dir, task_params)
    elif run_type == 'embed':
        task = EmbedTask(results_dir, task_params)
    elif run_type == 'concat':
        task = ConcatTask(results_dir, task_params)
    elif run_type == 'logistic':
        task = LogisticTask(results_dir, task_params)
    elif run_type == 'xgboost':
        task = XGBoostTask(results_dir, task_params)
    else:
        raise Exception("Only 'line', 'mul', 'embed' and 'concat' tasks are implemented, not '" + run_type + "'")
    return task, graphs_params

def do_runs():
    # run_type: 'line', 'mul', 'embed', 'concat', 'xgboost', (logistic)
    # old data: 'enron', 'radoslaw', 'facebook', 'catalano', 'reality', 'haggle'
    # new data: 'facebook', 'reality', 'haggle'
    for run_type in ['mul', 'concat']:
        for data in ['enron', 'radoslaw', 'facebook', 'catalano', 'reality', 'haggle']:
            task, graphs_params = prepare_run(run_type=run_type, data=data)
            run_task(task=task, graphs_params=graphs_params)
            print(f"Finished run {run_type} with feats_idx {np.array(task.feats_names)[task.task_params['feats_idx']]} on data {data}: {task.final_scores}")
    print('Finished all runs!')

if __name__ == "__main__":
    print('Start: ', datetime.datetime.now())
    random.seed(0)
    if 'Code' not in os.listdir(os.getcwd()):
        os.chdir("..")
        if 'Code' not in os.listdir(os.getcwd()):
            raise Exception(
                "Bad pathing, use the command os.chdir() to make sure you work on UnlinkPrediction directory")

    # run_type: 'line', 'mul', 'embed', 'concat', 'xgboost', (logistic)
    # old data: 'enron', 'radoslaw', 'facebook', 'catalano'
    # new data: 'facebook', 'reality', 'haggle'

    # task, graphs_params = prepare_run(run_type='embed', data='reality', eval_future=True)
    # run_task(task=task, graphs_params=graphs_params)
    # print('Finished run!')

    do_runs()
    print('End: ', datetime.datetime.now())