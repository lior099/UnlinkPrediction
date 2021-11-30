import csv
import os
import pickle
import subprocess
import sys
import time

import networkx as nx
import nni
import torch
import yaml
import numpy as np
from importlib import reload

from Code.out_sources import create_feats_to_pkl
from Code.plots import plot_features_histogram, plot_loss_and_auc
from GraphSAINT.data.open_graph_benchmark.networkx_converter import convert

from subprocess import Popen, PIPE


SEP = '/'


class Task:
    def __init__(self, root='.', task_params=None):
        self.root = root
        self.task_params = task_params
        self.destination = None
        self.results_dir = None
        self.data_name = None
        self.scores = {'train': [], 'val': [], 'test': []}
        self.runtimes = []
        self.ids = []

    def prepare(self, graph_params):
        self.data_name = graph_params.data_name
        self.destination = SEP.join([self.root, "task_" + str(self), self.data_name])
        results_destination = SEP.join([self.destination, "results"])
        self.results_dir = SEP.join([results_destination, "_".join(["task", str(self), graph_params.name, "results"])])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.ids.append(graph_params.id)
        # results_file = os.path.join(results_dir, "_".join(["task", str(self), graph_params.name, "results"]) + '.csv')
        # return results_file

    def save_attributes(self, scores):
        # runtime_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "runtime.csv"])])
        # memory_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "memory.csv"])])
        scores_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "scores.csv"])])
        # self.save_to_file([['x'] + self.ids, [self.data_name] + self.runtimes], runtime_destination)
        # self.save_to_file([['x'] + self.ids, [self.data_name] + memory], memory_destination)
        self.save_to_file([['x'] + self.ids,
                           ['train'] + scores['train'],
                           ['val'] + scores['val'],
                           ['test'] + scores.get('test', [])], scores_destination)

    def save_to_file(self, lines_list, path):
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            for line in lines_list:
                wr.writerow(line)

    def reset(self):
        self.scores = {'train': [], 'val': [], 'test': []}
        self.runtimes = []
        self.ids = []


class LineTask(Task):
    # gs_args = None

    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        self.saved_model_path = ''
        self.gs_data_dir = None
        self.gs_config_file = None
        self.state_file = None

    # def run_on_graphs(self, graphs_params):
    #     for graph_params in graphs_params:
    #         self.run(graph_params)

    def prepare(self, graph_params):
        super().prepare(graph_params)
        self.gs_data_dir = SEP.join([self.destination, "graph_saint_data", graph_params.name])
        self.gs_config_file = SEP.join([self.destination, "graph_saint_data", self.data_name+'.yml'])
        self.state_file = SEP.join([self.results_dir, 'state.pkl'])
        self.loss_auc_image = SEP.join([self.results_dir, 'loss_and_auc.png'])

        default_task_params = {'transfer_model': False, 'multi_train': False, 'nni': False, 'train_params': {}}
        default_task_params.update(self.task_params)
        self.task_params = default_task_params

    def reset(self):
        super().reset()
        self.saved_model_path = ''

    def run(self, graph_params):
        if not graph_params.future or graph_params.mode != graph_params.future.mode:
            return
        start = time.time()
        print("task", str(self), 'graph', graph_params.name)
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prepare(graph_params)
        graph = graph_params.get_graph()

        print('current graph:', len(graph.nodes), 'nodes,', len(graph.edges), 'edges')
        future_graph = graph_params.future.get_graph()
        # just to check edge sharing: y = [1 if edge in future_graph.edges else 0 for edge in graph.edges]
        line_graph = nx.line_graph(graph)
        line_future_graph = nx.line_graph(future_graph)

        line_graph = nx.convert_node_labels_to_integers(line_graph, label_attribute='edge')
        print('current line graph:', len(line_graph.nodes), 'nodes,', len(line_graph.edges), 'edges')

        force_create_feats = False
        if not graph_params.feats or force_create_feats:
            print("Didn't find any features_{}.pkl file for graph " + graph_params.name + '. Creating:')
            create_feats_to_pkl(line_graph, graph_params)
        # for_testing(line_graph)
        graph_feats, feats_names = graph_params.get_feats(line=True)
        # plot_features_histogram(graph_feats, self.results_dir, feats_names)
        print("This graph mode is:", graph_params.mode)
        # all_auc_dict = {'train': [], 'val': [], 'test': []}
        convert(line_graph, self.gs_data_dir, future_graph=line_future_graph, feats=graph_feats, mode=graph_params.mode, test_seed=0)
        self.create_config(graph_params.mode)
        self.run_gs()
        model, auc_dict, history = self.load_gs_results()
        # TODO doesnt work when on gpu, fix.
        if graph_params.mode != 'test' and not torch.cuda.is_available():
            plot_loss_and_auc(history['train_loss'], history['train_auc'], history['val_loss'], history['val_auc'], self.loss_auc_image, self.data_name, graph_params.name)
        # all_auc_dict = {data_role: auc_list + [auc_dict[data_role]] for data_role, auc_list in all_auc_dict.items()}
        # auc_dict = {data_role: round(np.average(auc_list), 3) for data_role, auc_list in all_auc_dict.items()}


        if self.task_params['transfer_model']:
            self.saved_model_path = self.state_file
        # raise Exception()
        # self.runtimes.append(time.time() - start)
        print("auc results:", auc_dict)
        self.update_scores(auc_dict, graph_params.mode)
        # self.scores = {key: key_scores + [auc_dict[key]] for key, key_scores in self.scores.items()}

    def run_gs(self):
        if not torch.cuda.is_available():
            self.task_params['gpu'] = '-1'
            print('running gs on cpu')
        else:
            print('running gs on gpu', os.environ.get('CUDA_VISIBLE_DEVICES', self.task_params['gpu']))
        python_path = sys.executable
        # print('python_path:', python_path)
        # python_path = '/home/dsi/shifmal2/anaconda3/envs/py3/bin/python' if torch.cuda.is_available() else 'python'
        command = [python_path, 'GraphSAINT/graphsaint/pytorch_version/train.py', '--data_prefix', self.gs_data_dir,
                   '--train_config', self.gs_config_file, '--gpu', self.task_params['gpu'], '--dir_log', self.results_dir]
        # command = [python_path, 'GraphSAINT/graphsaint/pytorch_version/train.py', '--data_prefix', self.gs_data_dir,
        #            '--train_config', "GraphSAINT/train_config/table2/test.yml", '--gpu', gpu, '--dir_log', self.results_dir]
        if self.saved_model_path:
            with open(self.saved_model_path, 'rb') as file:
                args_global, timestamp, auc_dict, history = pickle.load(file)
                state_dict_path = '{}/pytorch_models/saved_model.pkl'.format(args_global.dir_log)
                command += ['--saved_model_path', state_dict_path]
        # print(' '.join(command))
        # LineTask.gs_args = command[2:]
        # move this to top
        from GraphSAINT.graphsaint.pytorch_version.train import start_gs_train
        start_gs_train(gs_args=command[2:])

        # p = Popen(command, stderr=subprocess.PIPE)
        # output, err = p.communicate()
        # p.wait()
        # if p.returncode != 0:
        #     raise Exception(err)

    def load_gs_results(self):
        with open(self.state_file, 'rb') as file:
            args_global, timestamp, auc_dict, history = pickle.load(file)
        # return self.open_gs_model(args_global, timestamp), auc_dict, history
        return None, auc_dict, history

    def open_gs_model(self, args_global, timestamp):
        from GraphSAINT.graphsaint.pytorch_version.train import prepare
        from GraphSAINT.graphsaint.utils import parse_n_prepare
        state_dict_path = '{}/pytorch_models/saved_model.pkl'.format(args_global.dir_log, timestamp)
        train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
        model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
        state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
        model_eval.load_state_dict(state_dict)
        model_eval.eval()

        return model_eval

    def create_config(self, mode, train_params=None):
        if not train_params:
            train_params = self.task_params['train_params']
        task_params_to_file = {key: value for key, value in self.task_params.items() if key not in ['train_params']}
        default = {
            'network': {
                'dim': 256,
                'aggr': 'concat',
                'loss': 'sigmoid',
                'arch': '1-0-1-0',
                'act': 'relu',
                'bias': 'norm'},
            'params': {
                'lr': 0.01,
                'dropout': 0.1,
                'weight_decay': 0.,
                'sample_coverage': 50,
                'norm_loss': True,
                'norm_aggr': True,
                'q_threshold': 50,
                'q_offset': 0},
            'phase': {
                'end': 300,
                'sampler': 'edge',
                'size_subg_edge': 4000,
                'size_subgraph': 1000,
                'size_frontier': 300,
                'num_root': 1250,
                'depth': 2
            },
            'task_params': task_params_to_file
        }
        # fill every missing value of train_params from default
        train_params = {group: [{key: train_params.get(key, value) for key, value in params.items()}] for group, params in default.items()}
        if mode == 'test':
            train_params['phase'][0]['end'] = 0
        with open(self.gs_config_file, 'w') as file:
            yaml.dump(train_params, file, default_flow_style=False)

    def avg_scores(self, list_of_scores):
        roles = list(list_of_scores[0].keys())
        snapshots = len(list_of_scores[0][roles[0]])
        avg_all = {}
        for role in roles:
            for snapshot in range(len(list_of_scores[0][role])):
                scores_of_mode_and_snapshot = [scores[role][snapshot] for scores in list_of_scores]
                avg = sum(scores_of_mode_and_snapshot) / len(scores_of_mode_and_snapshot)
                avg_all[role] = avg_all.get(role, [])
                avg_all[role].append(round(avg, 3))
        return avg_all

    def update_scores(self, auc_dict, mode):
        if mode == 'train':
            self.scores['train'].append(auc_dict['train'])
            self.scores['val'].append(auc_dict['val'])
        elif mode == 'test':
            self.scores['test'].append(auc_dict['test'])
        elif mode == 'train_test':
            self.scores['train'].append(auc_dict['train'])
            self.scores['val'].append(auc_dict['val'])
            self.scores['test'].append(auc_dict['test'])
    def __str__(self):
        return 'line'


def for_testing(graph):
    num_bins = 100
    degrees = [i[1] for i in graph.degree]
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(degrees, num_bins, color='red', alpha=0.7)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

    import seaborn

    seaborn.histplot(degrees, bins=num_bins)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()
    raise Exception()


if __name__ == '__main__':
    # os.system('python test.py --data_prefix aa --train_config '
    #           'GraphSAINT/train_config/table2/test.yml --gpu -1 --dir_log aa')
    # from subprocess import Popen, PIPE

    p = Popen(['python', 'test.py', '--data_prefix', 'aa'], stdout=PIPE)
    print('1#')
    output, err = p.communicate()
    print('output', str(output))
    print('err', str(err))
    rc = p.returncode
    # a = LineTask()
    # print('a',a)
    # print("_".join(["task", str(a)]))
