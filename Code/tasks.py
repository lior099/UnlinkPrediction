import csv
import math
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
from node2vec import Node2Vec
from sklearn import metrics

from Code.graphs import remove_test_edges_from_train, prepare_graphs
from Code.out_sources import create_feats_to_pkl
from Code.plots import plot_features_histogram, plot_loss_and_auc, plot_edges_over_snapshots, plot_disappearing_edges
from Code.toy import Toy
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
        self.name = None
        self.graph_params = None
        self.feats_names = None
        self.scores = []
        self.final_scores = None
        self.runtimes = []
        self.ids = []

    def prepare(self, graph_params):
        self.data_name = graph_params.data_name
        self.name = graph_params.name
        self.destination = SEP.join([self.root, "task_" + str(self), self.data_name])
        results_destination = SEP.join([self.destination, "results"])
        self.results_dir = SEP.join([results_destination, "_".join(["task", str(self), graph_params.name, "results"])])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.ids.append(graph_params.id)
        self.graph_params = graph_params
        # results_file = os.path.join(results_dir, "_".join(["task", str(self), graph_params.name, "results"]) + '.csv')
        # return results_file

    def save_attributes(self, scores, final_score):
        self.final_scores = final_score
        scores_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "scores.csv"])])
        self.save_to_file([['X'] + self.ids,
                           ['Scores'] + scores,
                           ['Final'] + [final_score]], scores_destination)

    def save_to_file(self, lines_list, path):
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            for line in lines_list:
                wr.writerow(line)

    def reset(self):
        self.scores = []
        self.final_scores = None
        self.runtimes = []
        self.ids = []

    def update_scores(self, score):
        self.scores.append(score)

    def avg_scores(self, list_of_scores):
        avg_all = []
        for snapshot in range(len(list_of_scores[0])):
            scores_of_snapshot = [scores[snapshot] for scores in list_of_scores]
            avg = sum(scores_of_snapshot) / len(scores_of_snapshot)
            avg_all.append(round(avg, 3))
        return avg_all

    def get_final_scores(self, avg_scores):
        avg_scores = [score for score in avg_scores if not math.isnan(score)]
        return round(sum(avg_scores) / len(avg_scores), 3)

    def run_on_snapshots(self, graphs_params):
        for graph_params in graphs_params:
            self.run(graph_params)

    def run(self, graph_params):
        pass

    # def filter_features(self, to_dict=False):
    #     if to_dict:
    #         graph_feats = {node: np.array(feats)[0] for node, feats in
    #                             zip(sorted(graph_params.get_graph()), graph_feats)}
    #         if self.task_params['feats_idx'] is not None:
    #             # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
    #             graph_feats = {node: feats[self.task_params['feats_idx']] for node, feats in graph_feats.items()}
    #     else:
    #         graph_feats_dict = {np.array(feats)[0] for node, feats in
    #                             zip(sorted(graph_params.get_graph()), graph_feats)}
    #         if self.task_params['feats_idx'] is not None:
    #             # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
    #             graph_feats_dict = {node: feats[self.task_params['feats_idx']] for node, feats in
    #                                 graph_feats_dict.items()}
    #     return gr

class GSTask(Task):
    # gs_args = None

    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        self.saved_model_path = ''
        self.gs_data_dir = None
        self.gs_config_file = None
        self.state_file = None
        self.loss_type = 'node'
        self.loss_action = None
        self.loss_auc_image = None
        self.categories_image = None
        self.scores = {'train': [], 'val': [], 'test': []}
        self.categories = {'Train-Train': [],
                           'Train-Validation': [],
                           'Train-Test': [],
                           'Validation-Validation': [],
                           'Validation-Test': [],
                           'Test-Test': []}


    # def run_on_graphs(self, graphs_params):
    #     for graph_params in graphs_params:
    #         self.run(graph_params)
    def run(self, graph_params):
        if not graph_params.future or graph_params.mode != graph_params.future.mode:
            return
        assert self.loss_type == 'edge'
        start = time.time()
        print("task", str(self), 'graph', graph_params.name)
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prepare(graph_params)
        graph = graph_params.get_graph(transform=True)

        print('current graph:', len(graph.nodes), 'nodes,', len(graph.edges), 'edges')
        future_graph = graph_params.future.get_graph()

        force_create_feats = self.task_params['create_feats']
        if not graph_params.feats or force_create_feats:
            print(f"Didn't find any features_X.pkl file for graph {graph_params.name}. Creating: (force is {str(force_create_feats)})")
            create_feats_to_pkl(graph, graph_params, line=False)
        # for_testing(line_graph)
        graph_feats, self.feats_names = graph_params.get_feats()
        # plot_features_histogram(graph_feats, self.results_dir, feats_names)
        print("This graph mode is:", graph_params.mode)
        # all_auc_dict = {'train': [], 'val': [], 'test': []}
        labels = graph_params.get_labels()
        convert(graph, self.gs_data_dir, labels, future_graph=future_graph, feats=graph_feats, mode=graph_params.mode,
                test_seed=0, role=graph_params.role)
        self.create_gs_config()
        self.run_gs()
        model, auc_dict, history, preds = self.load_gs_results()
        # TODO doesnt work when on gpu, fix.
        if graph_params.mode != 'test' and not torch.cuda.is_available():
            plot_loss_and_auc(history['train_loss'], history['train_auc'], history['val_loss'], history['val_auc'],
                              self.loss_auc_image, self.data_name, graph_params.name, task=str(self))
            categories_auc = self.eval_categories(preds, graph_params)
        # all_auc_dict = {data_role: auc_list + [auc_dict[data_role]] for data_role, auc_list in all_auc_dict.items()}
        # auc_dict = {data_role: round(np.average(auc_list), 3) for data_role, auc_list in all_auc_dict.items()}

        if self.task_params['transfer_model']:
            self.saved_model_path = self.state_file
        # raise Exception()
        # self.runtimes.append(time.time() - start)
        print("auc results:", auc_dict)
        self.update_scores(auc_dict, categories_auc)
        # self.scores = {key: key_scores + [auc_dict[key]] for key, key_scores in self.scores.items()}

    def save_attributes(self, scores, final_scores):
        self.final_scores = final_scores
        # runtime_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "runtime.csv"])])
        # memory_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "memory.csv"])])
        scores_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "scores.csv"])])
        # self.save_to_file([['x'] + self.ids, [self.data_name] + self.runtimes], runtime_destination)
        # self.save_to_file([['x'] + self.ids, [self.data_name] + memory], memory_destination)
        self.save_to_file([['x'] + self.ids,
                           ['train'] + scores['train'],
                           ['val'] + scores['val'],
                           ['test'] + scores.get('test', []),
                           ['final'] + [str(final_scores)]], scores_destination)

    def reset(self):
        self.saved_model_path = ''
        self.scores = {'train': [], 'val': [], 'test': []}
        self.final_scores = None
        self.categories = {'Train-Train': [],
                           'Train-Validation': [],
                           'Train-Test': [],
                           'Validation-Validation': [],
                           'Validation-Test': [],
                           'Test-Test': []}
        self.runtimes = []
        self.ids = []

    def prepare(self, graph_params):
        super().prepare(graph_params)
        self.gs_data_dir = SEP.join([self.destination, "graph_saint_data", graph_params.name])
        self.gs_config_file = SEP.join([self.destination, "graph_saint_data", self.data_name+'.yml'])
        self.categories_image = SEP.join([self.destination, 'categories.png'])
        self.state_file = SEP.join([self.results_dir, 'state.pkl'])
        self.loss_auc_image = SEP.join([self.results_dir, 'loss_and_auc.png'])

        default_task_params = {'transfer_model': False, 'mode': None, 'nni': False, 'train_params': {}}
        default_task_params.update(self.task_params)
        self.task_params = default_task_params

    def run_on_snapshots(self, graphs_params):
        graphs = [graph_params.get_graph() for graph_params in graphs_params]
        if self.task_params['mode'] == 'multi_train':
            remove_test_edges_from_train(graphs)
        for graph, graph_params in zip(graphs, graphs_params):
            graph_params.graph = graph
        # plot_disappearing_edges(graphs_params)
        prepare_graphs(graphs_params, edge_role=(str(self) == 'line'), transform=True)
        for graph_params in graphs_params:
            self.run(graph_params)


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
                   '--train_config', self.gs_config_file, '--gpu', self.task_params['gpu'], '--dir_log', self.results_dir,
                   '--loss_type', self.loss_type]
        # command = [python_path, 'GraphSAINT/graphsaint/pytorch_version/train.py', '--data_prefix', self.gs_data_dir,
        #            '--train_config', "GraphSAINT/train_config/table2/test.yml", '--gpu', gpu, '--dir_log', self.results_dir]
        if self.saved_model_path:
            with open(self.saved_model_path, 'rb') as file:
                args_global, timestamp, auc_dict, history, _ = pickle.load(file)
                state_dict_path = '{}/pytorch_models/saved_model.pkl'.format(args_global.dir_log)
                command += ['--saved_model_path', state_dict_path]
        if self.loss_action is not None:
            command += ['--loss_action', self.loss_action]
        # print(' '.join(command))
        # LineTask.gs_args = command[2:]
        # move this to top
        from GraphSAINT.graphsaint.pytorch_version.train import start_gs_train
        start_gs_train(gs_args=command[2:], graph_name=self.name)

        # p = Popen(command, stderr=subprocess.PIPE)
        # output, err = p.communicate()
        # p.wait()
        # if p.returncode != 0:
        #     raise Exception(err)

    def load_gs_results(self):
        with open(self.state_file, 'rb') as file:
            args_global, timestamp, auc_dict, history, preds = pickle.load(file)
        # return self.open_gs_model(args_global, timestamp), auc_dict, history
        return None, auc_dict, history, preds

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

    def create_gs_config(self, train_params=None):
        mode = self.graph_params.mode
        if not train_params:
            train_params = self.task_params['train_params']
        task_params_to_file = {key: value for key, value in self.task_params.items() if key not in ['train_params']}
        default = {
            'network': {
                'dim': 128,
                'aggr': 'concat',
                'loss': 'softmax',
                # 'loss_type': 'node',
                'arch': '1-0-1-0',
                'act': 'relu',
                'bias': 'norm'},
            'params': {
                'lr': 0.01,
                'dropout': 0.2,
                'weight_decay': 0.,
                'sample_coverage': 50,
                'norm_loss': True,
                'norm_aggr': True,
                'q_threshold': 50,
                'q_offset': 0},
            'phase': {
                'end': 1000,
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
        if str(self) == 'mul': #self.loss_type == 'edge':
            train_params['network'][0]['loss'] = 'sigmoid'#'sigmoid' 'softmax'
        with open(self.gs_config_file, 'w') as file:
            yaml.dump(train_params, file, default_flow_style=False)


    def update_scores(self, auc_dict, categories_auc):
        mode = self.graph_params.mode
        if mode == 'train':
            self.scores['train'].append(auc_dict['train'])
            self.scores['val'].append(auc_dict['val'])
        elif mode == 'test':
            self.scores['test'].append(auc_dict['test'])
        else:
            assert mode in ['train_test', 'const_train_test']
            self.scores['train'].append(auc_dict['train'])
            self.scores['val'].append(auc_dict['val'])
            self.scores['test'].append(auc_dict['test'])
        for key, auc in categories_auc.items():
            self.categories[key].append(auc)


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

    def get_final_scores(self, avg_scores):
        return {role: round(sum(scores) / len(scores), 3) for role, scores in avg_scores.items()}

    def eval_categories(self, preds, graph_params):
        labels = np.array(graph_params.get_labels())
        categories = self.get_categories(graph_params.role, graph_params)
        categories_auc = {}
        for key, indices in categories.items():
            categories_auc[key] = 0.5
            if len(indices):
                fpr, tpr, thresholds = metrics.roc_curve(labels[indices], preds[indices], pos_label=1)
                categories_auc[key] = round(metrics.auc(fpr, tpr), 3)
                categories_auc[key] = 0.5 if math.isnan(categories_auc[key]) else categories_auc[key]

        return categories_auc
        # plot_categories_auc(categories_auc)

    def get_categories(self, role, graph_params):
        edges = graph_params.get_graph(transform=True).edges
        categories = {'Train-Train': [],
                      'Train-Validation': [],
                      'Train-Test': [],
                      'Validation-Validation': [],
                      'Validation-Test': [],
                      'Test-Test': []}
        for i, (a, b) in enumerate(edges):
            if a in role['tr'] and b in role['tr']:
                categories['Train-Train'].append(i)
            elif (a in role['tr'] and b in role['va']) or (b in role['tr'] and a in role['va']):
                categories['Train-Validation'].append(i)
            elif (a in role['tr'] and b in role['te']) or (b in role['tr'] and a in role['te']):
                categories['Train-Test'].append(i)
            elif a in role['va'] and b in role['va']:
                categories['Validation-Validation'].append(i)
            elif (a in role['va'] and b in role['te']) or (b in role['va'] and a in role['te']):
                categories['Validation-Test'].append(i)
            else:
                assert a in role['te'] and b in role['te']
                categories['Test-Test'].append(i)
        return categories



class LineTask(GSTask):

    def run(self, graph_params):
        if not graph_params.future or graph_params.mode != graph_params.future.mode:
            return
        start = time.time()
        print("task", str(self), 'graph', graph_params.name)
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prepare(graph_params)
        graph = graph_params.get_graph()

        print('current graph:', len(graph.nodes), 'nodes,', len(graph.edges), 'edges')
        # future_graph = graph_params.future.get_graph()
        # just to check edge sharing: y = [1 if edge in future_graph.edges else 0 for edge in graph.edges]
        line_graph = graph_params.get_graph(line=True)
        line_future_graph = graph_params.future.get_graph(line=True)
        # Toy.set_graphs(graph, future_graph, line_graph, line_future_graph)
        Toy.draw_graph('Graph of '+graph_params.name)
        Toy.draw_line_graph('Line Graph of '+graph_params.name)

        print('current line graph:', len(line_graph.nodes), 'nodes,', len(line_graph.edges), 'edges')

        force_create_feats = self.task_params['create_feats']
        if not graph_params.feats or force_create_feats:
            print("Didn't find any line_features_{}.pkl file for graph " + graph_params.name + '. Creating: (force is '+str(force_create_feats)+')')
            create_feats_to_pkl(line_graph, graph_params, line=True)
        # for_testing(line_graph)
        graph_feats, self.feats_names = graph_params.get_feats()
        # graph_feats = self.filter_features()

        # plot_features_histogram(graph_feats, self.results_dir, feats_names)
        print("This graph mode is:", graph_params.mode)
        # all_auc_dict = {'train': [], 'val': [], 'test': []}
        labels = graph_params.get_labels()
        convert(line_graph, self.gs_data_dir, labels, future_graph=line_future_graph, feats=graph_feats, mode=graph_params.mode, test_seed=0, line=True, role=graph_params.edge_role)
        self.create_gs_config()
        self.run_gs()
        model, auc_dict, history, preds = self.load_gs_results()
        # TODO doesnt work when on gpu, fix.
        if graph_params.mode != 'test' and not torch.cuda.is_available():
            plot_loss_and_auc(history['train_loss'], history['train_auc'], history['val_loss'], history['val_auc'], self.loss_auc_image, self.data_name, graph_params.name, task=str(self))
            categories_auc = self.eval_categories(preds, graph_params)
        # all_auc_dict = {data_role: auc_list + [auc_dict[data_role]] for data_role, auc_list in all_auc_dict.items()}
        # auc_dict = {data_role: round(np.average(auc_list), 3) for data_role, auc_list in all_auc_dict.items()}


        if self.task_params['transfer_model']:
            self.saved_model_path = self.state_file
        # raise Exception()
        # self.runtimes.append(time.time() - start)
        print("auc results:", auc_dict)
        self.update_scores(auc_dict, categories_auc)
        # self.scores = {key: key_scores + [auc_dict[key]] for key, key_scores in self.scores.items()}
    def __str__(self):
        return 'line'

class MulTask(GSTask):
    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        self.loss_type = 'edge'
        self.loss_action = 'mul'

    def __str__(self):
        return 'mul'

class ConcatTask(GSTask):
    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        self.loss_type = 'edge'
        self.loss_action = 'cat'

    def __str__(self):
        return 'concat'

class EmbedTask(Task):

    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        # self.saved_model_path = ''
        # self.gs_data_dir = None
        # self.gs_config_file = None
        # self.state_file = None
        # self.loss_type = 'node'
        self.fodge_data_dir = None
        self.graphs_params = None
        self.state_file = None

    def run_on_snapshots(self, graphs_params):
        self.run(graphs_params)

    def prepare(self, graphs_params):
        self.data_name = graphs_params[0].data_name
        self.destination = SEP.join([self.root, "task_" + str(self), self.data_name])
        self.results_dir = SEP.join([self.destination, "results"])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.fodge_data_dir = SEP.join([self.destination, 'fodge_data'])
        if not os.path.exists(self.fodge_data_dir):
            os.makedirs(self.fodge_data_dir)
        self.graphs_params = graphs_params
        self.state_file = SEP.join([self.results_dir, 'state.pkl'])
        self.ids = [graph_params.id for graph_params in graphs_params]

    def run(self, graphs_params):
        start = time.time()
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prepare(graphs_params)

        # plot_edges_over_snapshots(graphs_params, path=SEP.join([self.results_dir, 'edges_labels.png']))
        # plot_disappearing_edges(graphs_params, path=SEP.join([self.results_dir, 'disappearing_edges.png']))
        self.create_fodge_config()
        self.run_fodge()
        embedding, all_embeddings = self.load_fodge_results()
        self.eval_fodge(all_embeddings)
        # TODO doesnt work when on gpu, fix.
        # if graph_params.mode != 'test' and not torch.cuda.is_available():
        #     plot_loss_and_auc(history['train_loss'], history['train_auc'], history['val_loss'], history['val_auc'],
        #                       self.loss_auc_image, self.data_name, graph_params.name)
        # all_auc_dict = {data_role: auc_list + [auc_dict[data_role]] for data_role, auc_list in all_auc_dict.items()}
        # auc_dict = {data_role: round(np.average(auc_list), 3) for data_role, auc_list in all_auc_dict.items()}

        # if self.task_params['transfer_model']:
        #     self.saved_model_path = self.state_file
        # raise Exception()
        # self.runtimes.append(time.time() - start)
        # print("auc results:", auc_dict)
        # self.update_scores(auc_dict)
        # self.scores = {key: key_scores + [auc_dict[key]] for key, key_scores in self.scores.items()}

    def create_fodge_config(self):
        file_name = SEP.join([self.fodge_data_dir, self.data_name+'.txt'])
        with open(file_name, 'w+', newline='') as file:
            wr = csv.writer(file)
            for graph_params in self.graphs_params:
                graph = graph_params.get_graph()
                for edge in graph.edges:
                    wr.writerow([edge[0], edge[1], graph_params.id])

    def run_fodge(self):
        from FODGE.main import start_fodge
        args = ['--name', self.data_name,
                '--datasets_path', self.fodge_data_dir,
                '--save_path', self.results_dir,
                '--initial_method', 'node2vec',
                '--dim', '128',
                '--epsilon', '0.04',
                '--alpha', '0.2',
                '--beta', '0.7',
                '--number', '50']
        start_fodge(args)

    def load_fodge_results(self):
        from FODGE.fodge.load_data import load_embedding
        with open(self.state_file, 'rb') as file:
            file_names = pickle.load(file)[0]
        embedding = load_embedding(self.results_dir, file_names[0])
        all_embeddings = load_embedding(self.results_dir, file_names[1])
        # return self.open_gs_model(args_global, timestamp), auc_dict, history
        return embedding, all_embeddings

    def eval_fodge(self, all_embeddings):
        eval_future = self.task_params.get('eval_future', False)
        all_embeddings = list(all_embeddings.values())
        all_embeddings = [{int(key): value for key, value in embeddings.items()} for embeddings in all_embeddings]
        graphs = [graph_params.get_graph() for graph_params in self.graphs_params]
        auc = []
        num_of_eval = len(self.graphs_params)-2 if eval_future else len(self.graphs_params)-1
        assert num_of_eval >= 1
        for i in range(num_of_eval):
            embedding1, embedding2 = all_embeddings[i], all_embeddings[i+1]
            if eval_future:
                edges = list(set(graphs[i].edges).intersection(set(graphs[i+1].edges)))
                eval_graph = self.graphs_params[i+1]
            else:
                edges = graphs[i].edges
                eval_graph = self.graphs_params[i]
            preds = self.predict(edges, embedding1, embedding2)
            labels = eval_graph.get_labels(edges=edges)
            # print(sum(labels), sum(preds))
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
            auc.append(round(metrics.auc(fpr, tpr), 3))
            # f1.append([metrics.f1_score(labels, preds, average="micro"), metrics.f1_score(labels, preds, average="macro")])
        print(auc)
        self.scores = auc



    def predict(self, edges, embedding, future_embedding):
        edges = [(int(edge[0]), int(edge[1])) for edge in edges]
        distances = [np.linalg.norm(embedding[edge[0]]-embedding[edge[1]]) for edge in edges]
        future_distances = [np.linalg.norm(future_embedding[edge[0]] - future_embedding[edge[1]]) for edge in edges]
        print(distances)
        print(future_distances)
        # preds = [1 if dis >= future_dis else 0 for dis, future_dis in zip(distances, future_distances)]
        preds = [dis - future_dis for dis, future_dis in zip(distances, future_distances)]
        return preds



    def __str__(self):
        return 'embed'

    # def apply_node2vec(self, graph):
    #     """
    #     Apply Node2Vec embedding
    #     """
    #     params_dict = {"dimension": 128, "walk_length": 80, "num_walks": 16, "workers": 2}
    #     node2vec = Node2Vec(graph, **params_dict)
    #     model = node2vec.fit()
    #     nodes = list(graph.nodes())
    #     my_dict = {}
    #     for node in nodes:
    #         my_dict.update({node: np.asarray(model.wv.get_vector(node))})
    #     X = np.zeros((len(nodes), params_dict['dimension']))
    #     for i in range(len(nodes)):
    #         X[i, :] = np.asarray(model.wv.get_vector(nodes[i]))
    #     # X is the embedding matrix and projections are the embedding dictionary
    #     return X, my_dict
    # pass





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
