import csv
import os
import pickle
import sys
import time

import networkx as nx
import torch

from Code.out_sources import create_feats_to_pkl
from Code.plots import plot_features_histogram
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
        self.scores = []
        self.runtimes = []
        self.ids = []

    def prepare(self, graph_params, data_name):
        self.destination = SEP.join([self.root, "task_" + str(self), data_name])
        self.data_name = data_name
        results_destination = SEP.join([self.destination, "results"])
        self.results_dir = SEP.join([results_destination, "_".join(["task", str(self), graph_params.name, "results"])])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.ids.append(graph_params.id)
        # results_file = os.path.join(results_dir, "_".join(["task", str(self), graph_params.name, "results"]) + '.csv')
        # return results_file

    def save_attributes(self, memory):
        runtime_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "runtime.csv"])])
        memory_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "memory.csv"])])
        scores_destination = SEP.join([self.destination, "_".join(["task", str(self), self.data_name, "scores.csv"])])
        self.save_to_file([['x'] + self.ids, [self.data_name] + self.runtimes], runtime_destination)
        self.save_to_file([['x'] + self.ids, [self.data_name] + memory], memory_destination)
        self.save_to_file([['x'] + self.ids, [self.data_name] + self.scores], scores_destination)


    def save_to_file(self, lines_list, path):
        with open(path, 'w', newline='') as file:
            wr = csv.writer(file)
            for line in lines_list:
                wr.writerow(line)


class LineTask(Task):
    def run(self, graph_params, data_name):
        start = time.time()
        print("task",str(self), 'graph', graph_params.name)
        self.prepare(graph_params, data_name)
        graph = graph_params.get_graph()

        print('current graph:', len(graph.nodes), 'nodes,', len(graph.edges), 'edges')
        future_graph = graph_params.future.get_graph()
        # just to check edge sharing: y = [1 if edge in future_graph.edges else 0 for edge in graph.edges]
        line_graph = nx.line_graph(graph)
        line_future_graph = nx.line_graph(future_graph)

        line_graph = nx.convert_node_labels_to_integers(line_graph, label_attribute='edge')
        print('current line graph:', len(line_graph.nodes),'nodes,',len(line_graph.edges),'edges')
        # return


        if not graph_params.feats:
            print("Didn't find any features_{}.pkl file for graph "+graph_params.name+'. Creating:')
            create_feats_to_pkl(line_graph, graph_params)
        # for_testing(line_graph)
        graph_feats, feats_names = graph_params.get_feats(line=True)
        plot_features_histogram(graph_feats, self.results_dir, feats_names)
        gs_data_path = SEP.join([self.destination, "graph_saint_data", graph_params.name])
        convert(line_graph, gs_data_path, future_graph=line_future_graph, feats=graph_feats)

        self.run_gs(gs_data_path)
        model, auc = self.load_gs_results()
        # raise Exception()
        self.scores.append(auc)
        self.runtimes.append(time.time() - start)
        print("FINAL AUC:", auc)

    def run_gs(self, gs_data_path):
        if torch.cuda.is_available():
            # python_path = '/home/dsi/shifmal2/anaconda3/envs/py3/bin/python'
            gpu = '2'
        else:
            # python_path = 'python'
            gpu = '-1'
        python_path = sys.executable
        # python_path = '/home/dsi/shifmal2/anaconda3/envs/py3/bin/python' if torch.cuda.is_available() else 'python'
        command = [python_path, 'GraphSAINT/graphsaint/pytorch_version/train.py', '--data_prefix', gs_data_path,
                   '--train_config', 'GraphSAINT/train_config/table2/test.yml', '--gpu', gpu, '--dir_log',
                   self.results_dir]
        # print(' '.join(command))
        p = Popen(command)
        output, err = p.communicate()
        p.wait()
        if p.returncode != 0:
            raise Exception(p.returncode)

    def load_gs_results(self):
        with open(SEP.join([self.results_dir + '/state.pkl']), 'rb') as file:
            args_global, timestamp, auc = pickle.load(file)
        # return self.open_gs_model(args_global, timestamp), auc
        return None, auc

    def open_gs_model(self, args_global, timestamp):
        from GraphSAINT.graphsaint.pytorch_version.train import prepare
        from GraphSAINT.graphsaint.utils import parse_n_prepare
        state_dict_path = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, timestamp)
        train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
        model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
        state_dict = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
        model_eval.load_state_dict(state_dict)
        model_eval.eval()

        return model_eval

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
