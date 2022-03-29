import os
import pickle
import random

import networkx as nx
import matplotlib.pyplot as plt

from Code.graphs import remove_test_edges_from_train
from data.open_graph_benchmark.networkx_converter import get_line_y
from graphsaint.pytorch_version.utils import to_numpy

SEP = '/'
class Toy:
    graph = None
    future_graph = None
    line_graph = None
    line_future_graph = None
    plot = False

    @staticmethod
    def set_graphs(graph, future_graph, line_graph, line_future_graph):
        Toy.graph = graph
        Toy.future_graph = future_graph
        Toy.line_graph = line_graph
        Toy.line_future_graph = line_future_graph
        Toy.plot = True

    @staticmethod
    def get_edges_in_triangle(graph, triangle_num=1):
        edges_in_triangle = []
        for edge in graph.edges():
            first_neighbors = set(graph.adj[edge[0]])
            second_neighbors = set(graph.adj[edge[1]])
            if len(first_neighbors.intersection(second_neighbors)) >= triangle_num:
                edges_in_triangle.append(edge)
        return edges_in_triangle

    @staticmethod
    def one_iteration(graph, stay_p, new_p=None, triangle_num=2):
        edges_in_triangle = Toy.get_edges_in_triangle(graph, triangle_num)
        old_edges = [edge for edge in graph.edges() if edge in edges_in_triangle or random.random() <= stay_p]
        old_edges = old_edges[:-1] if len(old_edges) == len(graph.edges()) else old_edges
        if not new_p:
            new_p = (len(graph.edges()) - len(old_edges)) / len(list(nx.non_edges(graph)))
        new_edges = [edge for edge in nx.non_edges(graph) if random.random() <= new_p]
        new_graph = nx.Graph()
        new_graph.add_nodes_from(graph.nodes)
        new_graph.add_edges_from(old_edges + new_edges)
        print("Edges: {}, Deleted: {}, Added: {}. Current: {}".format(len(graph.edges()), len(graph.edges()) - len(old_edges), len(new_edges), len(new_graph.edges())))
        return new_graph

    @staticmethod
    def draw_graph(title, graph=None, future_graph=None):
        if not Toy.plot:
            return
        if not graph:
            graph = Toy.graph
        if not future_graph:
            future_graph = Toy.future_graph
        # nx.draw(graph)
        pos = nx.spring_layout(graph, seed=0)
        y = [1 if edge in future_graph.edges else 0 for edge in graph.edges]
        link_edges = [edge for i, edge in enumerate(graph.edges) if y[i]]
        unlink_edges = [edge for i, edge in enumerate(graph.edges) if not y[i]]
        nx.draw_networkx_nodes(set(graph.nodes) - set(nx.isolates(graph)), pos, node_size=120)
        # nx.draw_networkx_edges(graph, pos, edgelist=graph.edges)
        nx.draw_networkx_labels(graph, pos)
        labels = {'link': 'Not Unlink (' + str(len(link_edges)) + ' edges)',
                  'unlink': 'Unlink (' + str(len(unlink_edges)) + ' edges)'}
        nx.draw_networkx_edges(graph, pos, edgelist=link_edges, edge_color='g', width=2, label=labels['link'])
        nx.draw_networkx_edges(graph, pos, edgelist=unlink_edges, edge_color='r', width=2, label=labels['unlink'])
        # plt.legend(('green', 'red'), (str(len(link_edges)),str(len(unlink_edges))))
        plt.title(title)
        plt.legend()
        plt.savefig(title+'.png')
        plt.show()
        print()

    @staticmethod
    def draw_line_graph(title, line_graph=None, line_future_graph=None, node_labels=None, val_idx=None):
        if not Toy.plot:
            return
        if not line_graph:
            line_graph = Toy.line_graph
        if not line_future_graph:
            line_future_graph = Toy.line_future_graph
        y = get_line_y(line_graph, line_future_graph)
        if val_idx is None:
            val_idx = []
        pos = nx.spring_layout(line_graph, seed=0)
        train_link_nodes = [node for i, node in enumerate(line_graph.nodes) if y[i] and i not in val_idx]
        train_unlink_nodes = [node for i, node in enumerate(line_graph.nodes) if not y[i] and i not in val_idx]
        val_link_nodes = [node for i, node in enumerate(line_graph.nodes) if y[i] and i in val_idx]
        val_unlink_nodes = [node for i, node in enumerate(line_graph.nodes) if not y[i] and i in val_idx]
        labels = {'train_link': 'Train - Not Unlink (' + str(len(train_link_nodes)) + ' nodes)',
                  'train_unlink': 'Train - Unlink (' + str(len(train_unlink_nodes)) + ' nodes)',
                  'val_link': 'Validation - Not Unlink (' + str(len(val_link_nodes)) + ' nodes)',
                  'val_unlink': 'Validation - Unlink (' + str(len(val_unlink_nodes)) + ' nodes)'
                  }
        nx.draw_networkx_nodes(train_link_nodes, pos, node_color='g', node_size=200, label=labels['train_link'])
        nx.draw_networkx_nodes(train_unlink_nodes, pos, node_color='r', node_size=200, label=labels['train_unlink'])
        nx.draw_networkx_nodes(val_link_nodes, pos, node_color='g', node_size=200, label=labels['val_link'],
                               edgecolors='b')
        nx.draw_networkx_nodes(val_unlink_nodes, pos, node_color='r', node_size=200, label=labels['val_unlink'],
                               edgecolors='b')
        # nx.draw_networkx_edges(graph, pos, edgelist=graph.edges)
        if node_labels is not None:
            node_labels = to_numpy(node_labels)
            node_labels = {node: round(val, 2) for node, val in zip(line_graph.nodes, node_labels)}
            nx.draw_networkx_labels(line_graph, pos, labels=node_labels, font_weight='bold', font_size=10)
        else:
            nx.draw_networkx_labels(line_graph, pos)
        # labels = {'link': 'Not Unlink (' + str(len(link_edges)) + ' edges)',
        #           'unlink': 'Unlink (' + str(len(unlink_edges)) + ' edges)'}
        nx.draw_networkx_edges(line_graph, pos, width=1)
        # plt.legend(('green', 'red'), (str(len(link_edges)),str(len(unlink_edges))))
        plt.title(title)
        plt.legend()
        plt.savefig(title+'.png')
        plt.show()
        print()




def create_toy(n=1000, p=0.01, snapshots=10, stay_p=0.9, new_p=0.002, triangle_num=2):
    graphs = []

    graph = nx.gnp_random_graph(n=n, p=p)
    # Toy.draw_graph(graph, triangle_num)

    for i in range(snapshots):
        new_graph = Toy.one_iteration(graph, stay_p, new_p, triangle_num)
        graphs.append(new_graph)
        graph = new_graph
        # draw_graph(graph, triangle_num)
    # remove_test_edges_from_train(graphs)
    [print(str(graph)) for graph in graphs]

    data_dir = SEP.join(['Data', 'toy'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = list(range(1, snapshots+1))
    for g, id in zip(graphs, ids):
        file_path = SEP.join(['Data', 'toy', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if 'Code' not in os.listdir(os.getcwd()):
        os.chdir("..")
        if 'Code' not in os.listdir(os.getcwd()):
            raise Exception(
                "Bad pathing, use the command os.chdir() to make sure you work on UnlinkPrediction directory")
    # params = {'n': 1000,
    #           'p': 0.01,
    #           'snapshots': 10,
    #           'stay_p': 0.9,
    #           'new_p': 0.002}
    # params = {'n': 30,
    #           'p': 0.03,
    #           'snapshots': 10,
    #           'stay_p': 0.6,
    #           'new_p': 0.03}
    # params = {'n': 300,
    #           'p': 0.055,
    #           'snapshots': 10,
    #           'stay_p': 0,
    #           'new_p': None}
    params = {'n': 20,
              'p': 0.15,
              'snapshots': 10,
              'stay_p': 0,
              'new_p': 0.1,
              'triangle_num': 1}
    create_toy(**params)

