import os
import random

import networkx as nx
import pickle
import numpy as np
import pandas as pd
# from gae.preprocessing import mask_test_edges

SEP = '/'

class Params:
    def __init__(self, data_name, name, id, size=None, p=None, rank=None, path=None, pkl=None, feats=None, mode=None, future=None):
        self.data_name = data_name
        self.name = name
        self.id = id
        self.size = size
        self.p = p
        self.rank = rank
        self.path = path
        self.pkl = pkl
        self.feats = feats
        self.mode = mode
        self.future = future
        if not (path or size and (p or rank)):
            raise Exception("Not enough params")

    def get_graph(self):
        if self.pkl:
            return graph_from_pkl(self.path)
        # if self.path:
        #     return graph_from_path()
        if self.rank is not None:
            self.p = self.rank / (self.size - 1)
        return random_graph(self.size, self.p)

    def get_feats(self, line=False):
        if line:
            return feats_from_pkl(self.feats)
        else:
            raise Exception("You shouldn't take the graph feats yet, only the line_graph feats")

    def __copy__(self):
        return Params(self.data_name, self.name, self.id, self.size, self.p, self.rank, self.path, self.pkl, self.feats, self.mode, future=None)

def random_graph(size, p):
    # edges = [[i, j, 1] for i in range(size) for j in range(size) if i != j and random.random() <= p]
    # graph = nx.DiGraph()
    # graph.add_weighted_edges_from(edges)
    graph = nx.fast_gnp_random_graph(size, p, directed=True)
    return graph

def graph_from_pkl(path):
    # with open('C:/Users/shifmal2/Downloads/for_lior/DBLP/mx_0.pkl', 'rb') as file:
    #     a = pickle.load(file)
    #     b = [1 for i in a if i.tolist()[0] != [0.0, 0.0, 0.0, 0.0]]
    with open(path, 'rb') as file:
        return pickle.load(file)


def feats_from_pkl(path):

    # with open('C:/Users/shifmal2/Downloads/for_lior/DBLP/mx_0.pkl', 'rb') as file:
    #     a = pickle.load(file)
    #     b = [1 for i in a if i.tolist()[0] != [0.0, 0.0, 0.0, 0.0]]
    with open(path, 'rb') as file:
        a = pickle.load(file)
        return a


def remove_test_edges_from_train(list_of_graphs):
    test_graph = list_of_graphs[-2]
    for graph in list_of_graphs[:-2]:
        for edge in test_graph.edges():
            if edge in graph.edges():
                graph.remove_edge(edge[0], edge[1])

def create_graphs_from_raw_enron(raw_dir):
    MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees.csv"]), nodetype=int, delimiter=",")
    for edge in MasterGraph.edges():
        MasterGraph[edge[0]][edge[1]]['weight'] = 1

    print(MasterGraph.number_of_nodes())
    print(MasterGraph.number_of_edges())

    G1 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_1.csv"]), nodetype=int, delimiter=",")
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1
    G2 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_2.csv"]), nodetype=int, delimiter=",")
    for edge in G2.edges():
        G2[edge[0]][edge[1]]['weight'] = 1
    G3 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_3.csv"]), nodetype=int, delimiter=",")
    for edge in G3.edges():
        G3[edge[0]][edge[1]]['weight'] = 1
    G4 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_4.csv"]), nodetype=int, delimiter=",")
    for edge in G4.edges():
        G4[edge[0]][edge[1]]['weight'] = 1
    G5 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_5.csv"]), nodetype=int, delimiter=",")
    for edge in G5.edges():
        G5[edge[0]][edge[1]]['weight'] = 1
    G6 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_6.csv"]), nodetype=int, delimiter=",")
    for edge in G6.edges():
        G6[edge[0]][edge[1]]['weight'] = 1
    G7 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_7.csv"]), nodetype=int, delimiter=",")
    for edge in G7.edges():
        G7[edge[0]][edge[1]]['weight'] = 1
    G8 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_8.csv"]), nodetype=int, delimiter=",")
    for edge in G8.edges():
        G8[edge[0]][edge[1]]['weight'] = 1

    G17 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_17.csv"]), nodetype=int, delimiter=",")
    for edge in G17.edges():
        G17[edge[0]][edge[1]]['weight'] = 1



    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    # test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.0, prevent_disconnect=True)
    list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G17]
    remove_test_edges_from_train(list_of_graphs)

    data_dir = SEP.join(['Data', 'enron'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'enron', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

def create_graphs_from_raw_radoslaw(raw_dir):
    MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "radoslaw.csv"]), nodetype=int, delimiter=",")
    for edge in MasterGraph.edges():
        MasterGraph[edge[0]][edge[1]]['weight'] = 1

    print(MasterGraph.number_of_nodes())
    print(MasterGraph.number_of_edges())

    G1 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m1.csv"]), nodetype=int, delimiter=",")
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1
    G2 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m2.csv"]), nodetype=int, delimiter=",")
    for edge in G2.edges():
        G2[edge[0]][edge[1]]['weight'] = 1
    G3 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m3.csv"]), nodetype=int, delimiter=",")
    for edge in G3.edges():
        G3[edge[0]][edge[1]]['weight'] = 1
    G4 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m4.csv"]), nodetype=int, delimiter=",")
    for edge in G4.edges():
        G4[edge[0]][edge[1]]['weight'] = 1
    G5 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m5.csv"]), nodetype=int, delimiter=",")
    for edge in G5.edges():
        G5[edge[0]][edge[1]]['weight'] = 1
    G6 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m6.csv"]), nodetype=int, delimiter=",")
    for edge in G6.edges():
        G6[edge[0]][edge[1]]['weight'] = 1
    G7 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m7.csv"]), nodetype=int, delimiter=",")
    for edge in G7.edges():
        G7[edge[0]][edge[1]]['weight'] = 1
    G8 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m8.csv"]), nodetype=int, delimiter=",")
    for edge in G8.edges():
        G8[edge[0]][edge[1]]['weight'] = 1
    G9 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m9.csv"]), nodetype=int, delimiter=",")
    for edge in G9.edges():
        G9[edge[0]][edge[1]]['weight'] = 1

    G18 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m18.csv"]), nodetype=int, delimiter=",")
    for edge in G18.edges():
        G18[edge[0]][edge[1]]['weight'] = 1

    list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G18]
    remove_test_edges_from_train(list_of_graphs)

    data_dir = SEP.join(['Data', 'radoslaw'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'radoslaw', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

def create_graphs_from_raw_facebook(raw_dir):
    MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m.csv"]), nodetype=int, delimiter=",")
    for edge in MasterGraph.edges():
        MasterGraph[edge[0]][edge[1]]['weight'] = 1

    print(MasterGraph.number_of_nodes())
    print(MasterGraph.number_of_edges())

    G1 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m1.csv"]), nodetype=int, delimiter=",")
    for edge in G1.edges():
        G1[edge[0]][edge[1]]['weight'] = 1
    G2 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m2.csv"]), nodetype=int, delimiter=",")
    for edge in G2.edges():
        G2[edge[0]][edge[1]]['weight'] = 1
    G3 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m3.csv"]), nodetype=int, delimiter=",")
    for edge in G3.edges():
        G3[edge[0]][edge[1]]['weight'] = 1
    G4 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m4.csv"]), nodetype=int, delimiter=",")
    for edge in G4.edges():
        G4[edge[0]][edge[1]]['weight'] = 1
    G5 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m5.csv"]), nodetype=int, delimiter=",")
    for edge in G5.edges():
        G5[edge[0]][edge[1]]['weight'] = 1
    G6 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m6.csv"]), nodetype=int, delimiter=",")
    for edge in G6.edges():
        G6[edge[0]][edge[1]]['weight'] = 1
    G15 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m15.csv"]), nodetype=int, delimiter=",")
    for edge in G15.edges():
        G15[edge[0]][edge[1]]['weight'] = 1

    list_of_graphs = [G1, G2, G3, G4, G5, G6, G15]
    remove_test_edges_from_train(list_of_graphs)

    data_dir = SEP.join(['Data', 'facebook'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2, 3, 4, 5, 6, 7]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'facebook', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

def create_graphs_from_raw_catalano(raw_file):
    rows = pd.read_csv(raw_file)
    edges_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
    for from_idx, to_idx, datetime in zip(rows['From'], rows['To'], rows['Datetime']):
        key = (int(datetime.split(' ')[0][-2:]) - 1)//2
        edges_dict[key].append((from_idx, to_idx))

    list_of_graphs = [nx.Graph(edges) for key, edges in edges_dict.items()]
    remove_test_edges_from_train(list_of_graphs)

    data_dir = SEP.join(['Data', 'catalano'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2, 3, 4, 5]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'catalano', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # create_graphs_from_raw_enron('RawData/Enron-employees')
    # create_graphs_from_raw_radoslaw('RawData/radoslaw-email')
    # create_graphs_from_raw_facebook('RawData/fb-forum')
    create_graphs_from_raw_catalano('RawData/CELL CALLS/CellPhoneCallRecords.csv')
    print()

