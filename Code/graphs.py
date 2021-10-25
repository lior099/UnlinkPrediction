import os
import random

import networkx as nx
import pickle
import numpy as np
# from gae.preprocessing import mask_test_edges

SEP = '/'

class Params:
    def __init__(self, name, id, size=None, p=None, rank=None, path=None, pkl=None, feats=None, future=None):
        self.name = name
        self.id = id
        self.size = size
        self.p = p
        self.rank = rank
        self.path = path
        self.pkl = pkl
        self.feats = feats
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
        return Params(self.name, self.id, self.size, self.p, self.rank, self.path, self.pkl, self.feats, future=None)

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

def graph_from_path():
    pass

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

    ## All the nodes are in MasterNodes
    MasterNodes = MasterGraph.nodes()

    for i in MasterNodes:
        if i not in G8.nodes():
            G8.add_node(i)

    adj_sparse = nx.to_scipy_sparse_matrix(G8)
    np.random.seed(0)  # make sure train-test split is consistent between notebooks

    adj_sparse = nx.to_scipy_sparse_matrix(G8)

    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    # test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.0, prevent_disconnect=True)
    data_dir = SEP.join(['Data', 'Enron'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G17]
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 17]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'Enron', 'graph_{}.pkl'.format(id)])
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

    data_dir = SEP.join(['Data', 'Radoslaw'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G18]
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 18]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', 'Radoslaw', 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # create_graphs_from_raw_enron('RawData/Enron-employees')
    create_graphs_from_raw_radoslaw('RawData/radoslaw-email')

