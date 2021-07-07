import random

import networkx as nx
import pickle

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
        if self.path:
            return graph_from_path()
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




