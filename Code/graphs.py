import os
import random

import networkx as nx
import pickle
import numpy as np
import pandas as pd
# from gae.preprocessing import mask_test_edges

SEP = '/'

class Params:
    def __init__(self, data_name, name, id, size=None, p=None, rank=None, path=None, pkl=None, mode=None, future=None):
        self.data_name = data_name
        self.name = name
        self.id = id
        self.size = size
        self.p = p
        self.rank = rank
        self.path = path
        self.pkl = pkl
        self.mode = mode
        self.future = future
        if not (path or size and (p or rank)):
            raise Exception("Not enough params")
        self.feats = None
        self.edge_feats = None
        self.graph = None
        self.total_nodes = None
        self.role = None
        self.edge_role = None

    def get_graph(self, transform=False, line=False):
        if self.graph is None:
            self.graph = graph_from_pkl(self.path)
            self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
            self.graph = self.get_gcc(self.graph)
        if line:
            line_graph = nx.line_graph(self.graph)
            return nx.convert_node_labels_to_integers(line_graph, label_attribute='original')
        elif transform:
            return nx.convert_node_labels_to_integers(self.graph, label_attribute='original')
        return self.graph
        # if self.pkl:
        #     return graph_from_pkl(self.path)
        # if self.path:
        #     return graph_from_path()
        # if self.rank is not None:
        #     self.p = self.rank / (self.size - 1)
        # return random_graph(self.size, self.p)

    # def get_role(self, transform=False):
    #
    #     # if transform:
    #     return self.role
    def get_gcc(self, graph):
        subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
        if len(subgraphs) > 1:
            gcc = max(subgraphs, key=len)
            print(f'Changed from graph: edges={len(graph.edges)} to gcc: edges={len(gcc.edges)}')
            return gcc
        return graph

    def get_labels(self, edges=None, dict_output=False):
        if edges is None:
            edges = self.get_graph().edges
        if dict_output:
            y = {tuple(sorted(edge)): 1 if edge in self.future.get_graph().edges else 0 for edge in edges}
            ratio = round(sum(y.values()) / len(y.values()), 3)
        else:
            y = [1 if edge in self.future.get_graph().edges else 0 for edge in edges]
            ratio = round(sum(y) / len(y), 3)
        # print(ratio, "of the data has positive label (how many links are still connected in next snapshot)")
        if ratio in [0, 1]:
            print('WARNING!!!! Found only 1 class...')
        return y


    def get_feats(self, type='node'):
        if type == 'node' and self.feats is not None:
            return feats_from_pkl(self.feats)
        elif type == 'edge' and self.edge_feats is not None:
            return feats_from_pkl(self.edge_feats)
        return None, []



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
    print("Removing all test edges from train snapshots")
    test_graph = list_of_graphs[-2]
    for graph in list_of_graphs[:-2]:
        for edge in test_graph.edges():
            if edge in graph.edges():
                graph.remove_edge(edge[0], edge[1])

def prepare_graphs(graphs_params, edge_role=False, transform=False, test_seed=0):
    # if line:
        # total_nodes = len(set([i for graph_params in graphs_params for i in graph_params.get_graph().edges]))
    const_roles = create_role(graphs_params, test_seed=test_seed)

    for graph_params in graphs_params:

        if graph_params.mode == 'const_train_test':
            roles = const_roles
        else:
            roles = create_role([graph_params], test_seed=test_seed)
        graph_params.role = transform_role(roles[0], graph_params, is_edge_role=False, transform=transform)
        if edge_role:
            graph_params.edge_role = transform_role(roles[1], graph_params, is_edge_role=True, transform=transform)
        # graph_params.total_nodes = total_nodes

def create_role(graphs_params, test_seed=0, equalify_test=False):
    # if line:
    #     edges = np.array(list(set([edge for graph_params in graphs_params for edge in graph_params.get_graph(line=line).nodes])))
    #     {graph.nodes[i]['original'] for i in range(len(graph.nodes))}
    # else:
    # edges = np.array(list(set([tuple(sorted(edge)) for graph_params in graphs_params for edge in graph_params.get_graph().edges])))
    nodes = [node for graph_params in graphs_params for node in graph_params.get_graph().nodes]
    nodes = np.array(list(dict.fromkeys(nodes)))
    # nodes1 = np.array(list(set([node for graph_params in graphs_params for node in graph_params.get_graph().nodes])))
    # ^ PROBLEM HERE!! set() is changing order of nodes
    if len(graphs_params) != 1:
        train_size = 0.6
        val_size = 0.2
    else:
        mode = graphs_params[0].mode
        if mode == 'train_test':
            train_size = 0.6
            val_size = 0.2
        elif mode == 'train':
            train_size = 0.8
            val_size = 0.2
        elif mode == 'test':
            train_size = 0.00
            val_size = 0.00
        else:
            raise Exception("mode has to be train_test/train/test")

    size = len(nodes)
    tr_idx, va_idx = int(train_size * size), int((train_size + val_size) * size)
    indices = shuffle_indices(size, va_idx, test_seed, equalify_test)
    assert len(indices) == len(nodes)

    role = {}
    role['tr'] = nodes[indices[:tr_idx]]
    role['va'] = nodes[indices[tr_idx:va_idx]]
    role['te'] = nodes[indices[va_idx:]]
    edge_role = get_edge_role(role, graphs_params)
    return role, edge_role

def shuffle_indices(size, va_idx, test_seed, equalify_test, labels=None):
    # note: labels is for equalify_test which is not yet implemented...
    indices = list(range(size))
    random.seed(test_seed)
    # if
    random.shuffle(indices)
    train_val_indices = indices[:va_idx]
    # note: There is no need in seed for train_val, even if we have const_train_val=True
    random.seed()
    random.shuffle(train_val_indices)
    indices[:va_idx] = train_val_indices
    return indices

def transform_role(role, graph_params, is_edge_role=True, transform=False):
    graph = graph_params.get_graph()
    role = remove_missing(role, graph, remove_edges=is_edge_role)
    if transform:
        graph = graph_params.get_graph(transform=True, line=is_edge_role)
        # node_to_transform = {node: node_transformed for node, node_transformed in zip(graph.nodes, graph_transformed.nodes)}
        node_to_transform = {graph.nodes[i]['original']: list(graph.nodes)[i] for i in range(len(graph.nodes))}
        transformed_role = {key: [node_to_transform[node] for node in nodes] for key, nodes in role.items()}
        return transformed_role

    # Example:
    # Max of "role" can be more then the length of nodes, calc with: max([max(val, key=lambda a: int(a)) for k, val in role.items()])
    # Max of "transformed_role" will always be as the number of nodes: max([max(val) for k, val in transformed_role.items()])
    # role = ["1","3","4"],["6","7","8"]  ==>  transformed_role = [1,2,3],[4,5,6]
    return role

def remove_missing(role, graph, remove_edges=True):
    # this function removes every item from role that is not in graph_items
    new_role = {'tr': [], 'va': [], 'te': []}
    graph_items = graph.edges if remove_edges else graph.nodes
    for key, items in role.items():
        # item can be node or edge, it depends if line or not
        for item in items:
            if item in graph_items:
                new_role[key].append(item)
    return new_role

def get_edge_role(role, graphs_params, transform=False):
    edge_role = {'tr': [], 'va': [], 'te': []}
    edges = set([tuple(sorted(edge)) for graph_params in graphs_params for edge in graph_params.get_graph(transform=transform).edges])
    for a, b in edges:
        if a in role['te'] or b in role['te']:
            edge_role['te'].append((a, b))
        elif a in role['va'] or b in role['va']:
            edge_role['va'].append((a, b))
        else:
            assert a in role['tr'] or b in role['tr']
            edge_role['tr'].append((a, b))
    return edge_role

def create_graphs_from_raw(raw_dir, master_graph_path, subgraph_path, indices, data_name, delimiter=','):
    # important note: graph with index 18 is graphs 1,...,8 combined !!!
    MasterGraph = nx.read_edgelist(SEP.join([raw_dir, master_graph_path]), nodetype=int, delimiter=",")
    for edge in MasterGraph.edges():
        MasterGraph[edge[0]][edge[1]]['weight'] = 1

    print(MasterGraph.number_of_nodes())
    print(MasterGraph.number_of_edges())
    list_of_graphs = []
    for index in indices:
        graph = nx.read_edgelist(SEP.join([raw_dir, f"{subgraph_path}{index}.csv"]), nodetype=int, delimiter=",")
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = 1
        list_of_graphs.append(graph)

    data_dir = SEP.join(['Data', data_name])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = list(range(1, len(indices)+1))
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', data_name, f'graph_{id}.pkl'])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

def create_graphs_from_raw_enron(raw_dir='RawData/Enron-employees'):
    create_graphs_from_raw(raw_dir=raw_dir,
                           master_graph_path='m_enron_employees.csv',
                           subgraph_path='m_enron_employees_',
                           indices=[1, 2, 3, 4, 5, 6, 7, 8],
                           data_name='enron')


def create_graphs_from_raw_radoslaw(raw_dir='RawData/radoslaw-email'):
    create_graphs_from_raw(raw_dir=raw_dir,
                           master_graph_path='radoslaw.csv',
                           subgraph_path='radoslaw_m',
                           indices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                           data_name='radoslaw')


def create_graphs_from_raw_facebook(raw_dir='RawData/fb-forum'):
    create_graphs_from_raw(raw_dir=raw_dir,
                           master_graph_path='fb-forum-m.csv',
                           subgraph_path='fb-forum-m',
                           indices=[1, 2, 3, 4, 5, 6],
                           data_name='facebook')


def create_graphs_from_raw_reality(raw_dir='RawData/Reality-call'):
    create_graphs_from_raw(raw_dir=raw_dir,
                           master_graph_path='reality_call.csv',
                           subgraph_path='reality_call_t',
                           indices=[1, 2, 3, 4, 5, 6, 7],
                           data_name='reality')


def create_graphs_from_raw_dublin(raw_dir='RawData/contacts_dublin'):
    create_graphs_from_raw(raw_dir=raw_dir,
                           master_graph_path='dublin.csv',
                           subgraph_path='dublin_w',
                           indices=[1, 2, 3, 4, 5, 6, 7, 8],
                           data_name='dublin')

def create_graphs_from_raw_haggle(raw_dir='RawData/Haggle'):
    file_path = raw_dir + '/haggle.contact'
    split_str = ' '
    data_name = 'haggle'
    time_index = 3
    training_frac = 0.9
    edges = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if '%' in line:
                continue
            edge = line.strip().replace('\t', ' ').split(split_str)
            if edge[time_index] == '\\N':
                raise Exception("No timestamps...")
            else:
                edges.append([edge[time_index], edge[0], edge[1]])

    sorted_edges = sorted(edges, key=lambda edge: int(edge[0]))

    G = nx.Graph()
    for e in sorted_edges:
        G.add_edge(e[1], e[2])
    print(nx.number_of_nodes(G), nx.number_of_edges(G))

    size = len(sorted_edges)
    train_edges = sorted_edges[:int(size * training_frac)]
    test_edges = sorted_edges[int(size * training_frac):]


    G_train = nx.Graph()
    for e in train_edges:
        G_train.add_edge(e[1], e[2])

    G_test = nx.Graph()
    for e in test_edges:
        if e[1] in G_train.nodes() and e[2] in G_train.nodes():
            G_test.add_edge(e[1], e[2])
    print('G_train:', nx.number_of_nodes(G_train), nx.number_of_edges(G_train))
    print('G_test:', nx.number_of_nodes(G_test), nx.number_of_edges(G_test))
    G_train.remove_edges_from(nx.selfloop_edges(G_train))
    G_test.remove_edges_from(nx.selfloop_edges(G_test))
    print("after remove selfloops:")
    print('G_train:', nx.number_of_nodes(G_train), nx.number_of_edges(G_train))
    print('G_test:', nx.number_of_nodes(G_test), nx.number_of_edges(G_test))
    list_of_graphs = [G_train, G_test]

    data_dir = SEP.join(['Data', data_name])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', data_name, f'graph_{id}.pkl'])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)



def create_graphs_from_raw_catalano(raw_file='RawData/CELL CALLS/CellPhoneCallRecords.csv'):
    rows = pd.read_csv(raw_file)
    edges_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
    for from_idx, to_idx, datetime in zip(rows['From'], rows['To'], rows['Datetime']):
        key = (int(datetime.split(' ')[0][-2:]) - 1)//2
        edges_dict[key].append((from_idx, to_idx))

    list_of_graphs = [nx.Graph(edges) for key, edges in edges_dict.items()]
    # remove_test_edges_from_train(list_of_graphs)

    data_name = 'catalano'
    data_dir = SEP.join(['Data', data_name])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ids = [1, 2, 3, 4, 5]
    for g, id in zip(list_of_graphs, ids):
        file_path = SEP.join(['Data', data_name, 'graph_{}.pkl'.format(id)])
        with open(file_path, 'wb') as file:
            pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)

# def create_graphs_from_raw_enron(raw_dir):
#     MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees.csv"]), nodetype=int, delimiter=",")
#     for edge in MasterGraph.edges():
#         MasterGraph[edge[0]][edge[1]]['weight'] = 1
#
#     print(MasterGraph.number_of_nodes())
#     print(MasterGraph.number_of_edges())
#
#     G1 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_1.csv"]), nodetype=int, delimiter=",")
#     for edge in G1.edges():
#         G1[edge[0]][edge[1]]['weight'] = 1
#     G2 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_2.csv"]), nodetype=int, delimiter=",")
#     for edge in G2.edges():
#         G2[edge[0]][edge[1]]['weight'] = 1
#     G3 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_3.csv"]), nodetype=int, delimiter=",")
#     for edge in G3.edges():
#         G3[edge[0]][edge[1]]['weight'] = 1
#     G4 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_4.csv"]), nodetype=int, delimiter=",")
#     for edge in G4.edges():
#         G4[edge[0]][edge[1]]['weight'] = 1
#     G5 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_5.csv"]), nodetype=int, delimiter=",")
#     for edge in G5.edges():
#         G5[edge[0]][edge[1]]['weight'] = 1
#     G6 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_6.csv"]), nodetype=int, delimiter=",")
#     for edge in G6.edges():
#         G6[edge[0]][edge[1]]['weight'] = 1
#     G7 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_7.csv"]), nodetype=int, delimiter=",")
#     for edge in G7.edges():
#         G7[edge[0]][edge[1]]['weight'] = 1
#     G8 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_8.csv"]), nodetype=int, delimiter=",")
#     for edge in G8.edges():
#         G8[edge[0]][edge[1]]['weight'] = 1
#
#     G17 = nx.read_edgelist(SEP.join([raw_dir, "m_enron_employees_17.csv"]), nodetype=int, delimiter=",")
#     for edge in G17.edges():
#         G17[edge[0]][edge[1]]['weight'] = 1
#
#
#
#     # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
#     # test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.0, prevent_disconnect=True)
#     list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G17]
#     # remove_test_edges_from_train(list_of_graphs)
#
#     data_dir = SEP.join(['Data', 'enron'])
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#     for g, id in zip(list_of_graphs, ids):
#         file_path = SEP.join(['Data', 'enron', 'graph_{}.pkl'.format(id)])
#         with open(file_path, 'wb') as file:
#             pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)
#
# def create_graphs_from_raw_radoslaw(raw_dir):
#     MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "radoslaw.csv"]), nodetype=int, delimiter=",")
#     for edge in MasterGraph.edges():
#         MasterGraph[edge[0]][edge[1]]['weight'] = 1
#
#     print(MasterGraph.number_of_nodes())
#     print(MasterGraph.number_of_edges())
#
#     G1 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m1.csv"]), nodetype=int, delimiter=",")
#     for edge in G1.edges():
#         G1[edge[0]][edge[1]]['weight'] = 1
#     G2 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m2.csv"]), nodetype=int, delimiter=",")
#     for edge in G2.edges():
#         G2[edge[0]][edge[1]]['weight'] = 1
#     G3 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m3.csv"]), nodetype=int, delimiter=",")
#     for edge in G3.edges():
#         G3[edge[0]][edge[1]]['weight'] = 1
#     G4 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m4.csv"]), nodetype=int, delimiter=",")
#     for edge in G4.edges():
#         G4[edge[0]][edge[1]]['weight'] = 1
#     G5 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m5.csv"]), nodetype=int, delimiter=",")
#     for edge in G5.edges():
#         G5[edge[0]][edge[1]]['weight'] = 1
#     G6 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m6.csv"]), nodetype=int, delimiter=",")
#     for edge in G6.edges():
#         G6[edge[0]][edge[1]]['weight'] = 1
#     G7 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m7.csv"]), nodetype=int, delimiter=",")
#     for edge in G7.edges():
#         G7[edge[0]][edge[1]]['weight'] = 1
#     G8 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m8.csv"]), nodetype=int, delimiter=",")
#     for edge in G8.edges():
#         G8[edge[0]][edge[1]]['weight'] = 1
#     G9 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m9.csv"]), nodetype=int, delimiter=",")
#     for edge in G9.edges():
#         G9[edge[0]][edge[1]]['weight'] = 1
#
#     G18 = nx.read_edgelist(SEP.join([raw_dir, "radoslaw_m18.csv"]), nodetype=int, delimiter=",")
#     for edge in G18.edges():
#         G18[edge[0]][edge[1]]['weight'] = 1
#
#     list_of_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G18]
#     # remove_test_edges_from_train(list_of_graphs)
#
#     data_dir = SEP.join(['Data', 'radoslaw'])
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     for g, id in zip(list_of_graphs, ids):
#         file_path = SEP.join(['Data', 'radoslaw', 'graph_{}.pkl'.format(id)])
#         with open(file_path, 'wb') as file:
#             pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)
#
# def create_graphs_from_raw_facebook(raw_dir):
#     MasterGraph = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m.csv"]), nodetype=int, delimiter=",")
#     for edge in MasterGraph.edges():
#         MasterGraph[edge[0]][edge[1]]['weight'] = 1
#
#     print(MasterGraph.number_of_nodes())
#     print(MasterGraph.number_of_edges())
#
#     G1 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m1.csv"]), nodetype=int, delimiter=",")
#     for edge in G1.edges():
#         G1[edge[0]][edge[1]]['weight'] = 1
#     G2 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m2.csv"]), nodetype=int, delimiter=",")
#     for edge in G2.edges():
#         G2[edge[0]][edge[1]]['weight'] = 1
#     G3 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m3.csv"]), nodetype=int, delimiter=",")
#     for edge in G3.edges():
#         G3[edge[0]][edge[1]]['weight'] = 1
#     G4 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m4.csv"]), nodetype=int, delimiter=",")
#     for edge in G4.edges():
#         G4[edge[0]][edge[1]]['weight'] = 1
#     G5 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m5.csv"]), nodetype=int, delimiter=",")
#     for edge in G5.edges():
#         G5[edge[0]][edge[1]]['weight'] = 1
#     G6 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m6.csv"]), nodetype=int, delimiter=",")
#     for edge in G6.edges():
#         G6[edge[0]][edge[1]]['weight'] = 1
#     G15 = nx.read_edgelist(SEP.join([raw_dir, "fb-forum-m15.csv"]), nodetype=int, delimiter=",")
#     for edge in G15.edges():
#         G15[edge[0]][edge[1]]['weight'] = 1
#
#     list_of_graphs = [G1, G2, G3, G4, G5, G6, G15]
#     # remove_test_edges_from_train(list_of_graphs)
#
#     data_dir = SEP.join(['Data', 'facebook'])
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     ids = [1, 2, 3, 4, 5, 6, 7]
#     for g, id in zip(list_of_graphs, ids):
#         file_path = SEP.join(['Data', 'facebook', 'graph_{}.pkl'.format(id)])
#         with open(file_path, 'wb') as file:
#             pickle.dump(g, file, protocol=pickle.HIGHEST_PROTOCOL)
#

#
# def create_graphs_from_raw_reality(raw_dir):
#     MasterGraph = nx.read_edgelist("Reality-call/reality_call.csv", nodetype=int, delimiter=",")
#     for edge in MasterGraph.edges():
#         MasterGraph[edge[0]][edge[1]]['weight'] = 1
#
#     print(MasterGraph.number_of_nodes())
#     print(MasterGraph.number_of_edges())
#
#     G1 = nx.read_edgelist(SEP.join([raw_dir, "reality_call_t1.csv"]), nodetype=int, delimiter=",")
#     for edge in G1.edges():
#         G1[edge[0]][edge[1]]['weight'] = 1
#     G2 = nx.read_edgelist("Reality-call/reality_call_t2.csv", nodetype=int, delimiter=",")
#     for edge in G2.edges():
#         G2[edge[0]][edge[1]]['weight'] = 1
#     G3 = nx.read_edgelist("Reality-call/reality_call_t3.csv", nodetype=int, delimiter=",")
#     for edge in G3.edges():
#         G3[edge[0]][edge[1]]['weight'] = 1
#     G4 = nx.read_edgelist("Reality-call/reality_call_t4.csv", nodetype=int, delimiter=",")
#     for edge in G4.edges():
#         G4[edge[0]][edge[1]]['weight'] = 1
#     G5 = nx.read_edgelist("Reality-call/reality_call_t5.csv", nodetype=int, delimiter=",")
#     for edge in G5.edges():
#         G5[edge[0]][edge[1]]['weight'] = 1
#     G6 = nx.read_edgelist("Reality-call/reality_call_t6.csv", nodetype=int, delimiter=",")
#     for edge in G6.edges():
#         G6[edge[0]][edge[1]]['weight'] = 1
#     G7 = nx.read_edgelist("Reality-call/reality_call_t7.csv", nodetype=int, delimiter=",")
#     for edge in G7.edges():
#         G7[edge[0]][edge[1]]['weight'] = 1
#     # G8 = nx.read_edgelist("Enron-employees/m_enron_employees_8.csv", nodetype = int, delimiter = ",")
#     # for edge in G8.edges():
#     #    G8[edge[0]][edge[1]]['weight'] = 1
#
#     G16 = nx.read_edgelist("Reality-call/reality_call_t16.csv", nodetype=int, delimiter=",")
#     for edge in G16.edges():
#         G16[edge[0]][edge[1]]['weight'] = 1


if __name__ == "__main__":
    # create_graphs_from_raw_enron()
    # create_graphs_from_raw_radoslaw()
    # create_graphs_from_raw_facebook()
    # create_graphs_from_raw_catalano()
    # create_graphs_from_raw_reality()
    # create_graphs_from_raw_dublin()
    create_graphs_from_raw_haggle()

    print()

