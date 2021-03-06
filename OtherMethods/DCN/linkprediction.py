# -*- coding: utf-8 -*-
"""Linkprediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_MT5JfWASbZecv51At0aN-L2zLY8UX68
"""

# Commented out IPython magic to ensure Python compatibility.
import networkx as nx
import random as rdm
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from random import choice
from random import seed
from random import sample

# Utilities
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
from tikzplotlib import save as tikz_save
from tikzplotlib import get_tikz_code

# # enumerate small graphs based on edges
# # 1 vertices
# G_1_0 = nx.Graph()
# G_1_0.add_nodes_from([1])
#
# # 2 vertices
# G_2_0 = nx.Graph()
# G_2_0.add_nodes_from([1,2])
# G_2_1 = nx.complement(G_2_0)
# G_2 = [G_2_0, G_2_1]
# G_2_edgeindex = [0, 1]
#
# # 3 vertices
# G_3_0 = nx.Graph()
# G_3_0.add_nodes_from([1,2,3])
# G_3_1 = nx.Graph()
# G_3_1.add_nodes_from([1,2,3])
# G_3_1.add_edge(1,2)
# G_3_2 = nx.complement(G_3_1)
# G_3_3 = nx.complement(G_3_0)
# G_3 = [G_3_0, G_3_1, G_3_2, G_3_3]
# G_3_edgeindex = [0, 1, 2, 3]
#
# # 4 vertices
# G_4_0 = nx.Graph()
# G_4_0.add_nodes_from([1,2,3,4])
# G_4_1 = nx.Graph()
# G_4_1.add_nodes_from([1,2,3,4])
# G_4_1.add_edge(1,2)
# G_4_2 = nx.Graph()
# G_4_2.add_nodes_from([1,2,3,4])
# G_4_2.add_edges_from([(1,2), (2,3)])
# G_4_3 = nx.Graph()
# G_4_3.add_nodes_from([1,2,3,4])
# G_4_3.add_edges_from([(1,2), (3,4)])
# G_4_4 = nx.Graph()
# G_4_4.add_edges_from([(1,2), (1,3), (1,4)])
# G_4_5 = nx.Graph()
# G_4_5.add_edges_from([(1,2), (2,3), (3,4)])
# G_4_6 = nx.complement(G_4_4)
# G_4_7 = nx.complement(G_4_3)
# G_4_8 = nx.complement(G_4_2)
# G_4_9 = nx.complement(G_4_1)
# G_4_10 = nx.complement(G_4_0)
# G_4 = [G_4_0, G_4_1, G_4_2, G_4_3, G_4_4, G_4_5, G_4_6, G_4_7, G_4_8, G_4_9, G_4_10]
# G_4_edgeindex = [0, 1, 2, 4, 7, 9, 10]
#
# # 5 vertices
# G_5_0 = nx.Graph()
# G_5_0.add_nodes_from([1,2,3,4,5])
# G_5_1 = nx.Graph()
# G_5_1.add_nodes_from([1,2,3,4,5])
# G_5_1.add_edge(1,2)
# G_5_2 = nx.Graph()
# G_5_2.add_nodes_from([1,2,3,4,5])
# G_5_2.add_edges_from([(1,2), (2,3)])
# G_5_3 = nx.Graph()
# G_5_3.add_nodes_from([1,2,3,4,5])
# G_5_3.add_edges_from([(1,2), (3,4)])
# G_5_4 = nx.Graph()
# G_5_4.add_nodes_from([1,2,3,4,5])
# G_5_4.add_edges_from([(1,2), (1,3), (1,4)])
# G_5_5 = nx.Graph()
# G_5_5.add_nodes_from([1,2,3,4,5])
# G_5_5.add_edges_from([(1,2), (2,3), (4,5)])
# G_5_6 = nx.Graph()
# G_5_6.add_nodes_from([1,2,3,4,5])
# G_5_6.add_edges_from([(1,2), (2,3), (3,4)])
# G_5_7 = nx.Graph()
# G_5_7.add_nodes_from([1,2,3,4,5])
# G_5_7.add_edges_from([(1,2), (2,3), (3,1)])
# G_5_8 = nx.Graph()
# G_5_8.add_edges_from([(1,2), (1,3), (1,4), (1,5)])
# G_5_9 = nx.Graph()
# G_5_9.add_nodes_from([1,2,3,4,5])
# G_5_9.add_edges_from([(1,2), (2,3), (3,4), (4,1)])
# G_5_10 = nx.Graph()
# G_5_10.add_edges_from([(1,2), (1,3), (1,4), (4,5)])
# G_5_11 = nx.Graph()
# G_5_11.add_nodes_from([1,2,3,4,5])
# G_5_11.add_edges_from([(1,2), (1,3), (1,4), (3,4)])
# G_5_12 = nx.Graph()
# G_5_12.add_edges_from([(1,2), (2,3), (3,4), (4,5)])
# G_5_13 = nx.Graph()
# G_5_13.add_edges_from([(1,2), (2,3), (3,1), (4,5)])
# G_5_14 = nx.Graph()
# G_5_14.add_edges_from([(1,2), (2,3), (3,4), (4,1), (1,5)])
# G_5_15 = nx.Graph()
# G_5_15.add_edges_from([(1,2), (2,3), (3,1), (4,1), (3,5)])
# G_5_16 = nx.Graph()
# G_5_16.add_edges_from([(1,2), (2,3), (3,1), (4,1), (1,5)])
# G_5_17 = nx.Graph()
# G_5_17.add_edges_from([(1,2), (2,3), (3,4), (4,5), (1,5)])
# G_5_18 = nx.complement(G_5_16)
# G_5_19 = nx.complement(G_5_14)
# G_5_20 = nx.complement(G_5_13)
# G_5_21 = nx.complement(G_5_12)
# G_5_22 = nx.complement(G_5_11)
# G_5_23 = nx.complement(G_5_10)
# G_5_24 = nx.complement(G_5_9)
# G_5_25 = nx.complement(G_5_8)
# G_5_26 = nx.complement(G_5_7)
# G_5_27 = nx.complement(G_5_6)
# G_5_28 = nx.complement(G_5_5)
# G_5_29 = nx.complement(G_5_4)
# G_5_30 = nx.complement(G_5_3)
# G_5_31 = nx.complement(G_5_2)
# G_5_32 = nx.complement(G_5_1)
# G_5_33 = nx.complement(G_5_0)
# G_5 = [G_5_0, G_5_1, G_5_2, G_5_3, G_5_4, G_5_5, G_5_6, G_5_7, G_5_8, G_5_9, G_5_10,
#       G_5_11, G_5_12, G_5_13, G_5_14, G_5_15, G_5_16, G_5_17, G_5_18, G_5_19, G_5_20,
#       G_5_21, G_5_22, G_5_23, G_5_24, G_5_25, G_5_26, G_5_27, G_5_28, G_5_29, G_5_30,
#       G_5_31, G_5_32, G_5_33]
# G_5_edgeindex = [0, 1, 2, 4, 8, 14, 20, 26, 30, 32, 33]
#
# G_set = [G_1_0] + G_2 + G_3 + G_4

"""LOAD DATA"""

# random walk with restart sampling
def random_walk_with_restart_sampling(und_graph, sample_percentage, restart_prob, jump_iteration = 100, sd = None):
    # set random seed
    seed(sd)
    
    # sample size round down to interger
    sample_size = int(nx.number_of_nodes(und_graph) * sample_percentage)
    
    # set starting node
    startnode = rdm.sample(und_graph.nodes(),1)[0]
    currentnode = startnode
    
    # used for jump when no new node visited in certain iteration
    restart_iteration = 0 
    last_number_of_nodes = 0
    
    # result node set and total iteration
    nodelist = set()
    total_iteration = 0
    
    while len(nodelist) < sample_size:
        # add current node
        total_iteration += 1
        
        
        nodelist.add(currentnode)
        
        # restart with certain prob
        x = rdm.random()
        if x < restart_prob:
            currentnode = startnode
        else:    
            # move a step forward
            nextnode = rdm.sample(list(und_graph[currentnode]),1)[0]
            currentnode = nextnode
        
        # find a new startnode if number of nodes in sample does not grow
        if restart_iteration < jump_iteration:
            restart_iteration += 1
        else:
            if last_number_of_nodes == len(nodelist):
                startnode = rdm.sample(und_graph.nodes(),1)[0]
                currentnode = startnode
            restart_iteration = 0
            last_number_of_nodes = len(nodelist)
    return und_graph.subgraph(nodelist) #,total_iteration

# pre-processing
def load_data(data):
    if data == 'facebook':
        path = 'fb-forum.txt'
        split_str = ','
        time_index = 2
    else:
        assert data == 'haggle'
        path = 'out.contact'
        split_str = ' '
        time_index = 3
    initial_edges = []
    timed_edges = []
    edge_index = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if '%' in line:
                continue
            edge = line.strip().replace('\t', ' ').split(split_str)
            if edge[time_index] == '\\N':
                initial_edges.append([edge[time_index], edge[0], edge[1]])
                edge_index[(edge[0], edge[1])] = 0
                edge_index[(edge[1], edge[0])] = 0
            else:
                timed_edges.append([edge[time_index], edge[0], edge[1]])
                edge_index[(edge[0], edge[1])] = edge[time_index]
                edge_index[(edge[1], edge[0])] = edge[time_index]
                
    sorted_edges = initial_edges + sorted(timed_edges, key=lambda edge: int(edge[0]))
    
    # G = nx.Graph()
    # print(nx.number_of_nodes(G), nx.number_of_edges(G))
    # for e in sorted_edges:
    #     G.add_edge(e[1], e[2])
    # # G_sub = random_walk_with_restart_sampling(G, 0.1, 0.15, sd=2)
    # print("subgraph done")
    # print(nx.number_of_nodes(G_sub), nx.number_of_edges(G))
    # G.remove_edges_from(nx.selfloop_edges(G))
    # print("after remove self loops:", nx.number_of_nodes(G), nx.number_of_edges(G))
    #
    # sub_sorted_edges = []
    # for e in G.edges():
    #     sub_sorted_edges.append([str(edge_index[e]), e[0], e[1]])

    return sorted_edges

def split_data(sorted_edges, training_precentage):
    size = len(sorted_edges)
    return sorted_edges[:int(size * training_precentage)], sorted_edges[int(size * training_precentage):]

def prepare_graph(train, test):
    G_train = nx.Graph()
    for e in train:
        G_train.add_edge(e[1], e[2])
    
    G_test = nx.Graph()
    for e in test:
        if e[1] in G_train.nodes() and e[2] in G_train.nodes():
            G_test.add_edge(e[1], e[2])

    print('G_train:', nx.number_of_nodes(G_train), nx.number_of_edges(G_train))
    print('G_test:', nx.number_of_nodes(G_test), nx.number_of_edges(G_test))
    G_train.remove_edges_from(nx.selfloop_edges(G_train))
    G_test.remove_edges_from(nx.selfloop_edges(G_test))
    print("after remove selfloops:")
    print('G_train:', nx.number_of_nodes(G_train), nx.number_of_edges(G_train))
    print('G_test:', nx.number_of_nodes(G_test), nx.number_of_edges(G_test))
    return G_train, G_test

def load_data_bitcoin(path):
    initial_edges = []
    timed_edges = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            #edge = line.strip().split()
            edge = line.strip().split(',')
            if edge[2] == '\\N':
                initial_edges.append([edge[2], edge[0], edge[1]])
            else:
                timed_edges.append([edge[2], edge[0], edge[1]])
                
    return initial_edges + sorted(timed_edges)

# return neighbors of neighbors of a node
def neighbors_neighbor(G, node):
    neighbors_neighbor = set()
    for n in G[node]:
        for nn in G[n]:
            neighbors_neighbor.add(nn)
    return neighbors_neighbor



"""Naive Similarity"""

# similarity function
# sim(u,v) = ??|N(u) ??? N(v)|/2 + (1-??)?????^dist(i,j)/(|N(u) ??? N(v)|-1)
# ?? in [0,1], ?? in (0,1)
def naive_decay_similarity(CN_graph):
    nodes = nx.number_of_nodes(CN_graph)
    sim_score = 0
    all_shortest_path_length = dict(nx.all_pairs_shortest_path_length(CN_graph))
    if nodes != 1:
        for i in all_shortest_path_length.keys():
            for j in all_shortest_path_length[i].keys():
                if i < j:
                    sim_score += (1/all_shortest_path_length[i][j])/(nodes-1)
    return sim_score

"""Model with Parameter Learning"""

# similarity function
# sim(u,v) = ??|N(u) ??? N(v)|/2 + (1-??)?????^dist(i,j)/(|N(u) ??? N(v)|-1)
# ?? in [0,1], ?? in (0,1)
def gamma_decay_similarity(CN_graph, alpha, gamma):
    nodes = nx.number_of_nodes(CN_graph)
    sim_score = nodes * alpha / 2
    all_shortest_path_length = dict(nx.all_pairs_shortest_path_length(CN_graph))
    if nodes != 1:
        for i in all_shortest_path_length.keys():
            for j in all_shortest_path_length[i].keys():
                if i < j:
                    sim_score += (1-alpha) * (gamma ** all_shortest_path_length[i][j])/(nodes-1)
    return sim_score

def multi_gamma_decay_similarity(CN_graphs, alpha, gamma):
    results = []
    for g in CN_graphs:
        results.append(gamma_decay_similarity(g, alpha, gamma))
    return results

def learning_scalar(actual, observe):
    xiyi = 0
    xi_2 = 0
    for i in range(len(actual)):
        xiyi += actual[i] * observe[i]
        xi_2 += actual[i] * actual[i]
    return xiyi/xi_2

def square_error(actual, observe):
    scalar = learning_scalar(actual, observe)
    error = 0
    for i in range(len(actual)):
        error += (actual[i] - observe[i]/scalar) ** 2
    return error

# def grid_search(G_set, actual, alpha_range, gamma_range):
#     results = []
#     for alpha in alpha_range:
#         for gamma in gamma_range:
#             observe = multi_gamma_decay_similarity(G_set, alpha, gamma)
#             error = square_error(actual, observe)
#             results.append((error, alpha, gamma))
#     return sorted(results)

"""Prepare testing set"""

# generate candidates at random
def random_candidates(G_train, G_test, sd = None):
    candidates = list(G_test.edges())
    size = 2*len(candidates)
    seed(sd)
    while len(candidates) < size:
        n1 = choice(list(G_test.nodes))
        n2 = choice(list(G_test.nodes))
        if (n1 != n2) and ((n1, n2) not in G_train.edges()) and ((n1, n2) not in G_test.edges()):
            candidates.append((n1, n2))
    return candidates

# generate candidates more than one common neighbors
def have_cn_candidates(G_train, G_test, sd = None):
    candidates = list(G_test.edges())
    size = 2*len(candidates)
    seed(sd)
    while len(candidates) < size:
        n1 = choice(list(G_test.nodes))
        nn = neighbors_neighbor(G_train, n1)
        n2 = choice(list(nn))
        if (n1 != n2) and ((n1, n2) not in G_train.edges()) and ((n1, n2) not in G_test.edges()):
            candidates.append((n1, n2))
    return candidates

# generate candidates more than one common neighbors
def have_cn2_candidates(G_train, G_test, sd = None):
    candidates = list(G_test.edges())
    size = 2*len(candidates)
    seed(sd)
    while len(candidates) < size:
        n1 = choice(list(G_test.nodes))
        nn = neighbors_neighbor(G_train, n1)
        n2 = choice(list(nn))
        if (n1 != n2) and ((n1, n2) not in G_train.edges()) and ((n1, n2) not in G_test.edges()) and (len(list(nx.common_neighbors(G_train, n1, n2)))>1):
            candidates.append((n1, n2))
    return candidates

# generate candidates distances less than 3
def less_3_candidates(G_train, G_test, sd = None):
    candidates = list(G_test.edges())
    size = 2*len(candidates)
    seed(sd)
    while len(candidates) < size:
        n1 = choice(list(G_test.nodes))
        currentnode = n1
        for i in range(3):
            nextnode = sample(list(G_train[currentnode]),1)[0]
            currentnode = nextnode
        n2 = currentnode
        if (n1 != n2) and ((n1, n2) not in G_train.edges()) and ((n1, n2) not in G_test.edges()):
            candidates.append((n1, n2))
    return candidates

def unlink_candidates(G_train):
    candidates = list(G_train.edges())
    return candidates



"""Link Prediction"""

# common_neighbor
def common_neighbor(G_train, edge_list):
    y_score = []
    for l in edge_list:
        y_score.append(len(list(nx.common_neighbors(G_train, l[0], l[1]))))
    return y_score

# Adamic Adar
def adamic_adar(G_train, edge_list):
    list_of_scores = list(nx.adamic_adar_index(G_train, ebunch=edge_list))
    y_score = [c for (a,b,c) in list_of_scores]
    return y_score

# resource_allocation_index
def resource_allocation(G_train, edge_list):
    y_score = [c for (a,b,c) in list(nx.resource_allocation_index(G_train, ebunch=edge_list))]
    return y_score

# jaccard_coefficient
def jaccard(G_train, edge_list):
    y_score = [c for (a,b,c) in list(nx.jaccard_coefficient(G_train, ebunch=edge_list))]
    return y_score

# preferential_attachment
def preferential_attach(G_train, edge_list):
    y_score = [c for (a,b,c) in list(nx.preferential_attachment(G_train, ebunch=edge_list))]
    return y_score

# Katz
def katz_index(G_train, edge_list, beta = 0.0005):
    y_score = []
    for l in edge_list:
        paths = list(nx.all_simple_paths(G_train, l[0], l[1], cutoff = 3))
        sc = 0
        for p in paths:
            sc += beta**(len(p)-1)
        y_score.append(sc)
    return y_score

# Naive CNS
def naive_CNS(G_train, edge_list):
    y_score = []
    for l in edge_list:
        cn = list(nx.common_neighbors(G_train, l[0], l[1]))
        cn_graph = G_train.subgraph(cn)
        y_score.append(naive_decay_similarity(cn_graph))
    return y_score

# Complete CNS
def complete_CNS(G_train, edge_list, alpha, gamma):
    y_score = []
    for l in edge_list:
        cn = list(nx.common_neighbors(G_train, l[0], l[1]))
        cn_graph = G_train.subgraph(cn)
        y_score.append(gamma_decay_similarity(cn_graph, alpha, gamma))
    return y_score
if __name__ == '__main__':
    mode = "unlink"
    data = 'haggle'
    sorted_edges = load_data(data)
    print(f"run {mode} on data {data}")

    train, test = split_data(sorted_edges, 0.9)
    train_train, train_validation = split_data(train, 0.75)
    G_train, G_test = prepare_graph(train, test)
    G_train_train, G_train_validation = prepare_graph(train_train, train_validation)

    print(nx.number_of_nodes(G_train), nx.number_of_edges(G_train))

    print(nx.number_of_nodes(G_train_train), nx.number_of_edges(G_train_train))

    if mode == 'link':
        candidates = have_cn_candidates(G_train, G_test, sd=1)
    # candidates = have_cn2_candidates(G_train, G_test, sd = 1)
    # candidates = random_candidates(G_train, G_test, sd = 1)
    # candidates = less_3_candidates(G_train, G_test, sd = 1)
    else:
        assert mode == 'unlink'
        candidates = unlink_candidates(G_train)

    #put it togather
    y_test = []
    for l in candidates:
        if l in G_test.edges():
            y_test.append(1)
        else:
            y_test.append(0)

    # G_train = G_train_train
    cn_score = common_neighbor(G_train, candidates)
    cn_fpr, cn_tpr, _ = roc_curve(y_test, cn_score)
    cn_roc_auc = round(auc(cn_fpr, cn_tpr), 3)
    print("common_neighbor = ", cn_roc_auc)

    aa_score = adamic_adar(G_train, candidates)
    aa_fpr, aa_tpr, _ = roc_curve(y_test, aa_score)
    aa_roc_auc = round(auc(aa_fpr, aa_tpr), 3)
    print("adamic_adar = ", aa_roc_auc)

    ra_score = resource_allocation(G_train, candidates)
    ra_fpr, ra_tpr, _ = roc_curve(y_test, ra_score)
    ra_roc_auc = round(auc(ra_fpr, ra_tpr), 3)
    print("resource_allocation = ", ra_roc_auc)

    j_score = jaccard(G_train, candidates)
    j_fpr, j_tpr, _ = roc_curve(y_test, j_score)
    j_roc_auc = round(auc(j_fpr, j_tpr), 3)
    print("jaccard = ", j_roc_auc)

    pa_score = preferential_attach(G_train, candidates)
    pa_fpr, pa_tpr, _ = roc_curve(y_test, pa_score)
    pa_roc_auc = round(auc(pa_fpr, pa_tpr), 3)
    print("preferential_attach = ", pa_roc_auc)

    katz_score = katz_index(G_train, candidates)
    katz_fpr, katz_tpr, _ = roc_curve(y_test, katz_score)
    katz_roc_auc = round(auc(katz_fpr, katz_tpr), 3)
    print("katz_index = ", katz_roc_auc)

    naive_CNS_score = naive_CNS(G_train, candidates)
    naive_fpr, naive_tpr, _ = roc_curve(y_test, naive_CNS_score)
    naive_roc_auc = round(auc(naive_fpr, naive_tpr), 3)
    print("naive_CNS = ", naive_roc_auc)

    complete_CNS_score = complete_CNS(G_train, candidates, 0, 0.2)
    complete_fpr, complete_tpr, _ = roc_curve(y_test, complete_CNS_score)
    complete_roc_auc = round(auc(complete_fpr, complete_tpr), 3)
    print("complete_CNS = ", complete_roc_auc)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(cn_fpr, cn_tpr, label='CN')
    plt.plot(aa_fpr, aa_tpr, label='AA')
    plt.plot(ra_fpr, ra_tpr, label='RA')
    plt.plot(j_fpr, j_tpr, label='Jaccard')
    plt.plot(pa_fpr, pa_tpr, label='PA')
    plt.plot(katz_fpr, katz_tpr, label='Katz')
    plt.plot(naive_fpr, naive_tpr, label='Naive')
    plt.plot(complete_fpr, complete_tpr, label='Complete')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()





