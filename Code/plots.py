import csv
import os

import matplotlib.pyplot as plt
import numpy as np

def plot_graph(data_paths, colors, x_label, y_label, save_path=None, my_labels=None, title=''):
    x_list, y_list, labels = [], [], []
    for path in data_paths:
        with open(path, 'r', newline='') as file:
            data = list(csv.reader(file))
            x_list.append([float(i) for i in data[0][1:]])
            y_list.append([float(i) for i in data[1][1:]])
            labels.append(data[1][0])
    if my_labels:
        labels = my_labels
    for x, y, color, label in zip(x_list, y_list, colors, labels):
        x, y = zip(*sorted(zip(x, y)))
        plt.plot(x, y, "-ok", color=color, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xticks(np.arange(0.0, 0.6, 0.1))
    # plt.xscale('log')
    if y_label not in ['Accuracy', 'AUC']:
        plt.yscale('log')
    plt.title(title)
    plt.grid(True, linestyle='--', which="both")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()

def plot_temp(data_paths, colors, x_label, y_label, save_path=None, my_labels=None, title='', y_range=None, plot_type='bar'):
    x_list, y_list, labels = [], [], []
    for path in data_paths:
        with open(path, 'r', newline='') as file:
            data = list(csv.reader(file))
            x_list.append([float(i) for i in data[0][1:]])
            y_list.append([float(i) for i in data[1][1:]])
            labels.append(data[1][0])
    if my_labels:
        labels = my_labels
    for x, y, color, label in zip(x_list, y_list, colors, labels):
        x, y = zip(*sorted(zip(x, y)))
        if plot_type == 'bar':
            plt.bar(x, y, color=color, label=label)
        else:
            assert plot_type == 'line'
            plt.plot(x, y, "-ok", color=color, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xticks(np.arange(0.0, 0.6, 0.1))
    # plt.xscale('log')
    if y_label not in ['Accuracy', 'AUC']:
        plt.yscale('log')
    plt.title(title)
    # plt.grid(True, linestyle='--', which="both")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    if y_range is None:
        y_range = [0.5, 0.7]
    plt.ylim(y_range)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.clf()

def plot_results(results, colors, x_label, y_label, graph_name, labels=None, title=''):
    paths = []
    for result in results:
        path = os.path.join("..",
                            "Results",
                            "_".join(["task", result["task"]]),
                            "_".join(["task", result["task"], result["name"]]),
                            "_".join(["task", result["task"], result["name"], result["graph"]]) + '.csv')
        paths.append(path)
    save_path = os.path.join("..", "Results", graph_name + ".png")
    plot_graph(paths, colors, x_label=x_label, y_label=y_label, save_path=save_path, my_labels=labels, title=title)
    # plot_graph(paths, colors, x_label="Fraction", y_label="Running Time[s]", save_path=save_path)
    # plot_graph([memory_destination], ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")


def plot_all_results():

    # plot_graph(runtime_destination, ["blue"], x_label="Fraction", y_label="Running Time[s]",
    #            save_path=runtime_destination[:-4] + ".png")
    # plot_graph(memory_destination, ["blue"], x_label="Fraction", y_label="Memory Usage[Mb]",
    #            save_path=memory_destination[:-4] + ".png")
    # runtime_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_runtime.csv'
    # memory_destination = '..\\Results\\task_1\\task_1_false_mass\\task_1_false_mass_memory.csv'



    results = [{'task': 'edge_data', 'name': '0,1_random_networkx', 'graph': 'memory'},
               {'task': 'edge_data', 'name': '0,1_random_lol', 'graph': 'memory'},
               {'task': 'edge_data', 'name': '0,1_random_dynamic_lol', 'graph': 'memory'}]
    plot_results(results, ['blue', 'red', 'black'], 'Nodes', 'Memory', 'task_edge_data_0,1_random_memory',
                 ['networkx', 'lol', 'dynamic_lol'], 'Task edge_data 0,1_random Memory')

    results = [{'task': 'edge_data', 'name': '0,1_random_networkx', 'graph': 'runtime'},
               {'task': 'edge_data', 'name': '0,1_random_lol', 'graph': 'runtime'},
               {'task': 'edge_data', 'name': '0,1_random_dynamic_lol', 'graph': 'runtime'}]
    plot_results(results, ['blue', 'red', 'black'], 'Nodes', 'Runtime', 'task_edge_data_0,1_random_runtime',
                 ['networkx', 'lol', 'dynamic_lol'], 'Task edge_data 0,1_random Runtime')

    results = [{'task': 'nodes', 'name': '0,1_random_networkx', 'graph': 'memory'},
               {'task': 'nodes', 'name': '0,1_random_lol', 'graph': 'memory'},
               {'task': 'nodes', 'name': '0,1_random_dynamic_lol', 'graph': 'memory'}]
    plot_results(results, ['blue', 'red', 'black'], 'Nodes', 'Memory', 'task_nodes_0,1_random_memory',
                 ['networkx', 'lol', 'dynamic_lol'], 'Task nodes 0,1_random Memory')

    results = [{'task': 'nodes', 'name': '0,1_random_networkx', 'graph': 'runtime'},
               {'task': 'nodes', 'name': '0,1_random_lol', 'graph': 'runtime'},
               {'task': 'nodes', 'name': '0,1_random_dynamic_lol', 'graph': 'runtime'}]
    plot_results(results, ['blue', 'red', 'black'], 'Nodes', 'Runtime', 'task_nodes_0,1_random_runtime',
                 ['networkx', 'lol', 'dynamic_lol'], 'Task nodes 0,1_random Runtime')

def plot_features_histogram(x, path, feats_names):
    # np.random.seed(10 ** 7)
    # mu = 121
    # sigma = 21
    # x = mu + sigma * np.random.randn(1000)

    num_bins = 100
    x = np.array(x)
    columns = [x[:,i] for i in range(x.shape[1])]
    for feats, name in zip(columns, feats_names):
        n, bins, patches = plt.hist(feats, num_bins, color='red', alpha=0.7)

        plt.xlabel(name)
        plt.ylabel('Distribution')

        plt.title(name+' Values Distribution', fontweight="bold")
        plt.savefig(path+'/'+name+'.png')
        plt.show()



def plot_loss_and_auc(train_loss, train_auc, val_loss, val_auc, path, data, snapshot, task):
    fig, axs = plt.subplots(2, 2)
    plt.suptitle(f'Loss and AUC of {data.title()} Snapshot {snapshot} Task {task}')
    axs[0, 0].plot(train_loss)
    axs[0, 0].grid(color="w")
    axs[0, 0].set_facecolor('xkcd:light gray')
    axs[0, 0].set_title("Train Loss")
    axs[1, 0].plot(train_auc)
    axs[1, 0].grid(color="w")
    axs[1, 0].set_facecolor('xkcd:light gray')
    axs[1, 0].set_title("Train AUC")
    axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].grid(color="w")
    axs[0, 1].set_facecolor('xkcd:light gray')
    axs[0, 1].plot(val_loss)
    axs[0, 1].set_title("Validation Loss")
    axs[1, 1].plot(val_auc)
    axs[1, 1].grid(color="w")
    axs[1, 1].set_facecolor('xkcd:light gray')
    axs[1, 1].set_title("Validation AUC")
    fig.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_edges_over_snapshots(graphs_params, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)


    graphs_params = graphs_params[:-1]
    graphs = [graph_params.get_graph() for graph_params in graphs_params]
    all_edges = [len(graph.edges) for graph in graphs]
    unlink_edges = [len(graph_params.get_labels()) - sum(graph_params.get_labels()) for graph_params in graphs_params]
    link_edges = [sum(graph_params.get_labels()) for graph_params in graphs_params]

    edges_that_stayed = []
    for graph_params in graphs_params:
        graph_edges = graph_params.get_graph().edges
        labels = graph_params.get_labels()
        edges = [edge for edge, label in zip(graph_edges, labels) if label]
        edges_that_stayed.append(edges)

    future_link_edges, future_unlink_edges = [], []
    for graph_params, edges in zip(graphs_params[:-1], edges_that_stayed):
        future = graph_params.future
        labels = future.get_labels(edges=edges)
        future_link_edges.append(sum(labels))
        future_unlink_edges.append(len(labels) - sum(labels))
        print()

    # future_link_edges = [sum(graph_params.future.get_labels(edges=edges_that_stayed)) for graph_params in graphs_params]
    # future_unlink_edges = [len(graph_params.get_labels(edges=edges_that_stayed)) - sum(graph_params.get_labels(edges=edges_that_stayed)) for graph_params in graphs_params]

    width = 0.35

    x = np.arange(1, len(graphs_params)+1)
    ids = [graph_params.id for graph_params in graphs_params]

    ax.bar(x-0.5*width, link_edges, color='green', label='Link Edges', width=width)
    ax.bar(x-0.5*width, unlink_edges, color='red', label='Unlink Egdes', width=width, bottom=link_edges)
    ax.bar(x+0.5*width, future_link_edges + [0], color='limegreen', label='Future Link Egdes', width=width)
    ax.bar(x + 0.5 * width, future_unlink_edges + [0], color='indianred', label='Future Unlink Egdes', width=width, bottom=future_link_edges+[0])
    # ax.bar(x + 0.5 * width, unlink_edges, color='indianred', label='Future Unlink Egdes', width=width,
    #        bottom=future_link_edges + [0])

    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Edges')

    ax.set_title('Edges Labels Over Snapshots of Data '+graphs_params[0].data_name.capitalize())

    ax.set_xticks([0] + ids)
    ax.legend()
    fig.savefig(path)
    plt.show()
    print()

def plot_disappearing_edges(graphs_params, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    graphs_params = graphs_params[:-1]
    graphs = [graph_params.get_graph() for graph_params in graphs_params]
    all_edges = [len(graph.edges) for graph in graphs]
    unlink_edges_count = [len(graph_params.get_labels()) - sum(graph_params.get_labels()) for graph_params in graphs_params]
    link_edges_count = [sum(graph_params.get_labels()) for graph_params in graphs_params]

    unlink_edges = []
    for graph_params in graphs_params:
        graph_edges = graph_params.get_graph().edges
        labels = graph_params.get_labels()
        edges = [edge for edge, label in zip(graph_edges, labels) if not label]
        unlink_edges.append(edges)

    future_ulinks = {0: [], 1: [], 2: []}
    for graph_params, edges in zip(graphs_params, unlink_edges):
        future = graph_params.future
        nodes = future.get_graph().nodes
        nodes_missing_list = []
        for edge in edges:
            nodes_missing = sum([edge[0] in nodes, edge[1] in nodes])
            nodes_missing_list.append(nodes_missing)
        for key, value in future_ulinks.items():
            future_ulinks[key].append(nodes_missing_list.count(key))

    # future_link_edges = [sum(graph_params.future.get_labels(edges=edges_that_stayed)) for graph_params in graphs_params]
    # future_unlink_edges = [len(graph_params.get_labels(edges=edges_that_stayed)) - sum(graph_params.get_labels(edges=edges_that_stayed)) for graph_params in graphs_params]

    width = 0.35

    x = np.arange(1, len(graphs_params) + 1)
    ids = [graph_params.id for graph_params in graphs_params]

    ax.bar(x - 0.5 * width, unlink_edges_count, color='red', label='Unlink Egdes', width=width, edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5 * width, link_edges_count, color='#00CC00', label='Link Edges', width=width, bottom=unlink_edges_count, edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5 * width, future_ulinks[2], color='#FFCCCC', label='Unlinks with no missing nodes', width=width, edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5 * width, future_ulinks[1], color='#FF9999', label='Unlinks with 1 missing node', width=width,
           bottom=future_ulinks[2], edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5 * width, future_ulinks[0], color='#FF6666', label='Unlinks with 2 missing nodes', width=width,
           bottom=[a+b for a,b in zip(future_ulinks[1], future_ulinks[2])], edgecolor='black', linewidth=0.5)
    # ax.bar(x + 0.5 * width, unlink_edges, color='indianred', label='Future Unlink Egdes', width=width,
    #        bottom=future_link_edges + [0])

    ax.set_xlabel('Snapshot')
    ax.set_ylabel('Edges')

    ax.set_title('Disappearing Edges Over Snapshots of Data ' + graphs_params[0].data_name.capitalize())

    ax.set_xticks([0] + ids)
    ax.legend()
    if path is not None:
        fig.savefig(path)
    plt.show()
    print()




def plot_categories_auc(categories, path, data_name, task):
    x = list(categories.keys())
    y = list(categories.values())
    plot_bars(x, y, x_label="Categories", y_label='AUC', title='Categories AUC of Data '+data_name.capitalize() + ' on Task '+ task.capitalize(), path=path)

def plot_bars(x, y, x_label, y_label, title, path):
    width = 0.35
    plt.bar(x, y, color='red', width=width)
    plt.xticks(x, rotation=15)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

if __name__ == '__main__':
    data = 'radoslaw'
    # plot_temp(['.\\Results\\task_line\\dblp\\task_line_dblp_scores0_8.csv'], ['blue'], 'Snapshot', 'AUC', save_path='.\\Results\\task_line\\dblp\\task_line_dblp_scores_bars.png', my_labels=['AUC'], title='AUC of DBLP snapshots')
    plot_temp(['.\\Results\\task_line\\' + data + '\\task_line_' + data + '_scores.csv'], ['blue'], 'Snapshot', 'AUC',
              save_path='.\\Results\\task_line\\' + data + '\\task_line_' + data + '_scores_bars.png', my_labels=['AUC'],
              title='AUC of ' + data + ' snapshots', y_range=[0.45, 0.8], plot_type='line')