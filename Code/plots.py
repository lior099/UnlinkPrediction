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


def plot_loss_and_auc(train_loss, train_auc, val_loss, val_auc, path, data, snapshot):
    fig, axs = plt.subplots(2, 2)
    plt.suptitle('Loss and AUC of '+data.title()+' snapshot ' + snapshot)
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

if __name__ == '__main__':
    data = 'radoslaw'
    # plot_temp(['.\\Results\\task_line\\dblp\\task_line_dblp_scores0_8.csv'], ['blue'], 'Snapshot', 'AUC', save_path='.\\Results\\task_line\\dblp\\task_line_dblp_scores_bars.png', my_labels=['AUC'], title='AUC of DBLP snapshots')
    plot_temp(['.\\Results\\task_line\\' + data + '\\task_line_' + data + '_scores.csv'], ['blue'], 'Snapshot', 'AUC',
              save_path='.\\Results\\task_line\\' + data + '\\task_line_' + data + '_scores_bars.png', my_labels=['AUC'],
              title='AUC of ' + data + ' snapshots', y_range=[0.45, 0.8], plot_type='line')