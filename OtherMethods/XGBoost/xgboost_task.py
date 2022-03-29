import os
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

from Code.graphs import prepare_graphs, get_edge_role, Params
from Code.out_sources import create_feats_to_pkl
from Code.tasks import Task

SEP = '/'

class XGBoostTask(Task):

    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)

    def run_on_snapshots(self, graphs_params):
        prepare_graphs(graphs_params, test_seed=None)
        for graph_params in graphs_params:
            self.run(graph_params)

    def prepare(self, graph_params):
        self.data_name = graph_params.data_name
        self.destination = SEP.join([self.root, "task_" + str(self), self.data_name])
        self.results_dir = SEP.join([self.destination, "results"])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.graph_params = graph_params

    def run(self, graph_params: Params):
        if not graph_params.future or graph_params.mode != graph_params.future.mode:
            return
        # print("task", str(self), 'graph', graph_params.name)
        self.prepare(graph_params)

        force_create_feats = self.task_params['create_feats']
        graph = graph_params.get_graph()
        if not graph_params.feats or force_create_feats:
            print("Didn't find any features_{}.pkl file for graph " + graph_params.name + '. Creating: (force is ' + str(force_create_feats) + ')')
            create_feats_to_pkl(graph, graph_params, features_type=self.task_params['features_type'])
        # self.create_fodge_config()

        graph_feats, feats_names = graph_params.get_feats()
        feats_dict = self.concat_nodes(graph_feats)

        graph_edge_feats, edge_feats_names = graph_params.get_feats(type='edge')
        edge_feats_dict = self.to_edge_dict(graph_edge_feats)

        if self.task_params['features_type'] == 'node':
            self.feats_names = feats_names
            auc = self.run_xgboost(feats_dict)
        elif self.task_params['features_type'] == 'edge':
            self.feats_names = edge_feats_names
            auc = self.run_xgboost(edge_feats_dict)
        else:
            assert self.task_params['features_type'] == 'node_edge'
            self.feats_names = np.append(feats_names, edge_feats_names)
            node_edge_feats_dict = self.combine_nodes_and_edges(feats_dict, edge_feats_dict)
            auc = self.run_xgboost(node_edge_feats_dict)
        self.update_scores(auc)
        # embedding, all_embeddings = self.load_fodge_results()
        # self.eval_fodge(all_embeddings)

    # def create_config(self):
    #     file_name = SEP.join([self.fodge_data_dir, self.data_name+'.txt'])
    #     with open(file_name, 'w+', newline='') as file:
    #         wr = csv.writer(file)
    #         for graph_params in self.graphs_params:
    #             graph = graph_params.get_graph()
    #             for edge in graph.edges:
    #                 wr.writerow([edge[0], edge[1], graph_params.id])

    def run_xgboost(self, feats_dict):

        edge_role = get_edge_role(self.graph_params.role, [self.graph_params])
        labels = self.graph_params.get_labels(dict_output=True)

        train_edges = edge_role['tr'] + edge_role['va']
        test_edges = edge_role['te']

        X_train = [feats_dict[edge] for edge in train_edges]
        X_test = [feats_dict[edge] for edge in test_edges]

        # if self.task_params['nodes_features']:
        #     graph_feats_dict = {node: np.array(feats)[0] for node, feats in
        #                         zip(sorted(graph_params.get_graph()), graph_feats)}
        #     if self.task_params['feats_idx'] is not None:
        #         # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
        #         graph_feats_dict = {node: feats[self.task_params['feats_idx']] for node, feats in
        #                             graph_feats_dict.items()}
        #     X_train = [np.append(graph_feats_dict[edge[0]], graph_feats_dict[edge[1]]) for edge in train_edges]
        #     X_test = [np.append(graph_feats_dict[edge[0]], graph_feats_dict[edge[1]]) for edge in test_edges]
        # else:
        #     graph_feats_dict = {tuple(sorted(edge)): np.array(feats)[0] for edge, feats in
        #                         zip(sorted(graph_params.get_graph().edges()), graph_feats)}
        #     if self.task_params['feats_idx'] is not None:
        #         # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
        #         graph_feats_dict = {edge: feats[self.task_params['feats_idx']] for edge, feats in
        #                             graph_feats_dict.items()}
        #     X_train = [graph_feats_dict[edge] for edge in train_edges]
        #     X_test = [graph_feats_dict[edge] for edge in test_edges]
        y_train = [labels[edge] for edge in train_edges]
        y_test = [labels[edge] for edge in test_edges]
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth = 1, random_state = 0).fit(X_train, y_train)
        # print("Mean Accuracy", clf.score(X_test, y_test))
        probs = clf.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1], pos_label=1)
        # print("AUC: ",round(metrics.auc(fpr, tpr), 3))
        return round(metrics.auc(fpr, tpr), 3)

    def concat_nodes(self, graph_feats):
        if graph_feats is None:
            return None
        feats_dict = {node: np.array(feats)[0] for node, feats in
                            zip(sorted(self.graph_params.get_graph()), graph_feats)}
        if self.task_params['feats_idx'] is not None:
            # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
            feats_dict = {node: feats[self.task_params['feats_idx']] for node, feats in
                                feats_dict.items()}
        edges = [tuple(sorted(edge)) for edge in sorted(self.graph_params.get_graph().edges())]
        feats_dict = {edge: np.append(feats_dict[edge[0]], feats_dict[edge[1]]) for edge in edges}
        return feats_dict

    def to_edge_dict(self, graph_edge_feats):
        if graph_edge_feats is None:
            return None
        edge_feats_dict = {tuple(sorted(edge)): np.array(feats)[0] for edge, feats in
                            zip(sorted(self.graph_params.get_graph().edges()), graph_edge_feats)}
        if self.task_params['feats_idx'] is not None:
            # print(f"Using features: {np.array(feats_names)[self.task_params['feats_idx']]}")
            edge_feats_dict = {edge: feats[self.task_params['feats_idx']] for edge, feats in
                                edge_feats_dict.items()}
        return edge_feats_dict

    def combine_nodes_and_edges(self, feats_dict, edge_feats_dict):
        assert len(feats_dict) == len(edge_feats_dict)
        node_edge_feats_dict = {edge: np.append(feats_dict[edge], edge_feats_dict[edge]) for edge in feats_dict.keys()}
        return node_edge_feats_dict

    def __str__(self):
        return 'xgboost'