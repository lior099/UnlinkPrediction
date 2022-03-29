import os

from Code.out_sources import create_feats_to_pkl
from Code.tasks import Task


class LogisticTask(Task):

    def __init__(self, root='.', task_params=None):
        super().__init__(root, task_params)
        # self.saved_model_path = ''
        # self.gs_data_dir = None
        # self.gs_config_file = None
        # self.state_file = None
        # self.loss_type = 'node'
        # self.fodge_data_dir = None
        # self.graphs_params = None
        # self.state_file = None

    def run_on_snapshots(self, graphs_params):
        for graph_params in graphs_params:
            self.run(graph_params)

    def prepare(self, graph_params):
        self.data_name = graph_params.data_name
        self.destination = SEP.join([self.root, "task_" + str(self), self.data_name])
        self.results_dir = SEP.join([self.destination, "results"])
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.graph_params = graph_params

    def run(self, graph_params):
        self.prepare(graph_params)

        force_create_feats = True
        graph = graph_params.get_graph()
        if not graph_params.feats or force_create_feats:
            print(
                "Didn't find any line_features_{}.pkl file for graph " + graph_params.name + '. Creating: (force is ' + str(
                    force_create_feats) + ')')
            create_feats_to_pkl(graph, graph_params)
        # self.create_fodge_config()
        graph_feats, feats_names = graph_params.get_feats()
        self.run_logistic(graph, graph_feats)
        embedding, all_embeddings = self.load_fodge_results()
        self.eval_fodge(all_embeddings)

    # def create_config(self):
    #     file_name = SEP.join([self.fodge_data_dir, self.data_name+'.txt'])
    #     with open(file_name, 'w+', newline='') as file:
    #         wr = csv.writer(file)
    #         for graph_params in self.graphs_params:
    #             graph = graph_params.get_graph()
    #             for edge in graph.edges:
    #                 wr.writerow([edge[0], edge[1], graph_params.id])

    def run_logistic(self, graph, graph_feats):
        print()

    def load_fodge_results(self):
        from FODGE.fodge.load_data import load_embedding
        with open(self.state_file, 'rb') as file:
            file_names = pickle.load(file)[0]
        embedding = load_embedding(self.results_dir, file_names[0])
        all_embeddings = load_embedding(self.results_dir, file_names[1])
        # return self.open_gs_model(args_global, timestamp), auc_dict, history
        return embedding, all_embeddings

    def eval_fodge(self, all_embeddings):
        eval_future = self.task_params.get('eval_future', False)
        all_embeddings = list(all_embeddings.values())
        all_embeddings = [{int(key): value for key, value in embeddings.items()} for embeddings in all_embeddings]
        graphs = [graph_params.get_graph() for graph_params in self.graphs_params]
        auc = []
        num_of_eval = len(self.graphs_params)-2 if eval_future else len(self.graphs_params)-1
        for i in range(num_of_eval):
            embedding1, embedding2 = all_embeddings[i], all_embeddings[i+1]
            if eval_future:
                edges = list(set(graphs[i].edges).intersection(set(graphs[i+1].edges)))
                eval_graph = self.graphs_params[i+1]
            else:
                edges = graphs[i].edges
                eval_graph = self.graphs_params[i]
            preds = self.predict(edges, embedding1, embedding2)
            labels = eval_graph.get_labels(edges=edges)
            # print(sum(labels), sum(preds))
            fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
            auc.append(round(metrics.auc(fpr, tpr), 3))
            # f1.append([metrics.f1_score(labels, preds, average="micro"), metrics.f1_score(labels, preds, average="macro")])
        print(auc)
        self.scores = auc



    def predict(self, edges, embedding, future_embedding):
        distances = [np.linalg.norm(embedding[edge[0]]-embedding[edge[1]]) for edge in edges]
        future_distances = [np.linalg.norm(future_embedding[edge[0]] - future_embedding[edge[1]]) for edge in edges]
        print(distances)
        print(future_distances)
        # preds = [1 if dis >= future_dis else 0 for dis, future_dis in zip(distances, future_distances)]
        preds = [dis - future_dis for dis, future_dis in zip(distances, future_distances)]
        return preds



    def __str__(self):
        return 'logistic'

