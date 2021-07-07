import pickle
import sys
import os

from GraphMeasures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from GraphMeasures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from GraphMeasures.features_algorithms.vertices.general import GeneralCalculator
from GraphMeasures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from GraphMeasures.features_infra.feature_calculators import FeatureMeta

sys.path.append(os.path.abspath('GraphMeasures'))
import numpy as np
from loggers import PrintLogger
from features_infra.graph_features import GraphFeatures

from features_algorithms.vertices.louvain import LouvainCalculator

from operator import itemgetter as at

SEP = '/'


def create_feats_to_pkl(gnx, params):
    path = os.path.dirname(params.path)
    logger = PrintLogger("MyLogger")
    #Average neighbor degree
    #General
    #Louvain
    #Closeness centrality
    #Load centrality
    #Betweenness centrality
    #Communicability betweenness centrality



    features_meta = {
        "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"nd_avg"}),
        "general": FeatureMeta(GeneralCalculator, {"general"}),
        "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
        "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
        "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load"})


        # "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
        # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator, {"comm_betweenness"})
    }  # Hold the set of features as written here.

    features = GraphFeatures(gnx, features_meta, dir_path=path, logger=logger)
    features.build()
    # raise Exception()
    mx = features.to_matrix(mtype=np.matrix)

    # print(list(features_meta.keys()))
    # print(features.features)
    sorted_features = map(at(1), sorted(features.items(), key=at(0)))
    sorted_features = [feature._print_name for feature in sorted_features if feature.is_relevant() and feature.is_loaded]
    file_path = SEP.join([path, 'mx_line_{}.pkl'.format(params.id)])
    with open(file_path, 'wb') as file:
        pickle.dump((mx, sorted_features), file, protocol=pickle.HIGHEST_PROTOCOL)
    params.feats = file_path

if __name__ == '__main__':
    print()
    # max = 57
    # avg = 6.96

    # max = 4.55
    # avg = 2.65

