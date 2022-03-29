import pickle
import sys
import os

from GraphMeasures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from GraphMeasures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from GraphMeasures.features_algorithms.vertices.general import GeneralCalculator
from GraphMeasures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from GraphMeasures.features_infra.feature_calculators import FeatureMeta
from features_algorithms.edges.edge_betweenness_centrality import EdgeBetweennessCalculator
from features_algorithms.edges.edge_current_flow_betweenness_centrality import EdgeCurrentFlowCalculator
from features_algorithms.edges.edge_degree_based_vertices import EdgeDegreeBasedCalculator
from features_algorithms.edges.minimum_edge_cut import MinimumEdgeCutCalculator
from features_algorithms.edges.neighbor_edges_histogram import NeighborEdgeHistogramCalculator
from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from features_algorithms.vertices.communicability_betweenness_centrality import \
    CommunicabilityBetweennessCentralityCalculator
from features_algorithms.vertices.motifs import MotifsNodeCalculator, MotifsEdgeCalculator

sys.path.append(os.path.abspath('GraphMeasures'))
import numpy as np
from loggers import PrintLogger
from features_infra.graph_features import GraphFeatures

from features_algorithms.vertices.louvain import LouvainCalculator

from operator import itemgetter as at

SEP = '/'


def create_feats_to_pkl(gnx, params, line=False, features_type='node'):
    if features_type == 'node_edge':
        create_feats_to_pkl(gnx, params, line=line, features_type='node')
        create_feats_to_pkl(gnx, params, line=line, features_type='edge')
        return
    path = os.path.dirname(params.path)
    logger = PrintLogger("MyLogger")
    # z-score is in features_infra/feature_calculators.py
    #Average neighbor degree
    #General
    #Louvain
    #Closeness centrality
    #Load centrality
    #Betweenness centrality
    #Communicability betweenness centrality


    if features_type == 'node':
        features_meta = {
            "general": FeatureMeta(GeneralCalculator, {"general"}),
            "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"nd_avg"}),
            "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
            "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
            "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load"}),
            "motifs_node": FeatureMeta(MotifsNodeCalculator, {"motif"}),


            "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
            # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator, {"comm_betweenness"})
        }  # Hold the set of features as written here.
        file_name = f'features_{params.id}.pkl'

    else:
        assert features_type == 'edge'
        features_meta = {
            "edge_betweenness": FeatureMeta(EdgeBetweennessCalculator, {"e_bet"}),
            "edge_current_flow": FeatureMeta(EdgeCurrentFlowCalculator, {"e_flow"}),
            "edge_degree_based": FeatureMeta(EdgeDegreeBasedCalculator, {"e_degree"}),
            "motifs_edge_3": FeatureMeta(MotifsEdgeCalculator, {"e_motifs"}),
            # "edge_minimum_cut": FeatureMeta(MinimumEdgeCutCalculator, {"e_min_cut"}),
            # "edge_neighbor_histogram": FeatureMeta(NeighborEdgeHistogramCalculator, {"e_neighbor_histogram"}),
        }  # Hold the set of features as written here.
        file_name = f'edge_features_{params.id}.pkl'

    features = GraphFeatures(gnx, features_meta, dir_path=path, logger=logger, is_max_connected=True)
    features.build()
    # raise Exception()
    mx = features.to_matrix(mtype=np.matrix)

    # print(list(features_meta.keys()))
    # print(features.features)
    sorted_features = [at[1] for at in features.items()]
    sorted_features = [feature._print_name for feature in sorted_features if feature.is_relevant() and feature.is_loaded]
    if 'motifs_node_3' in sorted_features:
        sorted_features.insert(sorted_features.index('motifs_node_3') + 1, 'motifs_node_4')
    if 'motifs_edge_3' in sorted_features:
        sorted_features.insert(sorted_features.index('motifs_edge_3') + 1, 'motifs_edge_4')
    # print(sorted_features)
    if line:
        file_name = f'line_features_{params.id}.pkl'
    file_path = SEP.join([path, file_name])
    with open(file_path, 'wb') as file:
        pickle.dump((mx, sorted_features), file, protocol=pickle.HIGHEST_PROTOCOL)
    params.feats = file_path

if __name__ == '__main__':
    print()
    # max = 57
    # avg = 6.96

    # max = 4.55
    # avg = 2.65

