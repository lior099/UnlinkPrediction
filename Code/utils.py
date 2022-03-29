import datetime
import itertools
import os
import random
import numpy as np

from Code.main import prepare_run, run_task


def run_over_feature_combinations():
    # run_type: line, mul, embed, concat, xgboost, (logistic)
    # old data: enron, radoslaw, facebook, catalano
    # new data: facebook, reality, haggle
    subsets = []
    for L in range(1, 10):
        for subset in itertools.combinations(range(0, 9), L):
            subsets.append(list(subset))
    for run_type in ['xgboost']:
        for data in ['haggle']:
            for feats_idx in subsets:
                task, graphs_params = prepare_run(run_type=run_type, data=data, feats_idx=feats_idx)
                run_task(task=task, graphs_params=graphs_params)
                print(f"Finished run {run_type} with feats_idx {np.array(task.feats_names)[feats_idx]} on data {data}: {task.final_scores}")
    print('Finished all runs!')

if __name__ == "__main__":
    print('Start: ', datetime.datetime.now())
    random.seed(0)
    if 'Code' not in os.listdir(os.getcwd()):
        os.chdir("..")
        if 'Code' not in os.listdir(os.getcwd()):
            raise Exception(
                "Bad pathing, use the command os.chdir() to make sure you work on UnlinkPrediction directory")
    run_over_feature_combinations()
    print('End: ', datetime.datetime.now())