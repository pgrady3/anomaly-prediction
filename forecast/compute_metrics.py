# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for computing metrics for a submission to the nuscenes prediction challenge. """
import os
import argparse
import json
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.prediction.config import PredictionConfig, load_prediction_config
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction import PredictHelper


def compute_metrics(predictions: List[Dict[str, Any]],
                    helper: PredictHelper, config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of output.
    :param predictions: List of prediction JSON objects.
    :param helper: Instance of PredictHelper that wraps the nuScenes val set.
    :param config: Config file.
    :return: Metrics. Nested dictionary where keys are metric names and value is a dictionary
        mapping the Aggregator name to the results.
    """
    n_preds = len(predictions)
    containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}
    for i, prediction_str in enumerate(predictions):
        prediction = Prediction.deserialize(prediction_str)
        ground_truth = helper.get_future_for_agent(prediction.instance, prediction.sample,
                                                   config.seconds, in_agent_frame=False)
        for metric in config.metrics:
            containers[metric.name][i] = metric(ground_truth, prediction)
    aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for metric in config.metrics:
        for agg in metric.aggregators:
            aggregations[metric.name][agg.name] = agg(containers[metric.name])
    return aggregations, containers


def main(nusc: NuScenes, submission_path: str, config_name: str = 'predict_2020_icra.json') -> None:
    """
    Computes metrics for a submission stored in submission_path with a given submission_name with the metrics
    specified by the config_name.
    :param nusc: nuScenes data set object
    :param submission_path: Directory storing submission.
    :param config_name: Name of config file.
    """
    predictions = json.load(open(submission_path, "r"))
    helper = PredictHelper(nusc)
    config = load_prediction_config(helper, config_name)
    results, resultsfull = compute_metrics(predictions, helper, config)
    json.dump(results, open(submission_path.replace('.json', '_metrics.json'), "w"), indent=2)
    print('dumping full results...')
    np.savez(submission_path.replace('.json', '_metricsfull'), **resultsfull)

    print('Results from', submission_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Inference with baseline models.')
    parser.add_argument('--version', help='nuScenes version number.', default='v1.0-trainval')
    parser.add_argument('--data_root', help='Directory storing NuScenes data.', default=os.environ['NUSCENES'])
    parser.add_argument('--submission_path', help='Path storing the submission file.', default='output/covernet_preds.json')
    parser.add_argument('--config_name', help='Config file to use.', default='predict_2020_icra.json')
    args = parser.parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_root)
    main(nusc, args.submission_path, args.config_name)
    main(nusc, 'output/oracle_preds.json', 'predict_2020_icra.json')
    main(nusc, 'output/covernet_preds.json', 'predict_2020_icra.json')
