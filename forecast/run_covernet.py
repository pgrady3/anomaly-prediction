# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

""" Script for running baseline models on a given nuscenes-split. """

import argparse
import json
import os

from nuscenes import NuScenes
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle, Baseline
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet
import torch
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
import matplotlib.pyplot as plt
import pickle


PATH_TO_EPSILON_8_SET = "covernet/epsilon_8.pkl"


class CoverNetBaseline:
    def __init__(self, sec_from_now: float, helper: PredictHelper):
        """
        Inits Baseline.
        :param sec_from_now: How many seconds into the future to make the prediction.
        :param helper: Instance of PredictHelper.
        """
        assert sec_from_now % 0.5 == 0, f"Parameter sec from now must be divisible by 0.5. Received {sec_from_now}."
        self.helper = helper
        self.sec_from_now = sec_from_now
        self.sampled_at = 2  # 2 Hz between annotations.

        backbone = ResNetBackbone('resnet50')
        self.mtp = MTP(backbone, num_modes=2)

        self.covernet = CoverNet(backbone, num_modes=64)    # Note that the value of num_modes depends on the size of the lattice used for CoverNet.

        static_layer_rasterizer = StaticLayerRasterizer(helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
        self.mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        self.trajectories = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
        self.trajectories = torch.Tensor(self.trajectories)

    def __call__(self, token: str) -> Prediction:
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """
        instance_token_img, sample_token_img = token.split("_")
        # kinematics = _kinematics_from_tokens(self.helper, instance, sample)
        # cv_heading = _constant_velocity_heading_from_kinematics(kinematics, self.sec_from_now, self.sampled_at)

        # anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token_img]
        img = self.mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
        image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)

        plt.imshow(img)

        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                            self.helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                            self.helper.get_heading_change_rate_for_agent(instance_token_img,
                                                                                     sample_token_img)]])

        mtp_out = self.mtp(image_tensor, agent_state_vector)
        covernet_logits = self.covernet(image_tensor, agent_state_vector)

        covernet_probabilities = covernet_logits.argsort(descending=True).squeeze()
        covernet_probabilities = covernet_probabilities[:5]     # Print 5 most likely output
        covernet_trajectories = self.trajectories[covernet_probabilities]
        covernet_trajectories = covernet_trajectories.detach().cpu().numpy()
        covernet_probabilities = covernet_probabilities.detach().cpu().numpy()

        # Need the prediction to have 2d.
        return Prediction(instance_token_img, sample_token_img, covernet_trajectories, covernet_probabilities)


def main(version: str, data_root: str,
         split_name: str, output_dir: str, config_name: str = 'predict_2020_icra.json') -> None:
    """
    Performs inference for all of the baseline models defined in the physics model module.
    :param version: nuScenes data set version.
    :param data_root: Directory where the NuScenes data is stored.
    :param split_name: nuScenes data split name, e.g. train, val, mini_train, etc.
    :param output_dir: Directory where output should be stored.
    :param config_name: Name of config file.
    """

    print('Dataset dir:', data_root)
    nusc = NuScenes(version=version, dataroot=data_root)
    helper = PredictHelper(nusc)
    dataset = get_prediction_challenge_split(split_name, data_root)
    config = load_prediction_config(helper, config_name)

    oracle = PhysicsOracle(config.seconds, helper)
    cv_heading = ConstantVelocityHeading(config.seconds, helper)
    covernet = CoverNetBaseline(config.seconds, helper)

    cv_preds = []
    oracle_preds = []
    covernet_preds = []
    for token in dataset:
        cv_preds.append(cv_heading(token).serialize())
        oracle_preds.append(oracle(token).serialize())
        covernet_preds.append(covernet(token).serialize())

    json.dump(cv_preds, open(os.path.join(output_dir, "cv_preds.json"), "w"))
    json.dump(oracle_preds, open(os.path.join(output_dir, "oracle_preds.json"), "w"))
    json.dump(covernet_preds, open(os.path.join(output_dir, "covernet_preds.json"), "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform Inference with baseline models.')
    parser.add_argument('--version', help='nuScenes version number.', default='v1.0-mini')
    parser.add_argument('--data_root', help='Directory storing NuScenes data.', default=os.environ['NUSCENES'])
    parser.add_argument('--split_name', help='Data split to run inference on.', default='mini_train')
    parser.add_argument('--output_dir', help='Directory to store output files.', default='output')
    parser.add_argument('--config_name', help='Config file to use.', default='predict_2020_icra.json')

    args = parser.parse_args()
    main(args.version, args.data_root, args.split_name, args.output_dir, args.config_name)
