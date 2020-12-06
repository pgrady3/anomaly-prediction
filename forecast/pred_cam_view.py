# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
#   Adapted by Gerry Chen, 2020

""" Script for viewing camera images from a given data sample. """

import argparse
import json
import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

def get_im_and_box(nusc, token, cam_name="CAM_FRONT", imgAsName=False):
    """
    Plots the camera view with bounding box around the agent to be predicted
    :param ax: matplotlib axes object
    :param nusc: NuScenes object
    :param token: Prediction token consisting of instance_sample token pair
    :param cam_name: Name of camera to use
    :return: (image, box, camera_instrinsic)
    """
    instance_token, sample_token = token.split('_')
    # get tokens
    sample = nusc.get('sample', sample_token)
    sample_cam_token = nusc.get('sample_data', sample['data'][cam_name])['token']
    annotations_in_sample = [nusc.get('sample_annotation', annotation) for annotation in sample['anns']]
    annotation_instance, *_ = (ann for ann in annotations_in_sample if ann['instance_token'] == instance_token)
    # plot
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(sample_cam_token, selected_anntokens=[annotation_instance['token']])
    if imgAsName:
        im = data_path
    else:
        im = Image.open(data_path)
    if len(boxes) > 0:
        return im, boxes[0], camera_intrinsic
    else:
        return im, None, camera_intrinsic

def plot_cam_view(ax, nusc, token, cam_name='CAM_FRONT'):
    """
    Plots the camera view with bounding box around the agent to be predicted
    :param ax: matplotlib axes object
    :param nusc: NuScenes object
    :param token: Prediction token consisting of instance_sample token pair
    :param cam_name: Name of camera to use
    """
    im, box, camera_intrinsic = get_im_and_box(nusc, token, cam_name)
    ax.imshow(im)
    ax.set_title(cam_name)
    ax.axis('off')
    ax.set_aspect('equal')
    if box is not None:
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

def main(version: str, data_root: str,
         split_name: str, output_dir: str, config_name: str = 'predict_2020_icra.json') -> None:
    """
    Performs inference for all of the baseline models defined in the physics model module.
    :param version: nuScenes data set version.
    :param data_root: Directory where the NuScenes data is stored.
    :param split_name: nuScenes data split name, e.g. train, val, mini_train, etc.
    :param output_dir: Directory where predictions should be stored.
    :param config_name: Name of config file.
    """

    print('timing point A')
    nusc = NuScenes(version=version, dataroot=data_root)
    print('timing point B')
    helper = PredictHelper(nusc)
    print('timing point C')
    dataset = get_prediction_challenge_split(split_name, dataroot=data_root)
    print('timing point D')
    config = load_prediction_config(helper, config_name)
    print('timing point E')

    # rasterization
    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=3)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    # loop through training tasks
    for token in dataset[40:60:2]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 9))
        print(token)
        instance_token, sample_token = token.split('_')

        plot_cam_view(axes[1], nusc, token)
        plot_cam_view(axes[2], nusc, token, cam_name='CAM_FRONT_RIGHT')
        axes[0].imshow(mtp_input_representation.make_input_representation(instance_token, sample_token))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display data from a prediction task')
    parser.add_argument('--config_name', help='Config file to use.', default='predict_2020_icra.json')

    args = parser.parse_args()
    main('v1.0-mini', '/Users/Gerry/Downloads/v1.0-mini', 'mini_train', 'out/', args.config_name)
