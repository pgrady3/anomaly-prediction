from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.data_classes import Prediction
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from draw_future_agents import FutureAgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.map_expansion.map_api import NuScenesMap
from physics import ConstantVelocityHeading, PhysicsOracle
import json
import os
from pred_cam_view import plot_cam_view


class ManualAnnotate:
    def __init__(self):
        DATAROOT = '/home/patrick/datasets/nuscenes'    # This is the path where you stored your copy of the nuScenes dataset.
        self.nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
        self.mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

        self.helper = PredictHelper(self.nuscenes)
        self.physics_oracle = PhysicsOracle(sec_from_now=6, helper=self.helper)

        self.map_rasterizer = StaticLayerRasterizer(self.helper, meters_ahead=60, meters_behind=10, meters_left=35, meters_right=35)
        self.agent_rasterizer = FutureAgentBoxesWithFadedHistory(self.helper, meters_ahead=60, meters_behind=10, meters_left=35, meters_right=35)

        self.json_path = 'manual_results.json'
        self.annotations = []
        if os.path.exists(self.json_path):
            with open(self.json_path) as json_file:
                self.annotations = json.load(json_file)     # Load existing JSON file

    def draw_map(self, instance, sample, sec_forward, predictions=None):
        img_road = self.map_rasterizer.make_representation(instance, sample)
        img_agents = self.agent_rasterizer.make_representation(instance, sample, sec_forward=sec_forward, predictions=predictions)
        return Rasterizer().combine([img_road, img_agents])

    def select_random_sample(self):
        rand_idx = np.random.randint(0, len(self.mini_train))    # Select random sample
        instance, sample = self.mini_train[rand_idx].split("_")
        return instance, sample

    def calc_minFDE1(self):
        errs_noim = []
        errs_wim = []
        for d in self.annotations:
            errs_noim.append(d['err_noimage'])
            errs_wim.append(d['err_wimage'])

        return np.array(errs_noim).mean(), np.array(errs_wim).mean()

    def run_annotate(self):
        instance, sample = self.select_random_sample()
        predictions = self.physics_oracle(f"{instance}_{sample}")

        gt_path = self.helper.get_future_for_agent(instance, sample, 6, in_agent_frame=False)
        gt_end_pos = gt_path[-1, :]

        if np.linalg.norm(gt_path[0, :] - gt_path[-1, :]) > 40:     # If car moves more than this dist in 5.5 seconds
            return

        img_past = self.draw_map(instance, sample, -2, predictions)
        click_num = 0
        err_noimage = 0
        err_wimage = 0

        def onclick(event):
            nonlocal click_num, err_noimage, err_wimage
            if click_num == 0:
                click_pix = np.around(np.array((event.xdata, event.ydata)))
                trans_img = self.agent_rasterizer.make_trans_img()  # Get mapping from pixels - global meters
                click_global_pos = trans_img[int(click_pix[1]), int(click_pix[0]), :2]
                err_noimage = np.linalg.norm(click_global_pos - gt_end_pos, 2)

                token = instance + '_' + sample
                plot_cam_view(axes[1, 0], self.nuscenes, token)
                plot_cam_view(axes[1, 1], self.nuscenes, token, cam_name='CAM_FRONT_RIGHT')
                plt.axes(axes[0, 0])    # Reset active axes
                plt.draw()
            elif click_num == 1:
                click_pix = np.around(np.array((event.xdata, event.ydata)))
                trans_img = self.agent_rasterizer.make_trans_img()  # Get mapping from pixels - global meters
                click_global_pos = trans_img[int(click_pix[1]), int(click_pix[0]), :2]
                err_wimage = np.linalg.norm(click_global_pos - gt_end_pos, 2)

                img_future = self.draw_map(instance, sample, 6, predictions)
                axes[0, 1].imshow(img_future)
                axes[0, 1].set_title('E noim {:.2f}, wim {:.2f}'.format(err_noimage, err_wimage))
                plt.draw()
            else:
                anno = dict()
                anno['instance'] = instance
                anno['sample'] = sample
                anno['err_noimage'] = err_noimage
                anno['err_wimage'] = err_wimage
                self.annotations.append(anno)
                minFDE1_noimage, minFDE1_wimage = self.calc_minFDE1()
                print('Len {}, Mean FDE1 no image {:.2f}, with image {:.2f}'.format(len(self.annotations), minFDE1_noimage, minFDE1_wimage))

                with open(self.json_path, 'w') as outfile:
                    json.dump(self.annotations, outfile, indent=4)    # Write JSON to file

                plt.close(fig)

            click_num += 1

        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(10, 10)
        axes[0, 0].imshow(img_past)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


if __name__ == "__main__":
    annotator = ManualAnnotate()

    while True:
        annotator.run_annotate()

