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


class ManualAnnotate:
    def __init__(self):
        DATAROOT = '/home/patrick/datasets/nuscenes'    # This is the path where you stored your copy of the nuScenes dataset.
        nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
        self.mini_train = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)
        self.all_instances = get_prediction_challenge_split("mini_train", dataroot=DATAROOT)

        self.helper = PredictHelper(nuscenes)
        self.physics_oracle = PhysicsOracle(sec_from_now=6, helper=self.helper)

        self.map_rasterizer = StaticLayerRasterizer(self.helper, meters_ahead=60, meters_behind=10, meters_left=35, meters_right=35)
        self.agent_rasterizer = FutureAgentBoxesWithFadedHistory(self.helper, meters_ahead=60, meters_behind=10, meters_left=35, meters_right=35)

    def draw_map(self, instance, sample, sec_forward, predictions=None):
        img_road = self.map_rasterizer.make_representation(instance, sample)
        img_agents = self.agent_rasterizer.make_representation(instance, sample, sec_forward=sec_forward, predictions=predictions)
        return Rasterizer().combine([img_road, img_agents])

    def select_random_sample(self):
        rand_idx = np.random.randint(0, len(self.all_instances))    # Select random sample
        instance, sample = self.mini_train[rand_idx].split("_")
        return instance, sample

    def run_annotate(self):
        instance, sample = self.select_random_sample()
        predictions = self.physics_oracle(f"{instance}_{sample}")

        gt_path = self.helper.get_future_for_agent(instance, sample, 6, in_agent_frame=False)
        gt_end_pos = gt_path[-1, :]

        if np.linalg.norm(gt_path[0, :] - gt_path[-1, :]) > 40:     # If car moves more than this dist in 5.5 seconds
            return

        img_past = self.draw_map(instance, sample, -2, predictions)
        first_click = True

        def onclick(event):
            nonlocal first_click
            if first_click:
                first_click = False

                click_pos = np.array((event.xdata, event.ydata))
                click_pos = np.around(click_pos)
                trans_img = self.agent_rasterizer.make_trans_img()
                click_global_pos = trans_img[int(click_pos[1]), int(click_pos[0]), :2]
                err = np.linalg.norm(click_global_pos - gt_end_pos, 2)

                print('You clicked', click_global_pos)
                print('Agent pos', gt_end_pos)
                print('Error dist', err)


                img_future = self.draw_map(instance, sample, 6, predictions)
                axes[1].imshow(img_future)
                plt.show()
            else:
                plt.close(fig)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_past)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


if __name__ == "__main__":
    annotator = ManualAnnotate()

    while True:
        annotator.run_annotate()

