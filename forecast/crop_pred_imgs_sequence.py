'''crop_pred_imgs - crops images to contain only the vehicle of interest.
The metric used is hard coded but actually doesn't matter except for which folder the image gets put
into.

The output is a file structure that looks like:
imgs/
  |
  |-- class_00-05/
  |   |-- XXXXXX.jpg
  |   |-- XXXXXX.jpg
  |   |     ...
  |   +-- XXXXXX.jpg
  |-- class_05-10/
  |   |-- XXXXXX.jpg
  |   |-- XXXXXX.jpg
  |   |     ...
  |   +-- XXXXXX.jpg
  |
  |       ...
  |
  +-- class_75-80/
      |-- XXXXXX.jpg
      |-- XXXXXX.jpg
      |     ...
      +-- XXXXXX.jpg

where each folder is a bin of errors.  Currently, the middle column 'MinFDEK'
'''
import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import Box, view_points
import os
import warnings
from tqdm import tqdm
import json
import pickle
# from nuscenes import NuScenes
# from nuscenes.eval.prediction.config import PredictionConfig, load_prediction_config
# from nuscenes.eval.prediction.data_classes import Prediction
# from nuscenes.prediction import PredictHelper


def bb2d(im, bbox, cam):
    return view_points(bbox.corners(), view=cam, normalize=True).astype(np.int16)


def getbb(pred):
    for cam in pred.values():
        if cam[1] is not None:
            return view_points(cam[1].corners(), view=cam[2], normalize=True).astype(np.int16)
    return np.zeros((3, 8), dtype=np.int16)


def getallbb(imginfo):
    nsamples = len(imginfo)
    allbbs = np.zeros((3, 8, nsamples), dtype=np.int16)
    for i, pred in enumerate(imginfo):
        allbbs[:, :, i] = getbb(pred)
    return allbbs


def cropimg(img: Image, bbox: Box, camera_intrinsic: np.ndarray) -> Image:
    bbox_img = view_points(bbox.corners(), view=camera_intrinsic, normalize=True)
    # in case the annotation runs off the edge of the page
    bbox_img[bbox_img < 0] = 0
    bbox_img[0, bbox_img[0, :] > img.size[0]] = img.size[0]-1
    bbox_img[1, bbox_img[1, :] > img.size[1]] = img.size[1]-1
    # crop
    return img.crop((min(bbox_img[0]), min(bbox_img[1]), max(bbox_img[0]), max(bbox_img[1])))


def main():
    results = np.load('forecast/output/cv_preds_metricsfull.npz', allow_pickle=True)    # Load output of forecasting pipeline
    out_dir = 'forecast/dataset_seq/'
    out_pkl = 'forecast/dataset_big.pkl'

    imginfo = results['imginfo']
    tokens = results['tokens']
    fde = results['MinFDEK'][:, 1]
    good_samples = 0

    all_data = []

    for i, hist in enumerate(tqdm(imginfo)):  # Foreach "prediction" sample
        token = tokens[i]
        folder_name = os.path.join(out_dir, 'sample_' + str(good_samples))
        out_images = {}
        sample_data = {'images': []}

        # if i > 1000:
        #     break

        for t, pred in enumerate(hist): # Loop through all the timesteps, 6 of them
            for cam in pred.values():  # Find the first camera with the car in it
                if cam[1] is not None:
                    break

            if cam[1] is None:
                warnings.warn('annotation not found in sample {:d} - discarding image'.format(i))
                break

            im_name = os.path.join(folder_name, '{}.jpg'.format(t))
            try:
                imgcropped = cropimg(Image.open(cam[0]), cam[1], cam[2])
                if imgcropped.size[0] > 1600 or imgcropped.size[1] > 900 or imgcropped.size[0] < 1 or imgcropped.size[1] < 1:
                    raise "size too big somehow"

                out_images[im_name] = imgcropped
                sample_data['images'].append(imgcropped)
            except Exception as e:
                print('error on image {:d}'.format(i))
                print(e)
                break

            # out_data.append(im_name)
            cam = None

        if len(out_images) < 6:
            continue

        # If all good, then write sample
        out_data = {'FDE': fde[i], 'sample': token['sample'], 'instance': token['instance']}

        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # for k in out_images.keys():
        #     out_images[k].save(k, 'JPEG')
        # out_path = os.path.join(folder_name, 'anno.json')
        # json.dump(out_data, open(out_path, "w"), indent=2)

        sample_data.update(out_data)
        all_data.append(sample_data)

        good_samples += 1

    pickle.dump(all_data, open(out_pkl, 'wb'))


if __name__ == '__main__':
    main()

    # results = np.load('forecast/output/cv_preds_metricsfull.npz', allow_pickle=True)
    # allbbs = getallbb(results['imginfo'])
    # np.save('forecast/output/boundingboxes', allbbs)
