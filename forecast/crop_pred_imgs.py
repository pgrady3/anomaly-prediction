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
  # setup
  results = np.load('forecast/output/cv_preds_metricsfull.npz', allow_pickle=True)
  nsamples = results['MinFDEK'].shape[0]
  subdir = 'forecast/imgs_fixedsize/'
  # classification cutoffs
  fde = results['MinFDEK'][:, 1]
  binsize = 5
  fdecutoffs = np.arange(0, fde.max(), binsize, dtype=np.int16)  # 0-max in bin sizes of binsize
  def classification_ind(ind: int) -> int:
    return np.argwhere(fde[ind] >= fdecutoffs)[-1][0]
  def class_foldername(ind: int) -> str:
    i = classification_ind(ind)
    return 'class_{:02d}-{:02d}/'.format(fdecutoffs[i], fdecutoffs[i]+binsize)
  # create folders
  for cutoff in fdecutoffs:
    foldername = subdir + 'class_{:02d}-{:02d}/'.format(cutoff, cutoff+binsize)
    if not os.path.exists(os.path.dirname(foldername)):
      try:
        os.makedirs(os.path.dirname(foldername))
      except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
          raise
  # filter camera views that contain the bounding box and crop and save
  results['imagepaths'] = ['' for i in range(nsamples)]
  for i, pred in enumerate(results['imginfo']):  # pred: Dict[str, List[str, Any, np.ndarray]]
    # print(i)
    if i % 100 == 0:
      print('{:.1f}%% done - {:d} out of {:d} images'.format(i / nsamples * 100, i, nsamples))
    for cam in pred.values():  # cam: (str, Any, nd.ndarray)
      if cam[1] is not None:
        break
    if cam[1] is None:
      warnings.warn('annotation not found in sample {:d} - discarding image'.format(i))
      allimgs.append('INVALID')
    else:
      imname = subdir + class_foldername(i) + '{:06d}.jpg'.format(i)
      try:
        imgcropped = cropimg(Image.open(cam[0]), cam[1], cam[2])
        if imgcropped.size[0] > 1600 or imgcropped.size[1] > 900 or \
           imgcropped.size[0] < 1 or imgcropped.size[1] < 1:
          raise "size too big somehow"
      except Exception as e:
        print('error on image {:d}'.format(i))
        print(e)
        allimgs.append('ERROR')
        continue
      imgcropped.save(imname, 'JPEG')
      allimgs.append(imname)
    cam = None
  assert len(allimgs) == nsamples, 'mismatched images'

if __name__ == '__main__':
  # main()
  results = np.load('forecast/output/cv_preds_metricsfull.npz', allow_pickle=True)
  allbbs = getallbb(results['imginfo'])
  np.save('forecast/output/boundingboxes', allbbs)
