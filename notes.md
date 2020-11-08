# Notes

This file exists because Gerry was too lazy to clean up the code.

* Data
  * minor modifications were made to `compute_metrics.py`
  * `crop_pred_imgs.py` inputs a metrics file and outputs a directory structure with cropped images.  This was
    1. to make it easier so that later stages wouldn't need to interact with the nuscenes API at all anymore, and
    2. to place images into the standard directory structure expected by off-the-shelf classifiers.
  * in the future, creating image symlinks or just saving the metadata (ie filenames) into our own data structure would probably be easiest (ie image path + bounding box in image coords + metrics).
  * Note: you don't need the full dataset, only the metadata and keyframes.
  * covernet doesn't seem to work.
* Classification
  * I used a `tensorflow-image-detection` from [this github repo](https://github.com/ArunMichaelDsouza/tensorflow-image-detection) based on google's pretrained Inception network.
  * I used Google colab and the notebook is [here](https://colab.research.google.com/drive/1y3zvQDq4K70LZMObGwMCPqTlddma5bQs?usp=sharing).  It:
    * syncs with Google Drive, where I saved the data files.  Login is: gerrysnetflixlogin@gmail.com , (facebook message me for the password).  Alternatively, the download zip file is here: [google drive](https://drive.google.com/file/d/1-322yQqpt4RAYBeB8hR8Xhx6yI8MWh0T/view?usp=sharing)
  * Frank's lab computer had an outdated nvidia driver and I was too lazy to restart it, but it's pretty easy to work with jupyter notebooks over ssh by setting up an ssh tunnel over port 8888: `ssh -N -L 8888:localhost:8888 <remote_user>@<remote_host>`, or just normal coding using normal ssh utilities.
  * I didn't get resnet working, not sure why: [colab link](https://colab.research.google.com/drive/10CCs5chJS6QAI3DYVjzgof1opBBdheF1?usp=sharing)
    * I think I based this on [this article](https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5) with this corresponding [reference code](https://github.com/cfotache/pytorch_imageclassifier/blob/master/PyTorch_Image_Training.ipynb)
    * probably i didn't pay careful attention to the dimensions.
* Detection: TODO
  * [maskRCNN](https://github.com/matterport/Mask_RCNN) seems to be well documented and may be a good option

In summary, this repo is a mess.