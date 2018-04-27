# basic image mgmt functions
from __future__ import print_function
import numpy as np
import PIL

from IPython import display as ipyd
from lucid.misc.io import reading, showing
from skimage import io
# http://scikit-image.org/docs/dev/api/skimage.transform.html
from skimage import transform
from IPython import display as ipyd

# from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx, :]
 
def square_shrink(img, dim):
  smaller_dim = min(img.shape[:2])
  cropped = crop_center(img, smaller_dim, smaller_dim)
  small_cropped = transform.resize(cropped, (dim, dim, img.shape[2]), mode='constant')
  return small_cropped

def load_img(fname, dim=None):
  img = io.imread(fname)
  if not dim:
    return transform.resize(img, (img.shape[0], img.shape[1], img.shape[2]), mode='constant')
  small_img = square_shrink(img, dim=dim)
  return small_img

def upload_images():
  from google.colab import files
  uploaded = files.upload()
  img_fnames = []
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    img_fnames.append(fn)
  return img_fnames
