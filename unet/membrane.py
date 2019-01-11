from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tarfile
import platform
import pickle

from tframe import pedia
from tframe.data.base_classes import ImageDataAgent
from tframe.data.dataset import DataSet
import cv2

from keras.preprocessing.image import ImageDataGenerator


def adjustData(img, mask, flag_multi_class, num_class):
  if flag_multi_class:
    img = img / 255
    mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
    new_mask = np.zeros(mask.shape + (num_class,))
    for i in range(num_class):
      # for one pixel in the image, find the class in mask and convert it into one-hot vector
      # index = np.where(mask == i)
      # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
      # new_mask[index_mask] = 1
      new_mask[mask == i, i] = 1
    new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[
      1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class \
      else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1],
                                 new_mask.shape[2]))
    mask = new_mask
  elif np.max(img) > 1:
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
  return img, mask


class Membrane(ImageDataAgent):
  DATA_NAME = 'Membrane'
  TFD_FILE_NAME = 'Membrane.tfd'


  @staticmethod
  def data_augmentation(data_set):
    assert isinstance(data_set, DataSet)
    features = data_set.features
    label = data_set.targets

    features = features.reshape(features.shape )
    label = label.reshape(label.shape)

    data_generator = ImageDataGenerator(
      rotation_range=0.2,
      width_shift_range=0.05, height_shift_range=0.05,
      shear_range=0.05, zoom_range=0.05, horizontal_flip=False,
      fill_mode='nearest')

    x_batch, y_batch = next(data_generator.flow(
      features, label, batch_size=features.shape[0]))
    xs=[]
    ys=[]
    zs = []
    for x, y in zip(x_batch, y_batch):
      xx, yy = adjustData(x, y, False, 2)
      xs.append(xx)
      ys.append(yy)
      zs.append(xx)
      zs.append(yy)

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Z = np.concatenate(zs)

    data_set.features = X.reshape(x_batch.shape)
    data_set.targets = Y.reshape(y_batch.shape)
    Z = Z.reshape([-1, 512, 512])

    return data_set

  @classmethod
  def load(cls, data_dir, train_size, validate_size, test_size, **kwargs):
    train_set, val_set, test_set =  super().load(
      data_dir, train_size, validate_size, test_size,
      flatten=False, one_hot=False)
    assert isinstance(train_set, DataSet)
    # train_set.batch_preprocessor = Membrane.data_augmentation
    return train_set, val_set, test_set

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):

    train_dir = os.path.join(data_dir, 'train/images')
    label_dir = os.path.join(data_dir, 'train/labels')

    xs, ys = [], []

    imgs = os.listdir(train_dir)
    labels = os.listdir(label_dir)

    for i in range(len(imgs)):
      img = cv2.imread(os.path.join(train_dir,imgs[i]), 0)
      assert isinstance(img, np.ndarray)
      label = cv2.imread(os.path.join(label_dir, labels[i]), 0)
      assert isinstance(label, np.ndarray)
      x,y = adjustData(img, label, flag_multi_class=False, num_class=2)
      xs.append(np.reshape(x, newshape=(1,) + img.shape + (1,)))
      ys.append(np.reshape(y, newshape=(1,) + label.shape + (1,)))


    X = np.concatenate(xs)
    Y = np.concatenate(ys)

    return X, Y



if __name__ == '__main__':
  data_dir = 'E:/seg_club/unet/data'
  train_set, val_set, test_set = Membrane.load(data_dir, 600, -1, 14)

  xx = Membrane.data_augmentation(train_set)
  print(xx.targets.shape)
  print(xx.features.shape)
  print('Hello')

