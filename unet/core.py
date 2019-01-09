import os
import sys

absolute_path = os.path.abspath(__file__)

DIR_DEPTH = 1

ROOT = absolute_path
for _ in range(DIR_DEPTH + 1):
    ROOT = os.path.dirname(ROOT)
    sys.path.insert(0, ROOT)

from tframe import console, SaveMode
from tframe.trainers import TrainerHub
from unet.data_utils import load_data, probe
from tframe import Predictor

import numpy as np
from unet.membrane import labelVisualize


from_root = lambda path: os.path.join(ROOT, path)

th = TrainerHub(as_global=True)
th.data_dir = from_root('unet/data')
th.job_dir = from_root('unet/records_unet_alpha')

th.input_shape = [256, 256, 1]

th.allow_growth = False
th.gpu_memory_fraction = 0.35

th.save_mode = SaveMode.ON_RECORD
th.shuffle = True

def activate(export_false=False):
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Predictor)

  train_set, val_set, test_set = load_data(th.data_dir, 600, -1, 1)

  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th,
                probe=lambda t: probe(t, train_set))
  else:
    from tframe.data.images.image_viewer import ImageViewer
    from tframe.data.dataset import DataSet
    import cv2
    import skimage.transform as transform

    dir = os.path.join(th.data_dir, 'test')
    imgs = []

    images = os.listdir(dir)
    for i in range(len(images)):
      img = cv2.imread(os.path.join(dir, images[i]), 0)
      assert isinstance(img, np.ndarray)
      img = img/255
      img = transform.resize(img, [256, 256])
      imgs.append(img.reshape(1, 256, 256, 1))

    X = np.concatenate(imgs)

    test_set_a = DataSet(features=X)

    images = model.predict(test_set_a, batch_size=2)
    images = images.reshape([-1, 256, 256])
    viewer = ImageViewer(DataSet(features=images))
    viewer.show()
    # flag_multi_class = False
    # import skimage.io as io
    # for i, item in enumerate(images):
    #     img = labelVisualize(2, item) if flag_multi_class \
    # else item[:, :, 0]
    #     io.imsave(os.path.join("data", "%d_predicts.png" % i), img)

console.end()