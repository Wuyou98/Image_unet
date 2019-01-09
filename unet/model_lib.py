import tensorflow as tf
from tframe import Classifier, Predictor
from tframe import losses, metrics

from tframe.configs.config_base import Config
from tframe.layers.common import Input, Linear, Dropout, Activation, Flatten
from tframe.layers.common import Reshape
from tframe.layers.convolutional import Conv2D, Deconv2D
from tframe.layers.merge import Concatenate
from tframe.layers.pooling import MaxPool2D

from nets import Unet


def unet(th):
  assert isinstance(th, Config)

  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))

  def add_encoder_block(filters, kernel_size=3, add_pool=True, drop_out=False):
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation.ReLU())
    model.add(Conv2D(filters, kernel_size))
    output = model.add(Activation.ReLU())
    if drop_out: output = model.add(Dropout(0.5))
    if add_pool: model.add(MaxPool2D((2, 2), 2))
    return output

  def add_decoder_block(filters, convX, ks1=2, ks2=3):
    model.add(Deconv2D(filters, ks1, strides=(2, 2)))
    model.add(Activation.ReLU())
    model.add(Concatenate(convX))
    model.add(Conv2D(filters, ks2))
    model.add(Activation.ReLU())
    model.add(Conv2D(filters, ks2))
    model.add(Activation.ReLU())

  # Construct encoder part
  conv64 = add_encoder_block(64)
  conv128 = add_encoder_block(128)
  conv256 = add_encoder_block(256)
  conv512 = add_encoder_block(512, drop_out=True)
  add_encoder_block(1024, add_pool=False, drop_out=True)

  # Construct decoder part
  add_decoder_block(512, conv512)
  add_decoder_block(256, conv256)
  add_decoder_block(128, conv128)
  add_decoder_block(64, conv64)

  # Add output layers
  model.add(Conv2D(2, 3))
  model.add(Activation.ReLU())
  model.add(Conv2D(1, 1))
  model.add(Activation('sigmoid'))

  model.build(optimizer=tf.train.AdamOptimizer(th.learning_rate),
              loss='binary_cross_entropy',  metric='accuracy')

  return model


def unet_0(th):
  assert isinstance(th, Config)

  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  # model.add(Reshape(shape=th.input_shape + [1]))
  model.add(Unet())
  model.add(Activation('sigmoid'))

  model.build(optimizer=tf.train.AdamOptimizer(th.learning_rate),
              loss='binary_cross_entropy',  metric='accuracy')

  return model


def unet_beta(th):
  assert isinstance(th, Config)

  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))

  def add_encoder_block(filters, kernel_size=3, add_pool=True, drop_out=False):
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation.ReLU())
    model.add(Conv2D(filters, kernel_size))
    output = model.add(Activation.ReLU())
    if drop_out: output = model.add(Dropout(0.5))
    if add_pool: model.add(MaxPool2D((2, 2), 2))
    return output

  def add_decoder_block(filters, convX, ks1=2, ks2=3):

    model.add(Deconv2D(filters, ks1, strides=(2, 2)))
    model.add(Activation.ReLU())
    model.add(Concatenate(convX))
    model.add(Conv2D(filters, ks2))
    model.add(Activation.ReLU())
    model.add(Conv2D(filters, ks2))
    model.add(Activation.ReLU())

  # Construct encoder part
  conv64 = add_encoder_block(64)
  conv128 = add_encoder_block(128)
  conv256 = add_encoder_block(256)
  # conv512 = add_encoder_block(512, drop_out=True)
  add_encoder_block(1024, add_pool=False, drop_out=True)
  # add_encoder_block(512, add_pool=False, drop_out=True)

  # Construct decoder part
  # add_decoder_block(512, conv512)
  add_decoder_block(256, conv256)
  add_decoder_block(128, conv128)
  add_decoder_block(64, conv64)

  # Add output layers
  model.add(Conv2D(2, 3))
  model.add(Activation.ReLU())
  model.add(Conv2D(1, 1))
  model.add(Activation('sigmoid'))

  model.build(optimizer=tf.train.AdamOptimizer(th.learning_rate),
              loss='binary_cross_entropy', metric='accuracy')

  return model




if __name__ == '__main__':
  th = Config()
  th.input_shape = [256, 256, 1]
  th.learning_rate = 1e-4
  th.mark = 'Hello'
  th.show_structure_detail = True

  model = unet(th)


