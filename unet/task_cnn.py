import tensorflow as tf
from tframe import console
import core
import model_lib as models


def main(_):
  console.start('Task CNN (MEMBRANE)')

  th = core.th
  th.job_dir = './records_unet_alpha'
  th.model = models.unet
  th.suffix = '01'

  th.batch_size = 2
  th.learning_rate = 1e-4

  th.epoch = 3
  th.early_stop = True
  th.patience = 5
  th.print_cycle = 1
  th.validation_per_round = 4
  th.val_batch_size = 10
  th.validate_train_set = True
  th.export_tensors_upon_validation = True
  # th.probe_cycle = 1
  th.warm_up = False

  th.save_model = True
  th.overwrite = True
  th.gather_note = True
  th.summary = False
  th.warm_up_thres = 0
  #
  th.train = False
  th.mark = 'unet_{}'.format('x')
  core.activate()


if __name__ == '__main__':
  tf.app.run()



