from tframe import DataSet
from unet.membrane import Membrane


def load_data(path, train_size, val_size, test_size):
    train_set, val_set, test_set = Membrane.load(
        path, train_size, val_size, test_size, flatten=False, one_hot=False
    )
    assert isinstance(test_set, DataSet)
    assert isinstance(train_set, DataSet)
    assert isinstance(val_set, DataSet)

    return train_set, val_set, test_set


def probe(trainer, data_set):
    from tframe import Classifier
    from tframe.trainers.trainer import Trainer
    from tframe.data.dataset import DataSet
    from tframe.data.images.image_viewer import ImageViewer
    import numpy as np
    assert isinstance(trainer, Trainer)
    assert isinstance(data_set, DataSet)
    model = trainer.model
    assert isinstance(model, Classifier)
    image = model.predict(data_set, batch_size=2)
    image = image.reshape([-1, 512, 512])
    assert isinstance(image, np.ndarray)
    viewer = ImageViewer(DataSet(features=image))
    viewer.show()

    a = 1

    # print('shape = {}'.format(image.shape))



if __name__ == '__main__':
    from tframe.data.images.image_viewer import ImageViewer

    data_path = './data'
    train_set, val_set, test_set = load_data(data_path, 600, -1, 15)

    ImageViewer.show_images(train_set, interleave_key='targets')


