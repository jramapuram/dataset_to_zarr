import os
import torch
import zarr
import functools
import numpy as np

from PIL import Image

from .abstract_dataset import AbstractLoader


def load_zarr(path, segment='train'):
    """Load the required npz file and return images + labels."""
    choices = [f for f in os.listdir(path) if segment in f]
    assert len(choices) == 2, "found multiple {} files: {}".format(path, choices)

    # Separate the data filename from the label filename
    data_filename = [c for c in choices if 'data' in c][0]
    label_filename = [c for c in choices if 'label' in c][0]

    # load the data and the labels
    data = zarr.open(os.path.join(path, data_filename), mode='r')
    labels = zarr.open(os.path.join(path, label_filename), mode='r')
    return data, labels


class ZarrDataset(torch.utils.data.Dataset):
    """Loads a zarr array for the segment and labels and iterates."""

    def __init__(self, path, segment='train', transform=None, target_transform=None):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform
        self.segment = segment.lower().strip()  # train or test or val

        # load the images and labels
        self.data, self.labels = load_zarr(self.path, segment=self.segment)

    def __getitem__(self, index):
        sample, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(np.uint8(sample.squeeze() * 255), (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ZarrLoader(AbstractLoader):
    """Simple zarr dataset loader. """

    def __init__(self, path, batch_size, num_replicas=1,
                 train_sampler=None, test_sampler=None, valid_sampler=None,
                 train_transform=None, train_target_transform=None,
                 test_transform=None, test_target_transform=None,
                 valid_transform=None, valid_target_transform=None,
                 cuda=True, **kwargs):

        # Curry the train and test dataset generators.
        train_generator = functools.partial(ZarrDataset, path=path, segment='train')
        test_generator = functools.partial(ZarrDataset, path=path, segment='test')
        valid_generator = None
        if len([f for f in os.listdir(path) if 'valid' in f]) > 0:
            valid_generator = functools.partial(ZarrDataset, path=path, segment='valid')

        super(ZarrLoader, self).__init__(batch_size=batch_size,
                                         train_dataset_generator=train_generator,
                                         test_dataset_generator=test_generator,
                                         valid_dataset_generator=valid_generator,
                                         train_sampler=train_sampler,
                                         test_sampler=test_sampler,
                                         valid_sampler=valid_sampler,
                                         train_transform=train_transform,
                                         train_target_transform=train_target_transform,
                                         test_transform=test_transform,
                                         test_target_transform=test_target_transform,
                                         valid_transform=valid_transform,
                                         valid_target_transform=valid_target_transform,
                                         num_replicas=num_replicas, cuda=cuda, **kwargs)
        self.output_size = self.determine_output_size()
        self.loss_type = 'ce'  # TODO: dyanmically find this.

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        self.input_shape = list(test_img.size()[1:])
        print("derived image shape = ", self.input_shape)
