import os
import zarr
import argparse
from tqdm import tqdm

from datasets.loader import get_loader

parser = argparse.ArgumentParser(description='')

# Task parameters
parser.add_argument('--task', type=str, default="fashion",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--chunk-size', type=int, default=64,
                    help='chunk size for zarr; set this to a typical minibatch size (default: 64)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--output-data-dir', type=str, default='./processed',
                    help='directory which contains zarr data')
args = parser.parse_args()


# Read the full dataset with train / test / val
dataset = get_loader(task=args.task, data_dir=args.data_dir, batch_size=args.batch_size, cuda=False)


def loader_to_zarr(loader, loader_size, feature_shape, output_zarr_prefix):
    """Takes a train/test/val loader and creates a zarr array"""
    if not os.path.isdir(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    z_dataset = zarr.open(os.path.join(args.output_data_dir, '{}_{}_data.zarr'.format(args.task, output_zarr_prefix)), mode='w',
                          shape=(loader_size, *feature_shape), chunks=(args.chunk_size, *feature_shape), dtype='float32')
    z_labels = zarr.open(os.path.join(args.output_data_dir, '{}_{}_labels.zarr'.format(args.task, output_zarr_prefix)), mode='w',
                         shape=(loader_size), chunks=(args.chunk_size), dtype='int64')

    for idx, (sample, label) in tqdm(enumerate(loader)):
        if len(sample.shape) == 4:
            sample = sample.squeeze(0)

        z_dataset[idx] = sample
        z_labels[idx] = label


splits = ['train_loader', 'test_loader', 'valid_loader']
sample_count_dict = {
    'train_loader': dataset.num_train_samples,
    'test_loader': dataset.num_test_samples,
    'valid_loader': dataset.num_valid_samples,
}

for s in splits:
    if hasattr(dataset, s):
        # Grab the corresponding set
        ldr = getattr(dataset, s)

        # Ignore cases where valid_loader is an attr, but is None
        if ldr is None:
            continue

        # save to zarr
        loader_to_zarr(ldr, sample_count_dict[s], dataset.input_shape, s.split('_')[0])
