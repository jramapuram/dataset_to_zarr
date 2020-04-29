# dataset_to_zarr

Convert datasets from jramapuram/datasets to zarr

## TODO

Currently zarr_dataset is just wrapped with an pytorch loader.
This is not efficient because it pulls samples of size 1.
