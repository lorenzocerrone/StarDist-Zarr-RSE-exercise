import zarr
import numpy as np
from pathlib import Path


def load_image_from_zarr(image_path: Path | str,
                         key='raw',
                         resolution: int = 0,
                         channel: int = 0,
                         slices: tuple[slice, ...] = None):

    image_path = Path(image_path).absolute()
    assert image_path.exists(), f"Image file {image_path} does not exist"

    with zarr.open(str(image_path), 'r') as f:
        file = f[key][resolution][channel] if slices is None else f[key][resolution][channel][slices]
    return file


def get_image_shape(image_path: Path, key='raw'):
    with zarr.open(str(image_path), 'r') as f:
        return f[key].shape


def create_image_to_zarr(image: np.ndarray,
                         image_path,
                         key='raw',
                         resolution: int = 0,
                         metadata=None, mode='a'):
    z = zarr.open(image_path, mode=mode)
    key = f'{key}/{resolution}'
    z.create_dataset(key, data=image, compression='gzip', overwrite=True)
    if metadata is not None:
        z.attrs.update(metadata)
