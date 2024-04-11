from stardist_zarr.io import load_image_from_zarr, get_image_shape, create_image_to_zarr
from warnings import warn


def stardist2D_stacked(file_infos: dict,
                       model_name: str = '2D_versatile_fluo',
                       min_size: int = 1000,
                       threshold: float = 0.5):
    # Load image from zarr
    assert 'path' in file_infos, "Key 'path' missing in file_infos"
    assert 'plate_to_segment' in file_infos, "Key 'plate_to_segment' missing in file_infos"
    assert 'target_channel' in file_infos, "Key 'target_channel' missing in file_infos"
    if 'output_name' not in file_infos:
        warn("Key 'output_name' missing in file_infos. Results will be saved in the default location "
             "'nuclei_stadist'.")

    from pathlib import Path
    image = load_image_from_zarr(file_infos['path'],
                                 key=file_infos['plate_to_segment'],
                                 channel=file_infos['target_channel'],
                                 resolution=0)
    print(image.shape)
    # setup model

    # predict

    # save results to zarr
    pass
