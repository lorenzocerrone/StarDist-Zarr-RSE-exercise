from warnings import warn

import numpy as np
from csbdeep.utils import normalize
from scipy.ndimage import zoom
from skimage.metrics import contingency_table
from skimage.morphology import remove_small_objects
from stardist.models import StarDist2D

from stardist_zarr.io import load_image_from_zarr, create_image_to_zarr, get_image_shape


def stack_2d_segmentation(slices_segmentations: list[np.ndarray], threshold: float = 0.5):
    """
    !!!!!!
     PART OF THE CODE IS TAKEN FROM A PREVIOUS PROJECT OF MINE
     PLEASE DO NOT EVALUATE THIS AS PART OF THE CHALLENGE
    !!!!

    This function stacks 2D segmentations in a 3D volume by finding the best match between labels in consecutive slices.
    The best match is found by maximizing the intersection over the minimum size of the two labels, and then checking
        if the intersection is greater than the overlap threshold.
    This method is not robust in general, but since here there are only a few z-slices, it should work well enough.

    Args:
        slices_segmentations (list[np.ndarray]): list of 2D segmentations to be stacked
        threshold (float, optional): threshold for the overlap. Defaults to 0.5.

    Returns:
        np.ndarray: 3D segmentation
    """
    stacked_segmentation = np.zeros((len(slices_segmentations), *slices_segmentations[0].shape),
                                    dtype=slices_segmentations[0].dtype)
    stacked_segmentation[0] = slices_segmentations[0]

    for i in range(1, len(slices_segmentations)):
        slice_1 = stacked_segmentation[i - 1]  # correct indexing
        slice_2 = slices_segmentations[i]
        ct = contingency_table(slice_1, slice_2)

        sizes_map = {idx: size for idx, size in zip(*np.unique(slice_2, return_counts=True))}
        # find the best match for each label (if the intersection is greater than the overlap threshold)
        transactions = {}
        for idx, size in zip(*np.unique(slice_1, return_counts=True)):
            best_match_idx = np.argmax(ct[idx])
            best_match_size = sizes_map[best_match_idx]

            min_size = min(size, best_match_size)
            intersection = ct[idx, best_match_idx]
            # intersection over min size of the two labels (to avoid bias towards bigger labels and is normalized to 1)
            io_min = intersection / min_size

            if io_min > threshold and idx != 0 and best_match_idx != 0:
                transactions[best_match_idx] = idx

        # Execute the transactions on the current slice
        new_slice = np.zeros_like(slice_2)
        for idx in sizes_map.keys():
            if idx in transactions:
                new_slice[slice_2 == idx] = transactions[idx]
            else:
                new_slice[slice_2 == idx] = idx

        stacked_segmentation[i] = new_slice

    return stacked_segmentation


class StarDist2DWrapperFor3D:
    """
    This class is a wrapper around the StarDist2D model to segment 3D images by applying the model to each z-slice
    """
    def __init__(self, model_name: str, model_additional_kwargs: dict = None):
        model_additional_kwargs = {} if model_additional_kwargs is None else model_additional_kwargs
        self.model = StarDist2D(**model_additional_kwargs).from_pretrained(model_name)

    def predict_z_instances(self, img: np.ndarray, min_size=1000, threshold=0.5):
        assert img.ndim == 3

        slice_segmentations = []
        for z_img in img:
            pred_seg, _ = self.model.predict_instances(z_img)
            pred_seg = remove_small_objects(pred_seg, min_size=min_size)
            slice_segmentations.append(pred_seg)

        return stack_2d_segmentation(slice_segmentations, threshold=threshold)


def segment_image(image: np.ndarray,
                  model: StarDist2DWrapperFor3D,
                  min_size: int = 1000, threshold: float = 0.5,
                  image_normalization_kwargs: dict = None
                  ):
    # normalize image
    image_normalization_kwargs = dict(pmax=90) if image_normalization_kwargs is None else image_normalization_kwargs
    image = normalize(image, **image_normalization_kwargs)

    # setup model and predict
    pred_seg = model.predict_z_instances(image, min_size=min_size, threshold=threshold)
    return pred_seg


def stardist2D_stacked(file_infos: dict,
                       model_name: str = '2D_versatile_fluo',
                       min_size: int = 1000,
                       threshold: float = 0.5,
                       image_normalization_kwargs: dict = None
                       ):
    """
    This function segments nuclei in 3D images by applying a 2D stardist model to each z-slice and then stacking the
    segmentations in 3D. The stacking is done by finding the best match between labels in consecutive slices.
    The process is done in 4 blocks to avoid memory issues and to follow the original plate arrangement.
    The results are saved in the same zarr file as the original image following the same plate arrangement and the
    OME-zarr labels convention.

    Args:
        file_infos (dict): dictionary containing the information about the file to segment. It should contain the keys
            'path', 'plate_to_segment', 'target_channel', and 'output_name'.
        model_name (str, optional): name of the stardist model to use. Defaults to '2D_versatile_fluo'.
        min_size (int, optional): minimum size of the objects to keep after the 2D segmentation. Defaults to 1000.
        threshold (float, optional): threshold for the overlap (overlap measured as intersection over Min).
         Defaults to 0.5.

    """
    # Load image from zarr
    assert 'path' in file_infos, "Key 'path' missing in file_infos"
    assert 'plate_to_segment' in file_infos, "Key 'plate_to_segment' missing in file_infos"
    assert 'target_channel' in file_infos, "Key 'target_channel' missing in file_infos"
    if 'output_name' not in file_infos:
        warn("Key 'output_name' missing in file_infos. Results will be saved in the default location "
             "'nuclei_stadist'.")

    image_shape = get_image_shape(file_infos['path'],
                                  key=file_infos['plate_to_segment'],
                                  resolution=0,
                                  channel=file_infos['target_channel'])

    pred_seg = np.zeros(image_shape, dtype='uint16')
    model = StarDist2DWrapperFor3D(model_name)

    # slice image in 4 blocks to avoid memory issues (and follow the original plat arrangement)
    all_slices = [(slice(None, None), slice(0, image_shape[1] // 2), slice(0, image_shape[2] // 2)),
                  (slice(None, None), slice(0, image_shape[1] // 2), slice(image_shape[2] // 2, None)),
                  (slice(None, None), slice(image_shape[1] // 2, None), slice(0, image_shape[2] // 2)),
                  (slice(None, None), slice(image_shape[1] // 2, None), slice(image_shape[2] // 2, None))]

    for i, im_slice in enumerate(all_slices):
        print(f"Processing slice {i + 1}/4")
        image = load_image_from_zarr(file_infos['path'],
                                     key=file_infos['plate_to_segment'],
                                     channel=file_infos['target_channel'],
                                     resolution=0,
                                     slices=im_slice)
        pred_seg[im_slice] = segment_image(image, model=model, min_size=min_size, threshold=threshold,
                                           image_normalization_kwargs=image_normalization_kwargs)

    # This part should also be executed in smaller blocks, but for simplicity I will keep it as is
    # Save results to zarr
    label_key = f'{file_infos["plate_to_segment"]}/labels/{file_infos.get("output_name", "nuclei_stardist")}'
    for i in range(4):
        create_image_to_zarr(pred_seg,
                             image_path=file_infos['path'],
                             key=label_key,
                             resolution=i)

        # downsample the segmentation to match the original image pyramid resolution
        # IMPORTANT: this is a simple downsample by a factor of 2 in x and y, and 1 in z
        # and is handcrafted for the specific case of the data provided in the challenge.
        # also the downsampling in this way is not ideal and can lead to artifacts.
        pred_seg = zoom(pred_seg, (1, 0.5, 0.5), order=0)
