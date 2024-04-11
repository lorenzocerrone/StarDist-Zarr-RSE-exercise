# StarDist-Zarr-RSE-exercise-
Take home exercise for the RSE position

## Task details
* Use stardist to segment the nuclei in the DAPI channel of the provided zarr file (https://zenodo.org/records/10257532).
* (Optional) Try also alternative methods for segmentation (e.g. PlantSeg)
* Write a Python script that performs the segmentation and saves the results as a zarr file.
* The script should be configurable via config file (yaml or json).
* All relevant parameters should be configurable via the config file.
* The project should be pip installable.

## Main Objectives
* Release a clear and usable python package, with a nice API.
* The package should work for OME-Zarr files.
* Support for multi-resolution data.

