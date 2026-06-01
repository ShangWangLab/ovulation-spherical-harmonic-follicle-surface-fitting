This repository hosts the data processing pipeline for the follicle surface fitting method used in "Ovulation visualized in live mice". You will not be able to run this pipeline as-is without the Imaris image files. This code has been provided for reference and review purposes only; no attempt was made to facilitate use in a cross-platform environment.

Broadly, the data processing pipeline steps are as follows:
1. Preprocess the Imaris files to list metadata.
2. Resize the images to standardize the pixel spacing.
3. Register ovary images from Imaris to extract the rough center points of a manually-selected set of follicles.
4. Track those follicles through the time series.
5. Refine to find the exact centers using sub pixel alignement and extract sub volume time series surrounding those central points.
6. Use PyTorch to optimize a parametric surface to the sub volumes, based on a spherical harmonic model.
7. Re-optimize the parametric surface with the center point fixed at the center of mass.
8. Perform Miscellaneous data evaluation metrics.
9. Plot investigative data and generate data view overlays.

# Unique contents

* The data processing pipeline is labeled numerically in order of operation.
* See [metadata](metadata) for information about the follicles.
* See [oocyte_datasets.py](oocyte_datasets.py) for global follicle data and project settings.
* See [sphharmodel.py](sphharmodel.py) for details of the implementation of the spherical harmonic model and related integration functions.
* See [6_segment_follicles.py](6_segment_follicles.py) for the surface fitting procedure using PyTorch after extraction and preprocessing of the sub volume time series.

# References

The associated publication is pending review.

# Contact

Please reach out to Andre Faubert (afaubert@stevens.edu) with any questions or comments.
