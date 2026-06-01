import os

import numpy as np
import nrrd
from skimage.transform import rescale
from imaris_ims_file_reader.ims import ims

from oocyte_datasets import *


# Resolution 0 gives better contrast, but much slower than 1.
# Res 0 doesn't exist for later time points in 042623_hCG13h (corrupt).
#initial_res = 0

for di in TO_PROCESS:
    fn = FILE_NAMES[di]
    name = NAMES[di]
    # Expand the intensities onto this range.
    display_range = DISPLAY_RANGES[di]
    a = ims(ROOT + "Imaris/" + fn)
    a.change_resolution_lock(0)
    period = round(8.353*a.shape[2]/1000, 3)

    print(fn, display_range)
    header = {
        "kinds": 3 * ["space"],
        "spacings": 3 * [SCALE],
        "units": 3 * ["microns"],
        "n times": a.TimePoints,
        "time index": 0,
        "period": period,
        "period unit": "sec",
        "encoding": "gzip",
        "acquisition date": DATES[di],
        "label": "extracted and rescaled from " + fn,
        "group index": di,
    }

    dir_out = ROOT + "isoscale/" + name + "/"
    os.makedirs(dir_out, exist_ok=True)
    scale_ratio = np.array(a.resolution) / SCALE
    for t in range(a.TimePoints):
        path_out = dir_out + name + f"_t{t:03d}.nrrd"
        if os.path.exists(path_out):
            continue
        # All full-res time points after 152 are corrupt.
        if t == 153 and name == "042623_hCG13h":
            a.change_resolution_lock(1)
            scale_ratio = np.array(a.resolution) / SCALE
        arr = a[t, 0, :, :, :]
        arr = rescale(arr, scale_ratio, order=1, anti_aliasing=True,
                      preserve_range=True)
        arr -= display_range[0]
        arr *= 255 / (display_range[1] - display_range[0])
        np.round(arr, out=arr)
        np.clip(arr, 0, 255, out=arr)
        arr = arr.astype(np.uint8)
        header["time index"] = t
        nrrd.write(path_out,
                   arr, header=header, index_order="C")
