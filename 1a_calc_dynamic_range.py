import numpy as np
from imaris_ims_file_reader.ims import ims

from oocyte_datasets import *


lower = 0.1
upper = 0.99

for di in TO_PROCESS:
    fn = FILE_NAMES[di]
    a = ims(ROOT + "Imaris/" + fn)
    arr = a[0, 0, :, :, :]
    hist, _ = np.histogram(arr, 256, (0, 255))
    ch = np.cumsum(hist)

    a = np.searchsorted(ch, 0.1*arr.size, "left")
    b = np.searchsorted(ch, 0.99*arr.size, "right")

    print(NAMES[di], a, b, sep="\t")
