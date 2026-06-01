from imaris_ims_file_reader.ims import ims

from oocyte_datasets import *


for di in TO_PROCESS:
    fn = FILE_NAMES[di]
    a = ims(ROOT + "Imaris/" + fn)
    nt, nc, nz, ny, nx = a.shape
    sz, sy, sx = a.resolution
    print(NAMES[di], nx, ny, nz, nt, sx, sy, sz, sep="\t")
