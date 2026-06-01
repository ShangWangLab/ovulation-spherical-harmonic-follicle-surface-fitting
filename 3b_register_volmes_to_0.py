import os

import ants
import nrrd
import numpy as np

from oocyte_datasets import *


def struct_path(name, t):
    return ROOT + "isoscale/" + name + "/" + name + f"_t{t:03d}.nrrd"


def load_struct(name, t):
    p = struct_path(name, t)
    assert os.path.exists(p), f"Missing struct file {name} time {t}"
    return nrrd.read(p, index_order="C")[0]


def move_file(src, dst):
    # You can't move a file between drives with an OS call.
    with open(src, "rb") as fin:
        with open(dst, "wb") as fout:
            fout.write(fin.read())
    os.remove(src)


for di in TO_PROCESS:
    name = NAMES[di]
    print(name)
    dir_out = ROOT + "register/" + name + "/"
    os.makedirs(dir_out, exist_ok=True)
    struct0np = load_struct(name, 0)
    struct0 = ants.from_numpy(struct0np)
    n_skipped = 0
    it = 1
    while os.path.exists(struct_path(name, it)):
        name_out = f"{name}_{it:03d}"
        affine_path = dir_out + name_out + ".mat"
        if os.path.exists(affine_path):
            # Skipping existing transform.
            n_skipped += 1
            it += 1
            continue
        print(it)
        struct1 = load_struct(name, it)
        struct1 = ants.from_numpy(struct1)
        # Generates maps which project points from struct0's space onto struct1's.
        tx = ants.registration(fixed=struct0, moving=struct1, type_of_transform="SyN")
        # Temp directory is on the C: drive. Transfer the files between drives.
        move_file(tx["fwdtransforms"][0], dir_out + name_out + ".nii.gz")
        move_file(tx["fwdtransforms"][1], affine_path)
        # The inverse transform is not necessary.
        os.remove(tx["invtransforms"][1])  
        #struct0 = struct1  # Uncomment for adjacent registrations instead of global.
        it += 1
    print(f"Registration complete! Skipped {n_skipped} existing registrations.")
