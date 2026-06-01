import os

import ants
import numpy as np
import pandas as pd

from oocyte_datasets import *


dir_out = ROOT + "follicles/tracking/"
os.makedirs(dir_out, exist_ok=True)
for di in TO_PROCESS:
    name = NAMES[di]
    follicles = get_scan_follicles(di)
    print(name, "tracking", len(follicles), "follicles.")
    if not follicles:
        continue
    if all(os.path.exists(dir_out + f.label + ".npy") for f in follicles):
        print("Already processed.")
        continue
    dir_register = ROOT + "register/" + name + "/"

    # ANTs maps NumPy[i, j, k] -> xyz
    pts = pd.DataFrame(data={
        "x": [fol.x for fol in follicles],
        "y": [fol.y for fol in follicles],
        "z": [fol.z for fol in follicles]})
    tracking = [[(fol.z, fol.y, fol.x)] for fol in follicles]
    it = 1
    while os.path.exists((path := dir_register + f"{name}_{it:03d}") + ".mat"):
        if it % 50 == 0:
            print(it)
        txfwd = [path + ".nii.gz", path + ".mat"]
        ptsw = ants.apply_transforms_to_points(3, pts, txfwd)
        xyz = ptsw.values.round().astype(int)
        for i, track in enumerate(tracking):
            track.append(xyz[i][::-1])
        it += 1
    for track, f in zip(tracking, follicles):
        np.save(dir_out + f.label + ".npy", np.array(track, np.int32))
