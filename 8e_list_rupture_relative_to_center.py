import os

import numpy as np
import torch

from oocyte_datasets import *
from sphharmodel import SphericalSurface


dir_fit = ROOT + "follicles/refit/"

torch.no_grad()
device = torch.device("cpu")
model = SphericalSurface([0, 0, 0], 1, L_MAX)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    if not follicles:
        continue

    for fol in follicles:
        if not fol.use or fol.r is None or fol.hole_t is None:
            continue
        fol_fit_dir = dir_fit + fol.label + "/"
        if not os.path.isdir(fol_fit_dir):
            continue

        model_path = fol_fit_dir + f"{fol.hole_t:03d}.pt"
        state = torch.load(model_path)
        model.load_state_dict(state)
        dz, dy, dx = np.array(fol.hole_ijk) - model.get_center()
        
        print(f"{fol.label}\txyz=({dx:.2f}, {dy:.2f}, {dz:.2f})")
