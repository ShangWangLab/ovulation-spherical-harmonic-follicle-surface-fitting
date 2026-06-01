import csv
import os

import nrrd
import numpy as np
import torch

from oocyte_datasets import *
from sphharmodel import (
    SphericalSurface,
    CircularPatch,
)


def model_load(name):
    state = torch.load(fol_fit_dir + name)
    model.load_state_dict(state)


dir_fit = ROOT + "follicles/refit/"
dir_csv = ROOT + "follicles/csv/patch_curvature/"
os.makedirs(dir_csv, exist_ok=True)

torch.no_grad()
device = torch.device("cpu")
model = SphericalSurface([0, 0, 0], 1, L_MAX)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    print(NAMES[di], "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    for fol in follicles:
        if not fol.use or fol.r is None or fol.hole_t is None:
            continue
        csv_path = dir_csv + fol.label + ".csv"
        if not fol.force_update and os.path.exists(csv_path):
            continue
        fol_fit_dir = dir_fit + fol.label + "/"
        if not os.path.isdir(fol_fit_dir):
            continue
        print(fol.label)
        dir_fol_curv = ROOT + "follicles/surf/patch/" + fol.label + "/"
        os.makedirs(dir_fol_curv, exist_ok=True)
        model_names = os.listdir(fol_fit_dir)
        model_names.sort()
        nv = len(model_names)

        model_load(model_names[fol.hole_t])
        z, y, x = np.array(fol.hole_ijk) - model.get_center()
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        patch = CircularPatch(device, theta, phi, H_RAD_MEASURE)

        radii = []
        curvatures = []
        for it, name in enumerate(model_names):
            if it % 50 == 0:
                print(f"{it}/{nv}")
            model_load(name)
            radii.append(SCALE * model.mean_radius())
            curvatures.append(patch.mean_curvature(model, scale=SCALE_MM))
            patch.make_ply(dir_fol_curv + f"{it:03d}.ply", model,
                           scale=SCALE_MM, h_range=H_RANGE, centered=True)

        with open(csv_path, "w") as file:
            csv_dialect = csv.excel()
            csv_dialect.lineterminator = "\n"
            out = csv.writer(file, csv_dialect)
            out.writerow(
                ["T (index)", "Mean Radius (um)", "Mean Curvature (mm^-1)"])
            out.writerows(list(zip(range(nv), radii, curvatures)))
