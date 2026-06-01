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
dir_csv = ROOT + "follicles/csv/local_protrusion_index/"
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
        volumes_expected = []
        volumes_actual = []
        for it, name in enumerate(model_names):
            if it % 50 == 0:
                print(f"{it}/{nv}")
            model_load(name)
            radii.append(SCALE * model.mean_radius())
            v_e = patch.subtended_volume(model, scale=SCALE_MM, const_radius=True)
            v_a = patch.subtended_volume(model, scale=SCALE_MM, const_radius=False)
            volumes_expected.append(v_e)
            volumes_actual.append(v_a)

        LPIs = [a/e for a, e in zip(volumes_actual, volumes_expected)]
        with open(csv_path, "w") as file:
            csv_dialect = csv.excel()
            csv_dialect.lineterminator = "\n"
            out = csv.writer(file, csv_dialect)
            out.writerow(
                ["T (index)", "Mean Radius (um)",
                 "Expected (mm^3)", "Actual (mm^3)",
                 "Local Protrusion Index"])
            out.writerows(list(zip(range(nv), radii, volumes_expected,
                                   volumes_actual, LPIs)))
