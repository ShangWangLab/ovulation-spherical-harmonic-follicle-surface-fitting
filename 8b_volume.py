import csv
import os

import nrrd
import numpy as np
import torch

from oocyte_datasets import *
from sphharmodel import (
    SphericalSurface,
    IntegralSurface,
)


dir_fit = ROOT + "follicles/refit/"
dir_csv = ROOT + "follicles/csv/volume/"
os.makedirs(dir_csv, exist_ok=True)

torch.no_grad()
device = torch.device("cpu")
model = SphericalSurface([0, 0, 0], 1, L_MAX)
integral = IntegralSurface(device, 32, 64)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    print(NAMES[di], "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    for fol in follicles:
        if not fol.use or fol.r is None:
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
        
        volumes = []
        for it, name in enumerate(model_names):
            if it % 50 == 0:
                print(f"{it}/{nv}")
            state = torch.load(fol_fit_dir + name)
            model.load_state_dict(state)
            volumes.append(integral.volume(model, scale=SCALE_MM))

        with open(csv_path, "w") as file:
            csv_dialect = csv.excel()
            csv_dialect.lineterminator = "\n"
            out = csv.writer(file, csv_dialect)
            out.writerow(["T (index)", "Volume (mm^3)"])
            out.writerows(list(enumerate(volumes)))
