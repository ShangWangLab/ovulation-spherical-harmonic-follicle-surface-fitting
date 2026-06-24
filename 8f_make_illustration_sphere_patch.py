import os

import numpy as np
import torch

from oocyte_datasets import *
from sphharmodel import (
    SphericalSurface,
    CircularPatch,
)


dir_fit = ROOT + "follicles/refit/"
dir_surf = ROOT + "follicles/surf/"
di = 2
for fol in FOLLICLES:
    if fol.i_scan == 2 and fol.i_oocyte == 6:
        break
fol_fit_dir = dir_fit + fol.label + "/"
assert os.path.isdir(fol_fit_dir)

torch.no_grad()
device = torch.device("cpu")
model = SphericalSurface([0, 0, 0], 1, L_MAX)
state = torch.load(fol_fit_dir + "000.pt")
model.load_state_dict(state)
z, y, x = np.array(fol.hole_ijk) - model.get_center()
r = np.sqrt(x*x + y*y + z*z)
theta = np.arccos(z / r)
phi = np.arctan2(y, x)
patch = CircularPatch(device, theta, phi, H_RAD_MEASURE, n_rho=32, n_psi=64)

R = model.radius(patch)
R_ring_mean = R[(patch.n_rho - 1)*patch.n_psi:patch.n_rho*patch.n_psi].mean()
model_sphere = SphericalSurface(model.get_center(), R_ring_mean, L_MAX)

patch.make_ply(dir_surf + "illustration_patch.ply",
    model, scale=SCALE_MM, h_range=H_RANGE, centered=True)
patch.make_ply(dir_surf + "illustration_sphere_patch.ply",
    model_sphere, scale=SCALE_MM, h_range=H_RANGE, centered=True)
