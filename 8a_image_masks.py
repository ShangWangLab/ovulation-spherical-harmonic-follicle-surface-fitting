import os

import nrrd
import numpy as np
import torch

from oocyte_datasets import *
from sphharmodel import SphericalSurface


dir_img = ROOT + "follicles/img/"
dir_fit = ROOT + "follicles/refit/"

dir_mask = ROOT + "follicles/mask/"
dir_overlap = ROOT + "follicles/overlap/"
os.makedirs(dir_mask, exist_ok=True)
os.makedirs(dir_overlap, exist_ok=True)

header_overlaps = {
    "kinds": "RGB-color space space space time".split(),
    "encoding": "gzip",
}

model = SphericalSurface([0, 0, 0], 1, L_MAX)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    print(NAMES[di], "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    for fol in follicles:
        if not fol.use or fol.r is None:
            continue
        mask_path = dir_mask + fol.label + ".nrrd"
        overlaps_path = dir_overlap + fol.label + ".nrrd"
        if (not fol.force_update
            and all(map(os.path.exists, [mask_path, overlaps_path]))):
            continue
        print(fol.label)
        stack_path = dir_img + fol.label + ".nrrd"
        if not os.path.exists(stack_path):
            continue
        # Open the 4D stack of follicle volumes.
        stack, header = nrrd.read(stack_path, index_order="C")
        nv = stack.shape[0]
        fol_fit_dir = dir_fit + fol.label + "/"
        model_paths = [fol_fit_dir + f"{it:03d}.pt" for it in range(nv)]
        if not all(map(os.path.exists, model_paths)):
            print("Missing models.")
            continue
        
        masks = np.empty_like(stack)
        overlaps = np.empty(stack.shape + (3,), np.uint8)
        for it, volume in enumerate(stack):
            print(f"{it}/{nv}")
            state = torch.load(model_paths[it])
            model.load_state_dict(state)
            mask = model.to_mask(volume)
            masks[it] = 255 * mask.astype(np.uint8)
            red = volume.copy()
            red[mask] = 0
            overlaps[it, :, :, :, 0] = red
            overlaps[it, :, :, :, 1] = volume  # green
            overlaps[it, :, :, :, 2] = volume  # blue
        nrrd.write(mask_path, masks, header, index_order="C")
        nrrd.write(overlaps_path, overlaps, header_overlaps, index_order="C")
