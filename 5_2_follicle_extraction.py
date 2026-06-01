import os

import nrrd
import numpy as np

from oocyte_datasets import *


def struct_path(name, t):
    return ROOT + "isoscale/" + name + "/" + name + f"_t{t:03d}.nrrd"


def load_struct(name, t):
    p = struct_path(name, t)
    if not os.path.exists(p):
        return None
    return nrrd.read(p, index_order="C")[0]


def ijk2slice(p):
    """Turns a point into a list of slices with radius `r` centered around
    that point."""
    return [slice(max(0, v - R), v + R + 1) for v in p]


def assign_centered_at(arr_dst, arr_src, p):
    """Assign a arr0 to be a block from arr1 centered around p, allowing for
    mismatching sizes."""
    d = arr_src.shape
    src = ijk2slice(p)
    dst = [slice(max(0, R - v),
                 min(W, d[i] - v + W - R - 1))
          for i, v in enumerate(p)]
    arr_dst[*dst] = arr_src[*src]


W = SUB_SIZE
R = SUB_SIZE//2

dir_out = ROOT + "follicles/img/"
os.makedirs(dir_out, exist_ok=True)
dir_tracking = ROOT + "follicles/tracking2/"
for di in TO_PROCESS:
    name = NAMES[di]
    follicles = get_scan_follicles(di)
    print(name, "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    tracks = []
    tracked_follicles = []
    for f in follicles:
        path_in = dir_tracking + f.label + ".npy"
        path_out = dir_out + f.label + ".nrrd"
        if os.path.exists(path_in) and not os.path.exists(path_out):
            tracks.append(np.load(path_in))
            tracked_follicles.append(f)
    if not tracks:
        print("No tracked follicles need extraction.")
        continue

    nt = max(fol.t1 for fol in follicles) + 1
    img_out = [np.zeros((fol.t1 + 1, SUB_SIZE, SUB_SIZE, SUB_SIZE),
                        dtype=np.uint8)
               for fol in follicles]
    for it in range(nt):
        if it % 100 == 0:
            print(it)
        struct = load_struct(name, it)
        if struct is None:
            # This can occur if a follicle releases too near the end
            # or if files are missing.
            print(f"Truncated {name} unexpectedly as t={it}.")
            break
        for i, (img, track) in enumerate(zip(img_out, tracks)):
            if it < img.shape[0]:
                assign_centered_at(img[it], struct, track[it])
    for img, f in zip(img_out, tracked_follicles):
        vertical_pattern = img.mean(axis=(1, 3), keepdims=True)
        imgr = img - vertical_pattern
        imgr -= imgr.min()
        imgr *= 255 / imgr.max()
        imgr = imgr.astype(np.uint8)
        nrrd.write(dir_out + f.label + ".nrrd",
                   imgr,
                   header={"kinds": "space space space time".split()},
                   index_order="C")
