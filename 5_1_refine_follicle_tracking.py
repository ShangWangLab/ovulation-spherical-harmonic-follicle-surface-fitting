import os

import nrrd
import numpy as np
from numpy.fft import fft, ifft

from oocyte_datasets import *


def struct_path(name, t):
    return ROOT + "isoscale/" + name + "/" + name + f"_t{t:03d}.nrrd"


def load_struct(name, t):
    p = struct_path(name, t)
    if not os.path.exists(p):
        return None
    return nrrd.read(p, index_order="C")[0]


class WindowFunction:
    def __init__(self, size=SUB_SIZE):
        self.size = size
        self.w = np.hanning(size) ** 0.5

    def __call__(self, volume):
        return (volume
                * self.w[:, None, None]
                * self.w[None, :, None]
                * self.w[None, None, :])


def get_shift(fixed, moving):
    # Return a vector pointing from fixed onto moving.
    shape = fixed.shape
    # Both images should already be in the Fourier domain.
    R = fixed * moving.conjugate()
    R /= abs(R)
    r = abs(ifft3d(R))
    shift = np.asarray(np.unravel_index(np.argmax(r), shape))
    for i, v in enumerate(shift):
        if v >= shape[i] // 2:
            shift[i] -= shape[i]
    return shift


def fft3d(img):
    for i in range(3):
        img = fft(img, axis=i)
    return img


def ifft3d(img):
    for i in range(3):
        img = ifft(img, axis=i)
    return img


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
    arr_dst[:] = 0
    arr_dst[*dst] = arr_src[*src]


W = SUB_SIZE
R = W//2
win_func = WindowFunction()

dir_in = ROOT + "follicles/tracking/"
dir_out = ROOT + "follicles/tracking2/"
os.makedirs(dir_out, exist_ok=True)
for di in TO_PROCESS:
    name = NAMES[di]
    follicles = get_scan_follicles(di)
    print(name, "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    tracks = []
    tracked_follicles = []
    for f in follicles:
        path_in = dir_in + f.label + ".npy"
        path_out = dir_out + f.label + ".npy"
        if os.path.exists(path_in) and not os.path.exists(path_out):
            tracks.append(np.load(path_in))
            tracked_follicles.append(f)
    if not tracks:
        print("No tracked follicles need extraction.")
        continue

    nt = tracks[0].shape[0]

    img_tmp = np.empty(3*(SUB_SIZE,), dtype=np.float32)
    imgs_A = []
    struct0 = load_struct(name, 0)
    for track in tracks:
        assign_centered_at(img_tmp, struct0, track[0])
        imgs_A.append(fft3d(win_func(img_tmp)))
    for it in range(1, nt):
        if it % 100 == 0:
            print(it)
        struct = load_struct(name, it)
        if struct is None:
            # This can occur if a follicle releases too near the end
            # or if files are missing.
            print(f"Truncated {name} unexpectedly as t={it}.")
            break
        # Two stages: align to the previous example, then align that example
        # to the first example. This minimizes drift while retaining similarity
        # between adjacent time points.
        imgs_prev = [v for v in imgs_A]
        for i, track in enumerate(tracks):
            img_A = imgs_A[i]
            img_P = imgs_prev[i]
            assign_centered_at(img_tmp, struct, track[it])
            img_B = fft3d(win_func(img_tmp))
            shift = get_shift(img_P, img_B)
            track[it] -= shift
            assign_centered_at(img_tmp, struct, track[it])
            img_B = fft3d(win_func(img_tmp))
            shift = get_shift(img_A, img_B)
            track[it] -= shift
            imgs_prev[i] = img_B
            
    for track, f in zip(tracks, tracked_follicles):
        np.save(dir_out + f.label + ".npy", track)

