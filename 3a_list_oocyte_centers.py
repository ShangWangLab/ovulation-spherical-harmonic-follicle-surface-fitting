import numpy as np

from imagejroi import load_imagej_roi_points
from oocyte_datasets import *


def get_points(name: str, time: str):
    path = f"../oocyte_coords/{name}_t{time}.roi"
    points = load_imagej_roi_points(path, sort_z=False)
    points = points[:, ::-1]  # xyz -> zyx
    points = points.astype(np.int32)
    return points


for di in TO_PROCESS:
    name = NAMES[di]
    a = get_points(name, "000")
    b = get_points(name, "end")
    print(name, "000 -> end")
    print(a.shape[0], "->", b.shape[0], "points.")
    if a.shape[0] != b.shape[0]:
        print(a)
        print(b)
        continue
    #dh = a[:, 1] - b[:, 1]  # Where h is -y.
    #mean_y_dev = (b[:, 1] - a[:, 1]).mean()
    #c = b - (b - a).mean(axis=0, keepaxes=True)
    for i in range(a.shape[0]):
        #significant = dh[i] > 10
        print("{:2d} [{:3d} {:3d} {:3d}] -> [{:3d} {:3d} {:3d}]".format(
            i+1, *a[i, :], *b[i, :]))
   
