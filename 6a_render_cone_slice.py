import csv
import os

import nrrd
import numpy as np
from scipy import ndimage as ndi
from imaris_ims_file_reader.ims import ims
from PIL import Image

from oocyte_datasets import *


out_width = 800  # microns
# For original concept of measuring at time of hole visibility:
##time_indices = {
##    '031124_hCG12h_8':   (120, 206),
##    '042623_hCG13h_10':  (7, 58),
##    '042623_hCG13p5h_6': (0, 2),
##    '042623_hCG13p5h_9': (91, 160),
##    '050825_hCG11h_4':   (89, 141),
##    '051623_hCG12h_6':   (13, 89),
##    '070122_Ovary_2':    (21, 193),
##    '090523_hCG13p5h_7': (187, 241),
##    '091223_hCG12h_0':   (3, 21),
##}
# For new concept of 92% of original volume:
time_indices = {
    '031124_hCG12h_8':   (185,),
    '042623_hCG13h_10':  (36,),
    '042623_hCG13p5h_6': (8,),
    '042623_hCG13p5h_9': (162,),
    '050825_hCG11h_4':   (148,),
    '051623_hCG12h_6':   (55,),
    '070122_Ovary_2':    (95,),
    '090523_hCG13p5h_7': (248,),
    '091223_hCG12h_0':   (18,),
}


dir_fol_csv = ROOT + "follicles/csv/fit/"
dir_track = ROOT + "follicles/tracking2/"
dir_fol_section = ROOT + "follicles/section_image_92p/"
os.makedirs(dir_fol_section, exist_ok=True)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    follicles = [fol for fol in follicles if fol.use]
    print(NAMES[di], "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    volumes = ims(ROOT + "Imaris/" + FILE_NAMES[di])
    volumes.change_resolution_lock(0)
    img_res = np.array(volumes.resolution)  # microns
    out_res = min(img_res)
    display_range = DISPLAY_RANGES[di]

    for fol in follicles:
        print(fol.label)
        csv_path = dir_fol_csv + fol.label + ".csv"
        track_path = dir_track + fol.label + ".npy"
        if not (os.path.exists(csv_path) and os.path.exists(track_path)):
            continue

        times = time_indices[fol.label]
        #it = fol.hole_t  #fol.t1

        for iit, it in enumerate(times):
            section_path = dir_fol_section + fol.label + f"_{iit}.png"
            if not fol.force_update and os.path.exists(section_path):
                continue
             
            with open(csv_path, "r") as file:  
                cf = csv.reader(file)
                for _ in range(it + 1):
                    next(cf)
                line = next(cf)
                center_ijk = [float(v) for v in line[4:7]]
            track_ijk = np.load(track_path)[it]

            # Forming the vector space:
            img_center_ijk = track_ijk - SUB_SIZE//2 + center_ijk
            v_cen = np.array(center_ijk)
            v_hol = np.array(fol.hole_ijk)
            v1 = v_hol - v_cen
            v1 /= np.sqrt(np.sum(v1**2))

            #v2 = np.cross([0, 0, 1], v1)
            #v0 = np.cross(v1, v2)

            v0 = np.array([
                v1[2] * v1[0],
                v1[2] * v1[1],
                -(v1[1]**2 + v1[0]**2)
            ])
            v0 /= np.sqrt(np.sum(v0**2))

            # normalized -> px/micron
            v0 /= img_res
            v1 /= img_res
            img_center_ijk *= SCALE / img_res

            # Sample from Imaris image.
            wi = round(out_width / out_res)
            t = out_res * (np.arange(wi + 1) - wi/2)
            t1, t0 = np.meshgrid(t, t)
            # 3 x n x n
            s = (v0[:, None, None] * t0[None, :, :]
                 + v1[:, None, None] * t1[None, :, :])
            s += img_center_ijk[:, None, None]

            volume = volumes[it, 0, :, :, :]
            out = ndi.map_coordinates(volume, s, order=2,
                                      mode="constant", cval=0)
            out = out.astype(np.float32)
            out -= display_range[0]
            out *= 255 / (display_range[1] - display_range[0])
            np.round(out, out=out)
            np.clip(out, 0, 255, out=out)
            out = out.astype(np.uint8)
            img_out = Image.fromarray(out)
            img_out.save(section_path)
