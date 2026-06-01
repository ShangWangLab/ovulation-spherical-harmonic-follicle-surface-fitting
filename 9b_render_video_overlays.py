import csv
import math
import os
import subprocess

import numpy as np
from matplotlib import rcParams, pyplot as plt
from matplotlib.patches import Rectangle

from oocyte_datasets import *


class AnalysisFollicle:
    def __init__(self, line):
        (scan_id, oocyte_id, name, t_hole, t_release, crop_t, delta_t,
         t_deflate, before_deflate) = line[:9]
        self.label = name + "_" + oocyte_id
        self.t_hole = int(t_hole)
        self.t_release = int(t_release)
        self.t_deflate = int(t_deflate) + 1
        self.crop_t = int(crop_t) if crop_t else self.t_release
        self.delta_t = float(delta_t)
        self.volumes = None
        self.pvr = None
        self.has_pvr0 = before_deflate.upper() == "TRUE"
        
        self.t = (np.arange(1, self.crop_t + 1) - self.t_release) * self.delta_t
        self.dt_def_rel = (self.t_release - self.t_deflate) * self.delta_t

    def set_volumes(self, volumes):
        self.volumes = np.array(volumes[:self.crop_t])

    def set_patch_volume_ratios(self, pvr):
        self.pvr = np.array(pvr[:self.crop_t])
        pvr_pre_leak = self.pvr[:self.t_deflate]
        self.mean_pvr = pvr_pre_leak.mean()
        self.std_pvr = pvr_pre_leak.std()
        self.n_pvr = pvr_pre_leak.size
        self.se_pvr = self.std_pvr / np.sqrt(self.n_pvr)


    def pvr0(self):
        # Valid for 7 of 9.
        # 0-50 seconds before leaking begins
        t0 = self.t_deflate - math.ceil(50 / self.delta_t)
        return np.array(self.pvr[t0:self.t_deflate])

    def pvr1(self):
        # Valid for all 9.
        # 40-54 seconds before oocyte leaves
        t0 = self.t_release - math.ceil(54 / self.delta_t)
        t1 = self.t_release - math.floor(40 / self.delta_t)
        return np.array(self.pvr[t0:t1])


rcParams["font.family"] = ["Arial"]
rcParams["font.size"] = 12.0  # Default = 10.0
WIDTH, HEIGHT, DPI = 1920, 1080, 240  # = 8x4.5"

anafols = []
with open("../follicles_data_analysis.csv", "r") as file:
    cf = csv.reader(file)
    header = next(cf)
    for line in cf:
        if not line or not line[0]:
            break
        anafols.append(AnalysisFollicle(line))

for afol in anafols:
    chart_path = "../charts/volume/" + afol.label + ".csv"
    print(chart_path)
    volumes = []
    with open(chart_path, "r") as file:
        cf = csv.reader(file)
        header = next(cf)
        for line in cf:
            if not line or not line[0]:
                break
            volumes.append(float(line[1]))
    afol.set_volumes(volumes)

    chart_path = "../charts/patch_volume/" + afol.label + ".csv"
    print(chart_path)
    patch_volume_ratios = []
    with open(chart_path, "r") as file:
        cf = csv.reader(file)
        header = next(cf)
        for line in cf:
            if not line or not line[0]:
                break
            patch_volume_ratios.append(float(line[4]))
    afol.set_patch_volume_ratios(patch_volume_ratios)



def render_frame(dir_path, it, fol):
    fig = plt.figure(
        figsize=(WIDTH / DPI, HEIGHT / DPI),
        dpi=DPI,
        facecolor=(0, 0, 0, 0)  # transparent figure background
    )

    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # origin at top-left for easier overlay positioning
    ax.axis("off")
    ax.set_facecolor((0, 0, 0, 0))

    # ============================================================
    # LEFT COLOR BAR
    # ============================================================

    curvature_scale = 15.
    colorbar_w = 40
    colorbar_h = 620
    colorbar_x = 60
    colorbar_y = (1080 - colorbar_h) // 2

    gradient = np.linspace(1, 0, 512).reshape(-1, 1)

    ax.imshow(
        gradient,
        cmap="coolwarm",
        aspect="auto",
        extent=(
            colorbar_x,
            colorbar_x + colorbar_w,
            colorbar_y + colorbar_h,
            colorbar_y
        )
    )

    # Border
    tick_width = 1.
    ax.add_patch(Rectangle(
        (colorbar_x, colorbar_y),
        colorbar_w,
        colorbar_h,
        linewidth=tick_width,
        edgecolor="white",
        facecolor="none"
    ))

    # Tick labels
    n_ticks = 7
    ticks = np.linspace(-curvature_scale, curvature_scale, n_ticks)
    for i, val in enumerate(ticks):
        y = colorbar_y + colorbar_h - i * (colorbar_h / (n_ticks - 1))

        ax.plot(
            [colorbar_x + colorbar_w, colorbar_x + colorbar_w + 10],
            [y, y],
            color="white",
            linewidth=tick_width
        )

        ax.text(
            colorbar_x + colorbar_w + 23,
            y,
            f"{val:+2.0f}" if val != 0 else " 0",
            color="white",
            fontsize=12,
            family="monospace",
            va="center",
            ha="left"
        )

    # Label
    ax.text(
        colorbar_x,
        colorbar_y - 50,
        "Curvature (mm",
        color="white",
        fontsize=14,
        ha="left",
        va="bottom"
    )
    ax.text(
        colorbar_x + 332,
        colorbar_y - 50,
        ")",
        color="white",
        fontsize=14,
        ha="left",
        va="bottom"
    )
    # Superscript -1
    ax.text(
        colorbar_x + 304,
        colorbar_y - 80,
        "-1",
        color="white",
        fontsize=10,
        ha="left",
        va="bottom"
    )

    # ============================================================
    # TIMESTAMP (UPPER RIGHT)
    # ============================================================

    phys_time = fol.delta_t * (fol.t_release - 1 - it)
    minutes = int(phys_time / 60)
    seconds = round(phys_time % 60)
    timestamp = f"-{minutes:02d}:{seconds:02d}"

    ax.text(
        WIDTH - 70,
        70,
        timestamp,
        color="white",
        fontsize=18,
        ha="right",
        va="top",
        family="monospace",
        bbox=dict(
            facecolor=(0, 0, 0, 0.5),
            edgecolor="none",
            boxstyle="round,pad=0.2"
        )
    )

    # ============================================================
    # SCALE BAR (LOWER LEFT)
    # ============================================================

    scale_phys = 250  # microns
    frame_width_phys = 1300  # microns
    scale_x = 80
    scale_y = HEIGHT - 100

    scale_bar_px = WIDTH * scale_phys / frame_width_phys
    scale_bar_h = 20

    # Main bar
    ax.add_patch(Rectangle(
        (scale_x, scale_y),
        scale_bar_px,
        scale_bar_h,
        facecolor="white",
        edgecolor="none"
    ))

    # Label
    ax.text(
        scale_x + scale_bar_px / 2,
        scale_y - 18,
        f"{scale_phys} µm",
        color="white",
        fontsize=14,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="none", edgecolor="none")
    )

    # Coneness
    cone_ratio = fol.pvr[it]
    ax.text(
        WIDTH - 70,
        HEIGHT - 81,
        f"Cone ratio: {cone_ratio:.2f}",
        color="white",
        fontsize=14,
        fontfamily="monospace",
        ha="right",
        va="bottom"
    )

    # ============================================================
    # EXPORT
    # ============================================================

    plt.savefig(
        os.path.join(dir_path, f"{it:03d}.png"),
        transparent=True,
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)


for fol in anafols:
    print(fol.label)
    dir_path = "../render/overlays/" + fol.label
    os.makedirs(dir_path, exist_ok=True)
    for it in range(fol.pvr.size):
        render_frame(dir_path, it, fol)


    video_path = "../render/blender/" + fol.label + ".mp4"
    png_sequence = dir_path + "/%03d.png"
    output_path = "../render/overlaid/" + fol.label + ".mp4"

    cmd = [
        "ffmpeg",
        "-i", video_path,  # Input video
        
        "-framerate", "4",   # Match your PNG sequence frame rate
        "-start_number", "0",
        "-i", png_sequence,  # PNG sequence

        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto:eof_action=endall",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    # Run FFmpeg
    subprocess.run(cmd, check=True)
