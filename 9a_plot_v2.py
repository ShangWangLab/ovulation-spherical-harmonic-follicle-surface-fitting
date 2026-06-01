import csv
import math

import numpy as np
from matplotlib import rcParams, pyplot as plt

from oocyte_datasets import *


class AnalysisFollicle:
    def __init__(self, line):
        (ordinal, mouse_num, fol_num, scan_id, oocyte_id, name,
         delta_t, t_cross, t_open, t_release0, t_release1,
         before_deflate, t_deflate, t_v94, t_v92) = line[:15]
        self.ordinal = int(ordinal)  # Kohei's preferred figure ordering
        self.mouse_num = int(mouse_num)
        self.fol_num = int(fol_num) if fol_num else 0
        self.paper_label = "#" + mouse_num
        if fol_num:
            self.paper_label += "-" + fol_num
        self.label = name + "_" + oocyte_id
        self.delta_t = float(delta_t)
        self.nt = int(t_cross) + 1
        self.t_open = int(t_open) if t_open else 0
        self.t_release0 = int(t_release0)
        self.t_release1 = int(t_release1)
        self.has_open = before_deflate.upper() == "TRUE"
        self.t_deflate = int(t_deflate) + 1
        self.t_v94 = int(t_v94)
        self.t_v92 = int(t_v92)  # Time at which 92% of original volume is reached
        self.volumes = None
        self.pvr = None
        
        self.t = self.delta_t * (np.arange(self.nt) - self.t_open)

    def set_volumes(self, volumes):
        self.volumes = np.array(volumes)
        assert self.volumes.shape[0] == self.nt

    def set_patch_volume_ratios(self, pvr):
        self.pvr = np.array(pvr)
        pvr_pre_leak = self.pvr[:self.t_deflate]
        self.mean_volume = self.volumes[:self.t_deflate].mean()
        self.mean_pvr = pvr_pre_leak.mean()
        self.std_pvr = pvr_pre_leak.std()
        self.n_pvr = pvr_pre_leak.size
        self.se_pvr = self.std_pvr / np.sqrt(self.n_pvr)

    def pvr0(self):
        # Valid for 7 of 9.
        assert self.has_open
        # 0-50 seconds before leaking begins
        t0 = self.t_deflate - math.ceil(50 / self.delta_t)
        return np.array(self.pvr[t0:self.t_deflate])

    def pvr1(self):
        # Valid for all 9.
        # 0-14 seconds before the volume shrinks to 92% of original
        t0 = self.t_v94
        t1 = self.t_v92 + 1
        return np.array(self.pvr[t0:t1])


rcParams["font.family"] = ["Arial"]
rcParams["font.size"] = 12.0  # Default = 10.0
rcParams["figure.figsize"] = (12, 6.75)
rcParams["legend.loc"] = "lower center"
DPI = 300

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

    chart_path = "../charts/local_protrusion_index/" + afol.label + ".csv"
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


from scipy import stats
pvr0s = [f.pvr0().mean() for f in anafols if f.has_open]
pvr1s = [f.pvr1().mean() for f in anafols]
t_results = stats.ttest_ind(pvr1s, pvr0s,
                            equal_var=False,
                            alternative="two-sided")
print("Unpaired Welsh's T-test:")
print(t_results)
# t = ?
# p-value = ? (2.45%)
# DoF = ?


fig, axes = plt.subplots(3, 3)
for ax, afol in zip(axes.flat, anafols):
    t_release0 = afol.delta_t * (afol.t_release0 - afol.t_open)
    t_release1 = afol.delta_t * (afol.t_release1 - afol.t_open)
    t_deflate = afol.delta_t * (afol.t_deflate - afol.t_open)
    v_min = afol.volumes.min()
    v_max = afol.volumes.max()
    ax.plot(afol.t, afol.volumes)
    ax.axvline(t_deflate, color="k")
    ax.fill_between(afol.t, v_max, v_min,
                    where=(t_release0 <= afol.t) & (afol.t <= t_release1),
                    facecolor="k", alpha=.3)
    ax.set_title(afol.label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volume (mm^3)")
    #ax.set_ylim([0, 0.075])
    ax.grid(True)
plt.tight_layout()
#plt.suptitle("Volumes over time (separate)")
plt.savefig("../charts/volumes over time separate.png",
            dpi=DPI, bbox_inches="tight")
plt.show()


##deflation_ratio = 0.92
##fig, axes = plt.subplots(3, 3)
##for ax, afol in zip(axes.flat, anafols):
##    t_deflate = (afol.t_deflate - afol.crop_t) * afol.delta_t
##    t_hole = (afol.t_hole - afol.crop_t) * afol.delta_t
##    v_min = afol.volumes.min()
##    v_max = afol.volumes.max()
##    v_def = deflation_ratio * afol.mean_volume
##    ax.plot(afol.t, afol.volumes)
##    ax.plot([t_deflate, t_deflate], [v_min, v_max], "k-")
##    ax.plot([t_hole, t_hole], [v_min, v_max], "k-")
##    ax.plot([t_deflate, 0], [v_def, v_def], "k-")
##    #ax.plot(afol.volumes)
##    ax.set_title(afol.label)
##    ax.set_xlabel("Time (s)")
##    ax.set_ylabel("Volume (mm^3)")
##    #ax.set_ylim([0, 0.075])
##    ax.grid(True)
##plt.tight_layout()
##plt.savefig("../charts/volumes over time - deflation threshold - separate.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()
##
##fig, axes = plt.subplots(3, 3)
##for ax, afol in zip(axes.flat, anafols):
##    ax.plot(afol.volumes, label=afol.label)
##    ax.plot(afol.t_deflate, afol.volumes[afol.t_v92], "k.")
##    v_def = deflation_ratio * afol.mean_volume
##    ax.plot([0, afol.t_release], [v_def, v_def], "k-")
##    ax.set_title(afol.label)
##    ax.set_xlabel("Time (samples)")
##    ax.set_ylabel("Volume (mm^3)")
##    ax.grid(True)
##plt.savefig("../charts/volumes - deflation threshold - index.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()


##for afol in anafols:
##    plt.plot(afol.t, afol.volumes, label=afol.label)
##plt.xlabel("Time (s)")
##plt.ylabel("Volume (mm^3)")
##plt.ylim([0, 0.08])
##plt.grid(True)
##plt.legend(ncol=3)
###plt.title("Volumes over time (together)")
##plt.savefig("../charts/volumes over time together.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()
##
##for afol in anafols:
##    plt.plot(afol.volumes, label=afol.label)
##    plt.plot(afol.t_deflate, afol.volumes[afol.t_deflate], "k.")
##plt.xlabel("Time (samples)")
##plt.ylabel("Volume (mm^3)")
##plt.ylim([0, 0.08])
##plt.grid(True)
##plt.legend(ncol=3)
###plt.title("Volumes over index")
##plt.savefig("../charts/volumes over index.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()

fig, axes = plt.subplots(3, 3)
yrange = [0.8, 1.7]
for ax, afol in zip(axes.flat, anafols):
    t_release0 = afol.delta_t * (afol.t_release0 - afol.t_open)
    t_release1 = afol.delta_t * (afol.t_release1 - afol.t_open)
    t_deflate = afol.delta_t * (afol.t_deflate - afol.t_open)
    v_min = afol.pvr.min()
    v_max = afol.pvr.max()
    ax.plot(afol.t, afol.pvr)
    ax.axvline(t_deflate, color="k")
    ax.fill_between(afol.t, v_max, v_min,
                    where=(t_release0 <= afol.t) & (afol.t <= t_release1),
                    facecolor="k", alpha=.3)
    ax.set_title(afol.label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local Protrusion Index")
    ax.set_ylim(yrange)
    ax.grid(True)
plt.tight_layout()
#plt.suptitle("PVR over time (separate)")
plt.savefig("../charts/LPI over time separate.png",
            dpi=DPI, bbox_inches="tight")
plt.show()

##for afol in anafols:
##    plt.plot(afol.t, afol.pvr, label=afol.label)
##plt.xlabel("Time (s)")
##plt.ylabel("Patch Volume Ratio")
##plt.grid(True)
##plt.legend(loc="upper left", ncol=3)
###plt.title("PVR over time (together)")
##plt.savefig("../charts/PVR over time together.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()
##
##for afol in anafols:
##    plt.plot(afol.pvr[:afol.t_deflate], label=afol.label)
##plt.xlabel("Time (samples)")
##plt.ylabel("Patch Volume Ratio")
##plt.grid(True)
##plt.legend(loc="upper right", ncol=3)
###plt.title("PVR over index")
##plt.savefig("../charts/PVR over index.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()
##
##for f in anafols:
##    plt.plot(f.dt_def_rel, f.mean_pvr, ".", label=f.label, markersize=10)
##plt.xlabel("Delta T (s)")
##plt.ylabel("Mean PVR")
##plt.grid(True)
##plt.legend(loc="upper right", ncol=2)
###plt.title("Mean PVR vs. Delta T")
##plt.savefig("../charts/mean PVR v Delta T.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()

##for f in anafols:
##    t = f.dt_def_rel
##    #pvr0 = f.mean_pvr
##    #it = round(f.t_release - 30/f.delta_t)
##    #pvr1 = f.pvr[it-3:it+4].mean()
##    pvr1 = f.pvr1().mean()
##    if f.has_pvr0:
##        pvr0 = f.pvr0().mean()
##        plt.plot([t, t], [pvr0, pvr1], "^-", label=f.label, markersize=10)
##    else:
##        plt.plot(t, pvr1, "^", label=f.label, markersize=10)
##plt.xlabel("Delta T (s)")
##plt.ylabel("Mean PVR")
##plt.grid(True)
##plt.legend(loc="upper right", ncol=2)
###plt.title("Mean PVR -> 30sec before")
##plt.savefig("../charts/change in mean PVR v Delta T.png",
##            dpi=DPI, bbox_inches="tight")
##plt.show()

##for f in anafols:
##    plt.errorbar(f.dt_def_rel, f.mean_pvr, yerr=3*f.se_pvr, fmt="o", capsize=10)
##plt.xlabel("Delta T (s)")
##plt.ylabel("Mean PVR")
##plt.grid(True)
##plt.title("Mean PVR vs. Delta T, Err=3*SE")
##plt.show()


##labels0 = [f.label for f in anafols if f.has_pvr0]
##labels1 = [f.label for f in anafols]
##labels = labels0 + labels1
##pvr0s = [f.pvr0().mean() for f in anafols if f.has_pvr0]
##pvr1s = [f.pvr1().mean() for f in anafols]
##heights = pvr0s + pvr1s
##pvr0errs = [f.pvr0().std() for f in anafols if f.has_pvr0]
##pvr1errs = [f.pvr1().std() for f in anafols]
##errors = pvr0errs + pvr1errs
##colors = ["b"] * len(labels0) + ["y"] * len(labels1)
##w = 0.3
##plt.bar(labels0, pvr0s, yerr=pvr0errs, align="edge", width=-w)
##plt.bar(labels1, pvr1s, yerr=pvr1errs, align="edge", width=w)
##plt.xticks(rotation=25)
##plt.ylabel("Mean CR")
##plt.ylim([0.8, 1.4])
##plt.grid(True)
##plt.savefig("../charts/change in mean CR",
##            dpi=DPI, bbox_inches="tight")
##plt.show()


anafols.sort(key=lambda f: f.ordinal)

labels0 = [f.paper_label for f in anafols if f.has_open]
labels1 = [f.paper_label for f in anafols]
pvr0s = [f.pvr0().mean() for f in anafols if f.has_open]
pvr1s = [f.pvr1().mean() for f in anafols]
pvr0errs = [f.pvr0().std() for f in anafols if f.has_open]
pvr1errs = [f.pvr1().std() for f in anafols]
w = 0.3
plt.bar(labels0, pvr0s, yerr=pvr0errs, align="edge", width=-w)
plt.bar(labels1, pvr1s, yerr=pvr1errs, align="edge", width=w)
#plt.xticks(rotation=25)
plt.ylabel("Local Protrusion Index")
plt.ylim([0.8, 1.4])
plt.grid(True)
plt.savefig("../charts/change in mean LPI 94-92p.png",
            dpi=DPI, bbox_inches="tight")
plt.show()
