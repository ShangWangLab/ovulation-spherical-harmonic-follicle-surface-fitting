import numpy as np
from matplotlib import rcParams, pyplot as plt

from analysis_dataset import *


rcParams["font.family"] = ["Arial"]
rcParams["font.size"] = 12.0  # Default = 10.0
rcParams["figure.figsize"] = (12, 6.75)
rcParams["legend.loc"] = "lower center"
DPI = 300


times = {}
for f in ANAFOLS:
    t0 = max(0, f.t_open - round(25 / f.delta_t))
    t1 = math.ceil((f.t_open + f.t_release0) / 2)
    times[f.label] = (t0, t1)
print([t for label, t in sorted(times.items())])
for label, t in times.items():
    print(label, "\t:", t[0], "and", t[1])


from scipy import stats
pvr0s = [f.pvr0().mean() for f in ANAFOLS if f.has_open]
pvr1s = [f.pvr1().mean() for f in ANAFOLS]
t_results = stats.ttest_ind(pvr1s, pvr0s,
                            equal_var=False,
                            alternative="two-sided")
print("Unpaired Welsh's T-test:")
print("Low mean:", np.mean(pvr0s))
print("High mean:", np.mean(pvr1s))
print(t_results)
# t = 0.47
# p-value = 65%
# DoF = 14-15


fig, axes = plt.subplots(3, 3)
for ax, afol in zip(axes.flat, ANAFOLS):
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


fig, axes = plt.subplots(3, 3)
yrange = [0.8, 1.4]
for ax, afol in zip(axes.flat, ANAFOLS):
    t_release0 = afol.delta_t * (afol.t_release0 - afol.t_open)
    t_release1 = afol.delta_t * (afol.t_release1 - afol.t_open)
    t_deflate = afol.delta_t * (afol.t_deflate - afol.t_open)
    v_min = afol.pvr.min()
    v_max = afol.pvr.max()
    ax.plot(afol.t, afol.pvr)
    ax.axvline(t_deflate, color="k")
    ax.axhline(1.1, color="g", ls="--")
    ax.axhline(1.2, color="r", ls="--")
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


ANAFOLS.sort(key=lambda f: f.ordinal)

labels0 = [f.paper_label for f in ANAFOLS if f.has_open]
labels1 = [f.paper_label for f in ANAFOLS if f.has_open]
pvr0s = [f.pvr0().mean() for f in ANAFOLS if f.has_open]
pvr1s = [f.pvr1().mean() for f in ANAFOLS if f.has_open]
pvr0errs = [f.pvr0().std() for f in ANAFOLS if f.has_open]
pvr1errs = [f.pvr1().std() for f in ANAFOLS if f.has_open]
w = 0.3
plt.bar(labels0, pvr0s, yerr=pvr0errs, align="edge", width=-w)
plt.bar(labels1, pvr1s, yerr=pvr1errs, align="edge", width=w)
#plt.xticks(rotation=25)
plt.ylabel("Local Protrusion Index")
plt.ylim([0.8, 1.4])
plt.grid(True)
plt.savefig("../charts/change in mean LPI.png",
            dpi=DPI, bbox_inches="tight")
plt.show()
