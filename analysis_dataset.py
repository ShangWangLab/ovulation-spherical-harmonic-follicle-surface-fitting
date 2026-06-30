import math

import numpy as np


class AnalysisFollicle:
    def __init__(self, line):
        (ordinal, mouse_num, fol_num, scan_id, oocyte_id, name,
         delta_t, t_cross, t_open, t_release0, t_release1,
         t_deflate, t_v94, t_v92) = line[:15]
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
        self.has_open = bool(t_open) # before_deflate.upper() == "TRUE"
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
        assert self.has_open
        # Up to 50 seconds before opening.
        t0 = max(self.t_open - math.ceil(50 / self.delta_t), 0)
        return np.array(self.pvr[t0:self.t_open])

    def pvr1(self):
        # Valid for all follicles.
        # Between open and release release.
        #t0 = max(self.t_release0 - round(50 / self.delta_t), 0)
        #t1 = self.t_release0 - math.floor(30 / self.delta_t)
        return np.array(self.pvr[self.t_open:self.t_release0])

    def pvr0_old(self):
        # Valid for 7 of 9.
        assert self.has_open
        # 0-50 seconds before leaking begins
        t0 = self.t_deflate - math.ceil(50 / self.delta_t)
        return np.array(self.pvr[t0:self.t_deflate])

    def pvr1_old(self):
        # Valid for all 9.
        # 0-14 seconds before the volume shrinks to 92% of original
        t0 = self.t_v94
        t1 = self.t_v92 + 1
        return np.array(self.pvr[t0:t1])


def _LOAD_DATASETS():
    import csv
    
    with open("../follicles_data_analysis.csv", "r") as file:
        cf = csv.reader(file)
        header = next(cf)
        for line in cf:
            if not line or not line[0]:
                break
            ANAFOLS.append(AnalysisFollicle(line))

    for afol in ANAFOLS:
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


ANAFOLS = []
_LOAD_DATASETS()
