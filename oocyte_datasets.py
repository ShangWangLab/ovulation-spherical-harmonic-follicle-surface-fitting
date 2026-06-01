from math import pi as PI


class Follicle:
    def __init__(self, i_scan, i_oocyte, z, y, x, t0, t1,
                 use=False, d=None, hole_ijk=None, hole_t=None,
                 force_update=False):
        self.i_scan = i_scan
        self.i_oocyte = i_oocyte
        self.z = z
        self.y = y
        self.x = x
        self.t0 = t0
        self.t1 = t1
        self.use = use
        self.r = d/2 if d else None
        self.hole_ijk = hole_ijk
        self.hole_t = hole_t
        self.label = NAMES[i_scan] + "_" + str(i_oocyte)
        self.force_update = force_update

    def __repr__(self):
        name = "Follicle" if self.use else "Exclude"
        return f"{name} {self.i_scan}-{self.i_oocyte} xyzt=<{self.x}, {self.y}, {self.z}, {self.t0}-{self.t1}>"

    @staticmethod
    def parse_line(line):
        i_scan = int(line[0])
        i_oocyte = int(line[1])
        assert NAMES[i_scan] == line[2], "Follicle ID doesn't match its name."
        use = line[3].upper() == "TRUE"
        z, y, x, t0, t1 = map(int, line[4:9])
        if len(line) >= 9:
            d = line[9]
            d = float(d) if d else None
            hole_ijk = [(float(v) if v else None) for v in line[10:13]]
            hole_t = int(line[13]) if line[13] else None
        else:
            d = None
            hole_ijk = None
            hole_t = None
        force_update = line[14].upper() == "TRUE"
        return Follicle(i_scan, i_oocyte, z, y, x, t0, t1,
                        use, d=d, hole_ijk=hole_ijk, hole_t=hole_t,
                        force_update=force_update)


def get_scan_follicles(i: int):
    return [fol for fol in FOLLICLES if fol.i_scan == i]


def _LOAD_DATASETS():
    import csv
    
    with open("../datasets.csv", "r") as file:
        cf = csv.reader(file)
        header = next(cf)
        for file_name, name, date, lo, hi in cf:
            NAMES.append(name)
            FILE_NAMES.append(file_name)
            DATES.append(date)
            DISPLAY_RANGES.append((int(lo), int(hi)))

    with open("../follicles.csv", "r") as file:
        cf = csv.reader(file)
        header = next(cf)
        for line in cf:
            if not line or not line[0]:
                continue
            FOLLICLES.append(Follicle.parse_line(line))


H_RANGE = 15  # Curvature range to display (mm^-1)
# Half the subtended angle (radians) of conical region to measure.
# This is 40.4 degrees (1.5 steradians).
# Originally 36 degrees (1.2 steradians).
# solid angle = 2 * PI * (1 - cos(H_RAD_MEASURE)) steradians
H_RAD_MEASURE = 0.7055306192083447  # radians
L_MAX = 10  # Normally 9 or 10. The highest order of spherical harmonic model to fit
#T_POST_HOLE = 10  # How many time points after hole formation to keep
SCALE = 25.  # microns/pixel
SCALE_MM = SCALE / 1000
SUB_SIZE = 39  # pixels in each dimension
ROOT = "E:/oocyte_curvature/"
NAMES = []
FILE_NAMES = []
DATES = []
DISPLAY_RANGES = []
# Which items from the dataset to run the pipeline on.
TO_PROCESS = list(range(14))
FOLLICLES = []
_LOAD_DATASETS()
