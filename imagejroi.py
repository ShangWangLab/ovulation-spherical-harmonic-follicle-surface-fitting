from zipfile import ZipFile
import struct

import numpy as np
import numpy.typing as npt


def load_imagej_roi_lines(path: str) -> npt.NDArray[np.float32]:
    z_values = []
    lines = []
    with ZipFile(path, "r") as archive:
        members = archive.namelist()
        n_members = len(members)
        z_values = np.empty((n_members,), np.int64)
        lines = np.empty((n_members, 2, 2), np.float32)
        for i, member in enumerate(members):
            with archive.open(member, "r") as file:
                data = file.read()
                z_values[i], lines[i, :, :] = load_imagej_roi_line(data)
    return z_values, lines


def load_imagej_roi_line(data: bytes) -> npt.NDArray[np.float32]:
    assert len(data) >= 0x22, "Data is too short"
    assert data[:4] == b"Iout", "Incorrect magic number"
    assert data[6] == 3, "Data is not a line ROI"

    values = struct.unpack(">ffff", data[0x12:0x22])
    z, = struct.unpack(">i", data[0x38:0x3C])
    z -= 1  # ImageJ Z slices use 1-based indexing.
    return z, np.array(values, np.float32).reshape((2, 2))


def load_imagej_roi_points(path: str, sort_z=True) -> npt.NDArray[np.float32]:
    with open(path, "rb") as f:
        data = f.read()

    n_coords = struct.unpack(">h", data[0x10:0x12])[0]
    header2_offset = struct.unpack(">i", data[0x3C:0x40])[0]
    counters_offset = struct.unpack(
        ">i", data[header2_offset + 0x30: header2_offset + 0x34])[0]

    size = 4*n_coords
    base_x = 0x40 + 4*n_coords
    base_y = base_x + size
    xs = struct.unpack(">" + n_coords*"f", data[base_x : base_x + size])
    ys = struct.unpack(">" + n_coords*"f", data[base_y : base_y + size])
    cs = struct.unpack(
        ">" + n_coords*"i", data[counters_offset : counters_offset + size])
    ts = [(c >> 8) - 1 for c in cs]
    points = np.array([xs, ys, ts], np.float32).T
    if sort_z:
        points = points[np.argsort(points[:, 2]), :]
    return points


def rescale_imagej_roi_points(path_in: str, path_out: str,
                              scale: tuple[float]) -> npt.NDArray[np.float32]:
    with open(path_in, "rb") as f:
        data = f.read()
    data_out = bytearray(data)

    n_coords = struct.unpack(">h", data[0x10:0x12])[0]
    header2_offset = struct.unpack(">i", data[0x3C:0x40])[0]
    counters_offset = struct.unpack(
        ">i", data[header2_offset + 0x30: header2_offset + 0x34])[0]

    size = 4*n_coords
    base_x = 0x40 + 4*n_coords
    base_y = base_x + size
    xs = struct.unpack(">" + n_coords*"f", data[base_x : base_x + size])
    ys = struct.unpack(">" + n_coords*"f", data[base_y : base_y + size])
    cs = struct.unpack(
        ">" + n_coords*"i", data[counters_offset : counters_offset + size])
    ts = [(c >> 8) - 1 for c in cs]

    for i in range(n_coords):
        ix = base_x + i*4
        iy = base_y + i*4
        ic = counters_offset + i*4
        data_out[ix:ix+4] = struct.pack(">f", xs[i] * scale[0])
        data_out[iy:iy+4] = struct.pack(">f", ys[i] * scale[1])
        data_out[ic:ic+4] = struct.pack(">i", cs[i] & 0xFF | (round(ts[i] * scale[2] + 1) << 8))

    with open(path_out, "wb") as f:
        f.write(data_out)
