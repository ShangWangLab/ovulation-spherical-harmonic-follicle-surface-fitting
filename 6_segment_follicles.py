import csv
import os

import nrrd
import numpy as np
from scipy import ndimage as ndi
import torch

from oocyte_datasets import *
from sphharmodel import (
    UVSphere,
    SphericalSurface,
    tonp,
)


class ImgSample(torch.autograd.Function):
    ORDER = 3
    MODE = "nearest"
    
    @staticmethod
    def forward(ctx, coords, img_grad):
        coords_np = tonp(coords)

        I = ndi.map_coordinates(
            img_grad.volume,
            coords_np,
            order=ImgSample.ORDER,
            mode=ImgSample.MODE
        )

        ctx.save_for_backward(coords)
        ctx.gx = img_grad.gx
        ctx.gy = img_grad.gy
        ctx.gz = img_grad.gz

        return torch.tensor(
            I,
            dtype=torch.float32,
            device=coords.device
        )

    @staticmethod
    def backward(ctx, grad_output):
        coords, = ctx.saved_tensors
        coords_np = tonp(coords)

        Gx = ndi.map_coordinates(
            ctx.gx,
            coords_np,
            order=ImgSample.ORDER,
            mode=ImgSample.MODE
        )
        Gy = ndi.map_coordinates(
            ctx.gy,
            coords_np,
            order=ImgSample.ORDER,
            mode=ImgSample.MODE
        )
        Gz = ndi.map_coordinates(
            ctx.gz,
            coords_np,
            order=ImgSample.ORDER,
            mode=ImgSample.MODE
        )

        grad_coords = torch.stack([
            torch.tensor(Gz),
            torch.tensor(Gy),
            torch.tensor(Gx)
        ])
        grad_coords = grad_coords * grad_output
        return grad_coords, None


def sample_volume(volume, coords):
    coords = tonp(coords)
    sampled = ndi.map_coordinates(
        volume,
        coords,
        order=3,
        mode="nearest"
    )
    return sampled


class VolumeGradients:
    def __init__(self, volume, sigma=1.):
        self.sigma = sigma
        self.volume = volume
        self.gz = ndi.gaussian_filter(volume, sigma=sigma, order=[1, 0, 0])
        self.gy = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 1, 0])
        self.gx = ndi.gaussian_filter(volume, sigma=sigma, order=[0, 0, 1])


def radial_gradient(coords, center, img_grad):
    coords = tonp(coords)
    center = tonp(center)
    dx = coords[2] - center[2]
    dy = coords[1] - center[1]
    dz = coords[0] - center[0]
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm
    
    Gx = sample_volume(img_grad.gx, coords)
    Gy = sample_volume(img_grad.gy, coords)
    Gz = sample_volume(img_grad.gz, coords)

    radial_grad = Gx*nx + Gy*ny + Gz*nz
    return radial_grad


def volume_grid(shape):
    z = np.arange(shape[0])
    y = np.arange(shape[1])
    x = np.arange(shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return np.stack([Z.flatten(), Y.flatten(), X.flatten()])


def compute_loss(model, sph_coords, rad_grad):
    Z, Y, X, R = model(sph_coords, dilate=-SIGMA_FIT)
    coords = torch.stack([Z, Y, X])

    # Center drift penalty
    center_loss = model.center_loss()

    # Radius drift penalty
    radius_loss = model.radius_loss()

    # Maximize gradient magnitude
    gradient_intersection = ImgSample.apply(coords, rad_grad)
    gradient_loss = -torch.mean(
        gradient_intersection * sph_coords.get_area_weights())

    # Model parameter regularization.
    reg_loss = model.regularization()

    total = (WEIGHTS["center"] * center_loss
           + WEIGHTS["radius"] * radius_loss
           + WEIGHTS["gradient"] * gradient_loss
           + WEIGHTS["regularization"] * reg_loss)
    if USE_FIXED_PATCH:
        fixed_patch_loss = (WEIGHTS["fixed_patch"]
            / FIXED_PATCH_MASK.sum()
            * ((R[FIXED_PATCH_MASK] - FIXED_PATCH_R)**2).sum())
        total = total + fixed_patch_loss
    return total


def make_radial_gradient(center, vol_coords, img_grad):
    rad_grad = radial_gradient(vol_coords, CENTER_INIT, img_grad)
    rad_grad = rad_grad.reshape(img_grad.volume.shape)
    rad_grad = VolumeGradients(rad_grad, SIGMA_RAD_GRAD)
    return rad_grad


def fit_follicle_surface(model, volume, vol_coords, sph_coords):
    img_grad = VolumeGradients(volume, SIGMA_FIT)
    rad_grad = make_radial_gradient(CENTER_INIT, vol_coords, img_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=SCHEDULE[0][1])

    t_step = 0  # Total steps
    best_loss = float("inf")
    best_t_step = 0
    for L, lr, n_steps in SCHEDULE:
        #print("Stage", L, lr, n_steps)
        model.set_cutoff(L)
        optimizer.param_groups[0]["lr"] = lr
        s_step = 0  # Stage step
        while s_step < n_steps or t_step - best_t_step < N_PLATEAU:
            optimizer.zero_grad()
            loss = compute_loss(model, sph_coords, rad_grad)
            if best_loss - loss.item() > SIG_LOSS:
                best_t_step = t_step
                best_loss = loss.item()
            loss.backward()
            optimizer.step()
            s_step += 1
            t_step += 1
            #print(f"Step {t_step} Loss {loss.item():.4f}")
            #if t_step % 10 == 0:
            #    model.plot_sections()
        if L > 0:
            continue
        # After identifying the center with L=0, we adjust the radial gradient
        # to originate from that center.
        rad_grad = make_radial_gradient(model.get_center(), vol_coords, img_grad)
    return model, best_loss, t_step


#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print("Evaluating on", DEVICE)

N_THETA, N_PHI = 32, 64

SIGMA_FIT = 0.7
SIGMA_RAD_GRAD = 0.5
WEIGHTS = dict(
    center = 4e-4,
    radius = 1e-3,
    gradient = 1.0,
    regularization = 3e-4,  # 3e-4 is standard, but up to 1e-3 helps at times.
    fixed_patch = 1e-2,  # Optional
)
# L, learning rate, min. steps
SCHEDULE = [
    (0, 0.6, 10),  # Just the center and radius.
    (2, 0.7, 5),
    (4, 0.5, 3),
    (6, 0.3, 3),
    (8, 0.2, 3),
    (9, 0.1, 10),  # Normally 10
]
N_PLATEAU = 6  # How many steps to wait before giving up on training further.
SIG_LOSS = 0.01  # Significant loss to signal that we should keep training.

CENTER_INIT = 3 * [SUB_SIZE/2]
vol_coords = volume_grid(3 * [SUB_SIZE])
sph_coords = UVSphere(DEVICE, N_THETA, N_PHI, include_poles=False)
sph_coords_w_poles = UVSphere(DEVICE, N_THETA, N_PHI, include_poles=True)
sph_coords.get_harmonics(L_MAX)  # Precompute the highest harmonics needed.

# I can pin a single point on the surface to a location in space if I need to
# repair a stable surface defect. Increasing the initial diameter is prefered.
def fixed_coords(surface_zyx, sph_coords, rough_center):
    z, y, x = [a - b for a, b in zip(surface_zyx, rough_center)]
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2*np.pi)
    dist2 = ((sph_coords.theta - theta)**2
             + ((sph_coords.phi - phi) / np.sin(theta))**2)
    return r, torch.tensor(dist2 < 0.1, dtype=torch.bool, device=DEVICE)
USE_FIXED_PATCH = False
if USE_FIXED_PATCH:
    # For 050825_hCG11h_4:
    FIXED_PATCH_R, FIXED_PATCH_MASK = fixed_coords(
        [26, 17.3, 23.4], sph_coords, [17.1766, 16.3504, 22.0194])
    print(FIXED_PATCH_MASK.sum().item())


dir_img = ROOT + "follicles/img/"
dir_fit = ROOT + "follicles/fit/"
dir_surf = ROOT + "follicles/surf/"

dir_fol_csv = ROOT + "follicles/csv/fit/"
dir_fol_curve = ROOT + "follicles/curvature/"
os.makedirs(dir_fol_csv, exist_ok=True)
os.makedirs(dir_fol_curve, exist_ok=True)

for di in TO_PROCESS:
    follicles = get_scan_follicles(di)
    print(NAMES[di], "processing", len(follicles), "follicles.")
    if not follicles:
        continue

    for fol in follicles:
        if not fol.use or fol.r is None:
            continue
        print(fol.label)
        stack_path = dir_img + fol.label + ".nrrd"
        if not os.path.exists(stack_path):
            continue  # Previous step hasn't occurred.
        curvature_path = dir_fol_curve + fol.label + f"_NTx{N_THETA}x{N_PHI}.npy"
        csv_path = dir_fol_csv + fol.label + ".csv"
        if not fol.force_update and os.path.exists(csv_path):
            continue  # Already processed.

        # Open the 4D stack of follicle volumes.
        stack, header = nrrd.read(stack_path, index_order="C")
        stack = stack.astype(np.float32)
        stack -= stack.mean()
        stack /= stack.std()

        model = SphericalSurface(CENTER_INIT, fol.r, L_MAX).to(DEVICE)
        
        fol_fit_dir = dir_fit + fol.label + "/"
        fol_surf_dir = dir_surf + fol.label + "/"
        os.makedirs(fol_fit_dir, exist_ok=True)
        os.makedirs(fol_surf_dir, exist_ok=True)

        fit_loss = []
        fit_steps = []
        fit_radii = []
        fit_centers = []
        curvatures = np.empty((stack.shape[0], N_THETA, N_PHI), np.float32)
        # Iterate over each volume, fitting a surface and saving it, along with metrics.
        for it, volume in enumerate(stack):
            print(f"{it}/{stack.shape[0]}", end="")
            model, loss, steps = fit_follicle_surface(
                model, volume, vol_coords, sph_coords)
            radius = model.mean_radius().item()
            print(f" D={50*radius:.2f} steps={steps} loss={loss:.4f}")
            fit_loss.append(loss)
            fit_steps.append(steps)
            fit_radii.append(radius)
            fit_centers.append([v.item() for v in model.center])
            torch.save(model.state_dict(), fol_fit_dir + f"{it:03d}.pt")
            sph_coords_w_poles.make_ply(fol_surf_dir + f"{it:03d}.ply",
                model, scale=SCALE_MM, h_range=H_RANGE, centered=True)
            curvatures[it].flat = tonp(
                model.approx_curvature(sph_coords, SCALE_MM))

        # Save curvature maps.
        np.save(curvature_path, curvatures)
        
        # Save extra metrics to CSV.
        with open(csv_path, "w") as file:
            csv_dialect = csv.excel()
            csv_dialect.lineterminator = "\n"
            out = csv.writer(file, csv_dialect)
            out.writerow(["T (index)", "Training Steps", "Loss", "Diameter (um)",
                          "Cen Z (vox)", "Cen Y (vox)", "Cen X (vox)",
                          "Min. Curve (mm^-1)", "Max. Curve (mm^-1)"])
            for it, (l, s, r, cen) in enumerate(zip(
                    fit_loss, fit_steps, fit_radii, fit_centers)):
                dia = round(SCALE * 2 * r)
                h_min = round(curvatures[it].min(), 2)
                h_max = round(curvatures[it].max(), 2)
                cen = [round(v, 2) for v in cen]
                out.writerow([it, s, l, dia, *cen, h_min, h_max])
