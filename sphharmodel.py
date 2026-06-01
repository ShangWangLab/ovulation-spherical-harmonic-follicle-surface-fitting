import numpy as np
from scipy.special import sph_harm_y
import torch


def tonp(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def sh_index(l, m):
    """Linear indexing of spherical harmonics coefficients.

    Satisfying |m| <= l
    """
    return l*(l+1) + m


def num_sh_coeffs(L):
    """How many spherical harmonics coefficients exist."""
    return (L + 1)**2


def sph_harm_complex_to_real(m, y):
    if m == 0:
        r = y.real
    else:
        c = (-1)**m * np.sqrt(2)
        if m > 0:
            r = c * y.real
        else:
            r = c * y.imag
    return torch.tensor(r, dtype=torch.float32)


def real_sph_harm_d2(l, m, theta, phi):
    """2nd order derivatives:
    0th order (n,): Y
    1st order (n, 2): (dR/dtheta, dR/dphi)
    2nd order (n, 2, 2): (
        d2R/d2theta, d2R/dtheta/dphi
        d2R/dphi/dtheta, d2R/d2phi
    )
    """
    Y = sph_harm_y(l, abs(m), tonp(theta), tonp(phi), diff_n=2)
    return [sph_harm_complex_to_real(m, y) for y in Y]


def real_spherical_harmonic(l, m, theta, phi):
    Y = sph_harm_y(l, abs(m), tonp(theta), tonp(phi))
    return sph_harm_complex_to_real(m, Y)


def test_real_spherical_harmonics():
    # See https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    import math
    coords = UVSphere(torch.device("cpu"), 3, 3, False)
    theta = coords.theta
    phi = coords.phi

    # l = 0
    assert torch.allclose(
        real_spherical_harmonic(0, 0, theta, phi),
        torch.ones_like(theta)/math.sqrt(math.pi)/2), "y(0, 0)"

    # l = 1
    c = math.sqrt(3/(4*math.pi))
    assert torch.allclose(
        real_spherical_harmonic(1, -1, theta, phi),
        c * torch.sin(theta) * torch.sin(phi)), "y(1, -1)"
    assert torch.allclose(
        real_spherical_harmonic(1, 0, theta, phi),
        c * torch.cos(theta)), "y(1, 0)"
    assert torch.allclose(
        real_spherical_harmonic(1, 1, theta, phi),
        c * torch.sin(theta) * torch.cos(phi)), "y(1, +1)"

    # l = 2
    c = math.sqrt(15/(16*math.pi))
    assert torch.allclose(
        real_spherical_harmonic(2, -2, theta, phi),
        c * torch.sin(theta)**2 * torch.sin(2*phi)), "y(2, -2)"
    assert torch.allclose(
        real_spherical_harmonic(2, -1, theta, phi),
        c * torch.sin(2*theta) * torch.sin(phi)), "y(2, -1)"
    assert torch.allclose(
        real_spherical_harmonic(2, 0, theta, phi),
        math.sqrt(5/math.pi)/4 * (3*torch.cos(theta)**2 - 1)), "y(2, 0)"
    assert torch.allclose(
        real_spherical_harmonic(2, 1, theta, phi),
        c * torch.sin(2*theta) * torch.cos(phi)), "y(2, +1)"
    assert torch.allclose(
        real_spherical_harmonic(2, 2, theta, phi),
        c * torch.sin(theta)**2 * torch.cos(2*phi)), "y(2, +2)"


class SphCoords:
    """A collection of spherical coordinates, (theta, rho) with harmonic
    caching.

    Theta is defined as the angle between a point and the positive Z axis.
    Phi is defined as the angle from the X axis to the projection of the
    point onto the XY plane.
    """
    
    def __init__(self, theta, phi):
        self.theta = theta.flatten()
        self.phi = phi.flatten()
        self.nc = self.theta.shape[0]  # The number of coordinates
        self.device = theta.device
        self.L = -1  # Highest value of L calculated for harmonics.
        self.area_weights = None  # Area-weighted coordinate values.
        self.Ylm = None  # Spherical harmonic matrix, shape: (nc, (L+1)^2)
        self.dYlm = None  # 1st derivative matrix: (t, p)
        self.ddYlm = None  # 2nd derivative matrix: ((tt, tp), (tp, pp))

    def get_area_weights(self):
        if self.area_weights is None:
            self.area_weights = torch.sin(self.theta)
        return self.area_weights

    def get_harmonics(self, L):
        nlm = num_sh_coeffs(L)
        if L > self.L:
            self.L = L
            self.Ylm = torch.empty((self.nc, nlm), dtype=torch.float32,
                                   device=self.device)
            i = 0
            for l in range(self.L + 1):
                for m in range(-l, l+1):
                    self.Ylm[:, i] = real_spherical_harmonic(
                        l, m, self.theta, self.phi)
                    i += 1
        return self.Ylm[:, :nlm]

    def get_gradients(self, L):
        nlm = num_sh_coeffs(L)
        if self.dYlm is None or self.dYlm.shape[2] < nlm:
            self.L = L
            self.Ylm = torch.empty((self.nc, nlm), dtype=torch.float32,
                                   device=self.device)
            self.dYlm = torch.empty((self.nc, 2, nlm), dtype=torch.float32,
                                    device=self.device)
            self.ddYlm = torch.empty((self.nc, 2, 2, nlm), dtype=torch.float32,
                                     device=self.device)
            i = 0
            for l in range(self.L + 1):
                for m in range(-l, l+1):
                    Ylm, dYlm, ddYlm = real_sph_harm_d2(
                        l, m, self.theta, self.phi)
                    self.Ylm[:, i] = Ylm
                    self.dYlm[:, :, i] = dYlm
                    self.ddYlm[:, :, :, i] = ddYlm
                    i += 1
        return (self.Ylm[:, :nlm],
                self.dYlm[:, :, :nlm],
                self.ddYlm[:, :, :, :nlm])


class UVSphere(SphCoords):
    def __init__(self, device, n_theta=32, n_phi=64, include_poles=False):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.include_poles = include_poles
        
        phi = 2*torch.pi / n_phi * torch.arange(n_phi, device=device)
        # Exclude the end points, since they would be duplicates.
        theta = torch.pi/(n_theta+1) * torch.arange(1, n_theta+1, device=device)

        phi, theta = torch.meshgrid(phi, theta, indexing="ij")
        theta = theta.flatten()
        phi = phi.flatten()
        if include_poles:
            theta = torch.cat((theta, torch.tensor([0, torch.pi], device=device)))
            phi = torch.cat((phi, torch.tensor([0, 0], device=device)))
        super().__init__(theta, phi)

    def faces(self):
        # Faces are zero-based indices into the vertex array.
        n_theta, n_phi = self.n_theta, self.n_phi
        faces = []
        # Middle strip (quads)
        for p in range(n_phi):
            p2 = (p+1) % n_phi
            for t in range(n_theta - 1):
                ic0 = t + p * n_theta
                ic1 = t + p2 * n_theta
                faces.append((ic0 + 1, ic1 + 1, ic1, ic0))
        if not self.include_poles:
            return faces
        # Pole caps (triangles)
        for p in range(n_phi):
            a = n_phi*n_theta
            b = p*n_theta
            c = (p+1) % n_phi * n_theta
            x = n_theta - 1
            faces.append((a, b, c))  # Top cap
            faces.append((a + 1, c + x, b + x))  # Bottom cap
        return faces

    def make_ply(self, path_out, model, scale=25e-3, h_range=10., centered=False):
        # Calculate curvature and colors.
        curvatures = model.approx_curvature(self, scale)
        if self.include_poles:
            # Replace pole curvatures with the average of the nearest ring.
            t, p = self.n_theta, self.n_phi
            curvatures[t*p] = torch.mean(curvatures[t*torch.arange(p)])
            curvatures[t*p + 1] = torch.mean(curvatures[t*torch.arange(1, p+1) - 1])
        cmap = np.load("cmap_coolwarm.npy")
        vcolors = np.empty((self.nc, 4), dtype=np.uint8)
        for i, h in enumerate(curvatures):
            vcolors[i] = cmap[
                np.clip(round((h.item()/h_range + 1) * 255/2), 0, 255)]

        faces = self.faces()
        Z, Y, X, R = model(self, centered=centered)
        with open(path_out, "w") as file:
            print("ply", file=file)
            print("format ascii 1.0", file=file)
            print("element vertex", self.nc, file=file)
            for prop in "xyz":
                print("property float", prop, file=file)
            for prop in ["red", "green", "blue", "alpha"]:
                print("property uchar", prop, file=file)
            print("element face", len(faces), file=file)
            print("property list uchar uint vertex_indices", file=file)
            print("end_header", file=file)
            for z, y, x, c in zip(Z, Y, X, vcolors):
                print(scale*x.item(), scale*y.item(), scale*z.item(),
                      *map(int, c), file=file)
            for f in faces:
                print(len(f), *f, file=file)


class IntegralSurface(SphCoords):
    def __init__(self, device, n_theta=32, n_phi=64):
        self.n_theta = n_theta
        self.n_phi = n_phi
        
        theta = torch.pi/n_theta * (torch.arange(n_theta, device=device) + 1/2)
        phi = 2*torch.pi / n_phi * torch.arange(n_phi, device=device)

        phi, theta = torch.meshgrid(phi, theta, indexing="ij")
        theta = theta.flatten()
        phi = phi.flatten()
        super().__init__(theta, phi)

    def volume(self, model, scale):
        R = model.radius(self)
        # int 0:pi int 0:2pi int 0:R(theta, phi) r^2 sin(phi) dr dphi dtheta
        # = int 0:pi int 0:2pi sin(phi)*R(theta, phi)^3/3 dp dt
        # = Dp Dt / 3 * (sum 1:n_theta sum 1:n_phi R(theta, phi)^3 sin(phi))
        # Given,
        #   Dt = pi/n_theta
        #   Dp = 2*pi/n_phi
        # = 2*pi^2/3 * mean(R^3 * sin(phi))
        fudge = 0.822  # Account for sampling bias.
        return (scale**3 * np.pi**2 * 2
                / (3 * (self.n_theta * self.n_phi + fudge))
                * torch.sum(R**3 * torch.sin(self.theta)).item())

    def center_of_mass(self, model):
        Z, Y, X, R = model(self)
        v = torch.sum(R**3 * torch.sin(self.theta))
        center = [(torch.sum(u * R**3 * torch.sin(self.theta)) / v).item()
                  for u in (Z, Y, X)]
        return center


class CircularPatch(SphCoords):
    """A patch which is polar in theta phi space."""
    def __init__(self, device, c_theta, c_phi, Rho, n_rho=8, n_psi=16,
                 include_center=True):
        self.include_center = include_center
        self.Rho = Rho
        self.c_theta = c_theta
        self.c_phi = c_phi
        self.n_rho = n_rho
        self.n_psi = n_psi

        rho = Rho/n_rho * (torch.arange(n_rho) + 1/2)
        psi = 2*torch.pi/n_psi * torch.arange(n_psi)
        rho, psi = torch.meshgrid(rho, psi, indexing="ij")
        rho = rho.flatten()
        psi = psi.flatten()
        if include_center:
            rho = torch.cat((rho, torch.zeros((1,), device=device)))
            psi = torch.cat((psi, torch.zeros((1,), device=device)))
        self.rho = rho
        self.psi = psi

        # This still only roughly approximates a circle, but it's much closer
        # than unscaled phi.
        self.phi_scale = 1/np.sin(c_theta)
        theta = c_theta + rho * torch.cos(psi)
        phi = c_phi + self.phi_scale * rho * torch.sin(psi)
        super().__init__(theta, phi)

    def solid_angle(self):
        # Approximate; this model is not exactly a spherical cap.
        return 2 * np.pi * (1 - np.cos(self.Rho))

    def surface_area(self):
        # Approximate; this model is not exactly a spherical cap.
        return self.Rho**2 * self.solid_angle()

    def mean_curvature(self, model, scale):
        # This has a bug in how it weights the surface: the surface area
        # is actually distorted by sin(theta). This makes the average weight
        # areas closer to the poles more highly than the more equitorial areas.
        # It also neglects phi_scale, although the terms should cancel.
        H = model.approx_curvature(self, scale)
        # int 0:2pi int 0:Rho H(theta, phi) * rho drho dpsi
        #   / int 0:2pi int 0:Rho rho drho dpsi
        # int 0:2pi int 0:Rho H(theta, phi) * rho drho dpsi / pi * Rho^2
        # drho = Rho/n_rho
        # dpsi = 2pi/n_psi
        # Rho/n_rho * 2pi/n_psi / (pi Rho^2)
        # 2 / (Rho * n_psi * n_rho)
        # We deliberately do not count the center point when integrating.
        return (2 / (self.Rho * self.n_rho * self.n_psi)
                * torch.sum(H * self.rho).item())

    def subtended_volume(self, model, scale, const_radius=False):
        R = model.radius(self)
        
        if const_radius:
            #R = model.mean_radius()
            # Use the mean radius at the boundary instead of the whole
            # follicle.
            R = R[(self.n_rho - 1)*self.n_psi:self.n_rho*self.n_psi].mean()

        # int int int 0:R(theta, phi) r^2 sin(theta) dr dphi dtheta
        # = int int sin(phi)*R(theta, phi)^3/3 dp dt
        # dphi dtheta = phi_scale drho dpsi rho
        # (Drho Dpsi phi_scale / 3)(sum sum R(theta, phi)^3 rho sin(theta))
        # Given,
        #   Drho = Rho/n_rho
        #   Dpsi = 2*pi/n_psi
        # = 2*pi*Rho*phi_scale/3 * mean(R^3 * rho * sin(theta))
        return (scale**3 * 2 * np.pi * self.Rho * self.phi_scale / 3
                * torch.mean(R**3 * self.rho * torch.sin(self.theta)).item())

    def faces(self):
        # Faces are zero-based indices into the vertex array.
        n_rho, n_psi = self.n_rho, self.n_psi
        faces = []
        # Rings (quads)
        for r in range(n_rho - 1):
            for p in range(n_psi):
                p2 = (p+1) % n_psi
                ic0 = p + r * n_psi
                ic1 = p2 + r * n_psi
                faces.append((ic0 + n_psi, ic1 + n_psi, ic1, ic0))
        if not self.include_center:
            return faces
        # Center cap (triangles)
        a = n_rho*n_psi  # Last point
        for p in range(n_psi):
            b = p
            c = (p+1) % n_psi
            faces.append((a, b, c))  # Top cap
        return faces

    def edge_loop(self):
        # Edges are zero-based indices into the vertex array returned.
        n_rho, n_psi = self.n_rho, self.n_psi
        vert_indices = [(n_rho - 1) * n_psi + p for p in range(n_psi)] 
        edges = [(p, p % n_psi) for p in range(n_psi)]
        return vert_indices, edges

    def make_ply(self, path_out, model, scale=25e-3, h_range=15., centered=False):
        # Calculate curvature and colors.
        curvatures = model.approx_curvature(self, scale)
        cmap = np.load("cmap_coolwarm.npy")
        vcolors = np.empty((self.nc, 4), dtype=np.uint8)
        for i, h in enumerate(curvatures):
            vcolors[i] = cmap[
                np.clip(round((h.item()/h_range + 1) * 255/2), 0, 255)]

        faces = self.faces()
        Z, Y, X, R = model(self, centered=centered)
        with open(path_out, "w") as file:
            print("ply", file=file)
            print("format ascii 1.0", file=file)
            print("element vertex", self.nc, file=file)
            for prop in "xyz":
                print("property float", prop, file=file)
            for prop in ["red", "green", "blue", "alpha"]:
                print("property uchar", prop, file=file)
            print("element face", len(faces), file=file)
            print("property list uchar uint vertex_indices", file=file)
            print("end_header", file=file)
            for z, y, x, c in zip(Z, Y, X, vcolors):
                print(scale*x.item(), scale*y.item(), scale*z.item(),
                      *map(int, c), file=file)
            for f in faces:
                print(len(f), *f, file=file)


class SphericalHarmonicsRadius(torch.nn.Module):
    """R(theta, phi) using spherical harmonics."""

    # = 1/real_spherical_harmonic(0, 0, 0, 0).item()
    RADIUS_FACTOR = 2 * np.sqrt(np.pi)
    
    def __init__(self, radius_init: float, L_max: int):
        super().__init__()

        self.L = L_max
        self.L_max = L_max
        self.n_coeffs = num_sh_coeffs(L_max)
        self.coeffs = torch.nn.Parameter(
            1e-3 * torch.randn(self.n_coeffs))

        self.radius_init = radius_init
        # Initialize radius using the zeroth harmonic.
        self.coeffs.data[0] = radius_init * self.RADIUS_FACTOR

        def hook(grad):
            grad = grad.clone()
            grad[num_sh_coeffs(self.L):] = 0
            return grad
        self.coeffs.register_hook(hook)

    def forward(self, sph_coords):
        R = sph_coords.get_harmonics(self.L_max) @ self.coeffs
        return R

    def radius(self, sph_coords):
        return self.forward(sph_coords)

    def approx_curvature(self, sph_coords, scale):
        # Using the Laplacian of r. Not valid near poles.
        # See https://math.stackexchange.com/questions/4246543/linearzation-of-curvature-in-spherical-coordinates#:~:text=Since%20the%20surface%20is%20defined%20by%20r=f(%CE%B8%2C%CF%95)%2C,mean%20curvature%20of%20a%20sphere%20is%20positive).
        # Where scale is mm/voxel. Will probably be 25e-3.

        Y, dY, ddY = sph_coords.get_gradients(self.L_max)
        c = self.coeffs.detach()
        R = Y @ c
        dRtt = ddY[:, 0, 0, :] @ c
        dRpp = ddY[:, 1, 1, :] @ c

        sin_theta = torch.sin(sph_coords.theta)
        sin_theta[sin_theta == 0] = 1e-6
        # Mean curvature
        H = 1/R - (dRtt + dRpp/sin_theta**2) / (2*R**2)
        H /=  scale # 1/voxels -> 1/mm
        return H

    def set_cutoff(self, L):
        self.L = L

    def regularization(self, order=1):
        reg = 0
        ic = 0
        for l in range(self.L + 1):
            weight = l ** order
            for m in range(-l, l+1):
                reg += weight * self.coeffs[ic]**2
                ic += 1
        return reg

    def mean_radius(self):
        return self.coeffs[0].item() / self.RADIUS_FACTOR

    def radius_loss(self):
        return (self.coeffs[0] - self.radius_init * self.RADIUS_FACTOR)**2


class SphericalSurface(SphericalHarmonicsRadius):
    def __init__(self, center_init, radius_init, L_max=4, learn_center=True):
        super().__init__(radius_init, L_max)

        self.center_init = torch.tensor(center_init, dtype=torch.float32)
        self.center = torch.nn.Parameter(torch.tensor(
            center_init, dtype=torch.float32))
        self.center.requires_grad = learn_center

    def forward(self, sph_coords, dilate=0., centered=False):
        # The radius tends to be smaller than desired by a factor roughly
        # equal to sigma. We can train with a negative dilation to produce
        # slightly oversized fittings to compensate.
        R = super().forward(sph_coords) + dilate
        theta = sph_coords.theta
        phi = sph_coords.phi
        z0, y0, x0 = self.center
        sin_theta = torch.sin(theta)
        X = R * torch.cos(phi) * sin_theta
        Y = R * torch.sin(phi) * sin_theta
        Z = R * torch.cos(theta)
        if not centered:
            X = X + x0
            Y = Y + y0
            Z = Z + z0
        return Z, Y, X, R

    def radius(self, sph_coords):
        return super().forward(sph_coords)
    
    def get_center(self):
        return tonp(self.center)

    def center_loss(self):
        return torch.sum((self.center - self.center_init)**2)

    def to_mask(self, ref):
        """Given an reference volume, produce a like-shaped mask where
        regions inside of the model surface representation are true."""

        z, y, x = (torch.arange(ref.shape[i], dtype=torch.float32)
                   for i in range(3))
        z -= self.center[0].item()
        y -= self.center[1].item()
        x -= self.center[2].item()
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        Rs = torch.sqrt(X*X + Y*Y + Z*Z) + 1e-6

        phi = torch.atan2(Y, X)
        theta = torch.acos(Z / Rs)
        
        Rm = self.radius(SphCoords(theta, phi)).reshape(Rs.shape)
        inside = tonp(Rs <= Rm)
        return inside
        

    def plot_sections(self, n=64):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 3)
        # XY
        theta = torch.full((n,), torch.pi/2)
        phi = 2*torch.pi/n * torch.arange(n)
        R = self.radius(SphCoords(theta, phi))
        x = R * torch.cos(phi)
        y = R * torch.sin(phi)
        ax[0].plot(tonp(x), tonp(y), "r")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        # XZ
        thetafull = 2*torch.pi/n * torch.arange(n)
        theta = thetafull.clone()
        theta[n//2:] = 2*torch.pi - theta[n//2:]
        phi = torch.zeros((n,))
        phi[n//2:] = torch.pi
        R = self.radius(SphCoords(theta, phi))
        x = R * torch.sin(thetafull)
        y = R * torch.cos(thetafull)
        ax[1].plot(tonp(x), tonp(y), "b")
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Z")
        # YZ
        phi[:n//2] = torch.pi/2
        phi[n//2:] = -torch.pi/2
        R = self.radius(SphCoords(theta, phi))
        x = R * torch.sin(thetafull)
        y = R * torch.cos(thetafull)
        ax[2].plot(tonp(x), tonp(y), "m")
        ax[2].set_xlabel("Y")
        ax[2].set_ylabel("Z")
        for a in ax:
            a.grid(True)
            a.set_aspect("equal")
        plt.show()


def make_colormap():
    import matplotlib
    cmap = matplotlib.colormaps["coolwarm"](np.arange(256))
    cmap *= 255
    cmap.round(out=cmap)
    cmap = cmap.astype(np.uint8)
    np.save("cmap_coolwarm.npy", cmap)


if __name__ == "__main__":
    make_colormap()
    test_real_spherical_harmonics()
    dev = torch.device("cpu")
    coords = UVSphere(dev, 3, 6, True)
    model = SphericalSurface([1, 2, 3], 30)
##    coords.make_ply("test_sphere.ply", model, scale=0.1, h_range=0.5)

    c_theta = 30 * torch.pi/180
    c_phi = 60 * torch.pi/180
    #rho = 10 * torch.pi/180
    rho = 0.1 * torch.pi
    # Solid angle = 2*pi*(1 - cos(rho))
    patch = CircularPatch(dev, c_theta, c_phi, rho)
    print(patch.mean_curvature(model, scale=0.1), "vs.", 1/3)
    patch.make_ply("test_ring.ply", model, scale=0.1, h_range=0.5)

    cvol = patch.subtended_volume(model, 0.1, True)
    patch2 = CircularPatch(dev, np.pi/2, c_phi, rho)
    cvol2 = patch2.subtended_volume(model, 0.1, True)
    print("Cone volumes:", cvol, cvol2)
    # R = 3
    # r = R*sin(rho)
    # Flat ended cone: pi*r^2 * R/3
    print("Flat-ended cone volume:", np.pi*np.tan(rho)**2 * 3**3/3)
    # This ends up within 1.2% of the true volume when run on a uniform sphere,
    # even when calculated at different angles.
    print("Round-ended cone volume:", 2*np.pi*(1 - np.cos(rho)) * 3**3/3)

    surf = IntegralSurface(dev, 64, 128)
    print("CoM:", surf.center_of_mass(model))
    print("Volume:", surf.volume(model, scale=0.1), "vs.", 4/3 * np.pi * 3**3)
