#!/usr/bin/env python
"""
Ray-traced (line-of-sight integrated) brightness map of the SPH star.
Optically thin emissivity j ∝ rho^2, integrated along z.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from amuse.io import read_set_from_file
from amuse.units import units

# ---------------- load ----------------
parts = read_set_from_file("polytrope_relaxed.amuse", "amuse")
N = len(parts)

pos = parts.position.value_in(units.RSun)      # (N,3)
m   = parts.mass.value_in(units.MSun)          # (N,)
x, y, z = pos[:,0], pos[:,1], pos[:,2]

# ------- smoothing lengths + densities via k-NN -------
k = 32
tree = cKDTree(pos)
dists, _ = tree.query(pos, k=k+1)              # includes self at d=0
h = dists[:, -1] / 2.0                         # h = half the k-NN radius
rho = m * k / (4.0/3.0*np.pi*dists[:, -1]**3)  # simple k-NN density, MSun/RSun^3
print(f"N={N},  <h>={h.mean():.3f} RSun,  rho_c~{rho.max():.2f} MSun/RSun^3")

# ---------------- image grid ----------------
L    = 1.4                                     # half-size of image [RSun]
npix = 400
edges  = np.linspace(-L, L, npix+1)
centers = 0.5*(edges[:-1] + edges[1:])
image  = np.zeros((npix, npix))
dx = 2*L/npix

# Each particle contributes its emission measure m_i * rho_i,
# spread over the image as a projected Gaussian of width sigma = h_i.
weight = m * rho                               # ∝ ∫ rho^2 dV per particle

for i in range(N):
    s = max(h[i], dx)                          # don't let kernels be sub-pixel
    # bounding box: +/- 3 sigma
    ix0, ix1 = np.searchsorted(edges, [x[i]-3*s, x[i]+3*s])
    iy0, iy1 = np.searchsorted(edges, [y[i]-3*s, y[i]+3*s])
    ix0, iy0 = max(ix0-1, 0), max(iy0-1, 0)
    if ix0 >= ix1 or iy0 >= iy1:
        continue
    gx = np.exp(-0.5*((centers[ix0:ix1]-x[i])/s)**2)
    gy = np.exp(-0.5*((centers[iy0:iy1]-y[i])/s)**2)
    kern = np.outer(gy, gx)
    kern *= weight[i] / (2*np.pi*s*s)          # normalized 2D projection
    image[iy0:iy1, ix0:ix1] += kern

image /= dx*dx                                 # per unit area -> surface brightness

# ---------------- plots ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

im = ax1.imshow(image, origin='lower', extent=[-L,L,-L,L],
                cmap='inferno',
                norm=plt.matplotlib.colors.LogNorm(
                    vmin=image.max()*1e-4, vmax=image.max()))
ax1.set_xlabel('x [RSun]'); ax1.set_ylabel('y [RSun]')
ax1.set_title(r'Surface brightness  $\int \rho^2\, dz$')
fig.colorbar(im, ax=ax1, label='arbitrary units')

# radial profile
X, Y = np.meshgrid(centers, centers)
R = np.sqrt(X**2 + Y**2).ravel()
B = image.ravel()
rb = np.linspace(0, L, 40)
prof = [B[(R>=rb[j])&(R<rb[j+1])].mean() for j in range(len(rb)-1)]
ax2.semilogy(0.5*(rb[:-1]+rb[1:]), prof, 'o-')
ax2.set_xlabel('projected radius [RSun]')
ax2.set_ylabel('mean surface brightness')
ax2.set_title('Brightness profile')

plt.tight_layout()
plt.savefig('brightness_map.png', dpi=130)
print("Wrote brightness_map.png")
plt.show()
