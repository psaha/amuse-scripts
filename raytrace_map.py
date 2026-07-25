#!/usr/bin/env python
"""
Absorption-aware ray marching through the SPH star.
Grey opacity, blackbody source function:  I = int S exp(-tau) kappa rho dz
Observer looks along -z (image is the xy plane).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from amuse.io import read_set_from_file
from amuse.units import units

# ------------------ knobs ------------------
kappa   = 0.4          # grey opacity [cm^2/g]  (~electron scattering)
ngrid   = 256          # 3D grid cells per axis
L       = 1.5          # half-size of box [RSun]
mu_mol  = 0.6          # mean molecular weight (ionized, solar-ish)

# ------------------ constants (cgs) ------------------
RSun_cm = 6.957e10
MSun_g  = 1.989e33
m_p     = 1.6726e-24
k_B     = 1.3807e-16
sigma_sb= 5.6704e-5

# ------------------ load ------------------
parts = read_set_from_file("polytrope_relaxed.amuse", "amuse")
N   = len(parts)
pos = parts.position.value_in(units.RSun)
m   = parts.mass.value_in(units.MSun) * MSun_g          # g
u   = parts.u.value_in(units.erg / units.g)             # erg/g
T_p = (2.0/3.0) * mu_mol * m_p * u / k_B                # K
print(f"N={N}   T_c ~ {T_p.max():.3e} K   T_min ~ {T_p.min():.3e} K")

# smoothing lengths from k-NN (in RSun)
k = 64
tree = cKDTree(pos)
dists, _ = tree.query(pos, k=k+1)
h = dists[:, -1] / 2.0

# ------------------ deposit onto 3D grid ------------------
edges   = np.linspace(-L, L, ngrid+1)
centers = 0.5*(edges[:-1] + edges[1:])
dx      = 2.0*L/ngrid                                   # RSun
dx_cm   = dx * RSun_cm

rho_grid = np.zeros((ngrid, ngrid, ngrid))              # mass first
mT_grid  = np.zeros_like(rho_grid)                      # mass-weighted T

for i in range(N):
    s = max(h[i], dx)                                   # kernel width, >= 1 cell
    lo = np.searchsorted(edges, pos[i] - 3*s) - 1
    hi = np.searchsorted(edges, pos[i] + 3*s)
    lo = np.clip(lo, 0, ngrid); hi = np.clip(hi, 0, ngrid)
    if np.any(lo >= hi): continue
    gx = np.exp(-0.5*((centers[lo[0]:hi[0]] - pos[i,0])/s)**2)
    gy = np.exp(-0.5*((centers[lo[1]:hi[1]] - pos[i,1])/s)**2)
    gz = np.exp(-0.5*((centers[lo[2]:hi[2]] - pos[i,2])/s)**2)
    kern = gx[:,None,None]*gy[None,:,None]*gz[None,None,:]
    tot = kern.sum()
    if tot <= 0: continue
    kern *= m[i]/tot                                    # conserve mass exactly
    rho_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += kern
    mT_grid [lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += kern*T_p[i]

T_grid = np.where(rho_grid > 0, mT_grid/np.maximum(rho_grid, 1e-300), 0.0)
rho_grid /= dx_cm**3                                    # g/cm^3
print(f"grid rho_max = {rho_grid.max():.3e} g/cm^3")
print(f"tau across a central column ~ {kappa*rho_grid[ngrid//2,ngrid//2,:].sum()*dx_cm:.3e}")

# ------------------ ray march along -z ------------------
S_grid  = sigma_sb * T_grid**4 / np.pi                  # source function
image   = np.zeros((ngrid, ngrid))
trans   = np.ones((ngrid, ngrid))                       # transmittance so far
tau_tot = np.zeros((ngrid, ngrid))

for kz in range(ngrid-1, -1, -1):                       # front (obs side) -> back
    dtau = kappa * rho_grid[:, :, kz] * dx_cm
    emit = S_grid[:, :, kz] * (1.0 - np.exp(-dtau))     # slab's own emission
    image += trans * emit
    trans *= np.exp(-dtau)
    tau_tot += dtau

# brightness temperature: T_b = (pi I / sigma)^(1/4)
T_b = (np.pi*np.maximum(image, 0)/sigma_sb)**0.25

# ------------------ plots ------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(T_b.T, origin='lower', extent=[-L, L, -L, L], cmap='afmhot')
ax1.set_xlabel('x [RSun]'); ax1.set_ylabel('y [RSun]')
ax1.set_title('Brightness temperature [K]')
fig.colorbar(im1, ax=ax1)

X, Y = np.meshgrid(centers, centers, indexing='ij')
R = np.sqrt(X**2 + Y**2).ravel()
B = image.ravel()
rb = np.linspace(0, L, 40)
prof = np.array([B[(R >= rb[j]) & (R < rb[j+1])].mean() for j in range(len(rb)-1)])
ax2.plot(0.5*(rb[:-1]+rb[1:]), prof/prof.max(), 'o-')
ax2.set_xlabel('projected radius [RSun]')
ax2.set_ylabel('I / I_max')
ax2.set_title('Normalized brightness profile')
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('raytrace_map.png', dpi=130)
print("Wrote raytrace_map.png")
plt.show()
