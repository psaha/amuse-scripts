import matplotlib.pyplot as plt
from main import SPH
import numpy as np

np.random.seed(42)

sp = SPH()

# Parameter of each star
R1 = 0.95
R2 = 0.75
M1 = 3
M2 = 2
N1 = 600
N2 = 400
N = N1 + N2
a = 10
G = 1
d = 3e6
lam = 4e-7

# Parameter for SPH
h1 = 0.23
h2 = 0.2
n1 = 3
n2 = 3
k1 = 0.1
k2 = 0.1
nu1 = 0.2
nu2 = 0.1

# Parameter for time
dt = 0.4
tEnd = 120
Nt = int(np.ceil(tEnd / dt))
plotRealTime = True
t = 0

# parameter for Harmonic accelaration
lmbda1 = sp.getHarmonic(k1, n1, M1, R1)
lmbda2 = sp.getHarmonic(k2, n2, M2, R2)

# randomly selected positions and velocities   
pos = np.random.randn(N, 3)             
vel = np.zeros(pos.shape)

# Create center of mass for each star
pos[:N1, 0] -= a*M2/(M1+M2)
pos[N1:, 0] += a*M1/(M1+M2)

# Calculate orbital velocity
vorb = np.sqrt(G*(M1 + M2)/a)
vel[:N1, 1] += vorb*M2/(M1+M2)
vel[N1:, 1] -= vorb*M1/(M1+M2)
             
# calculate initial gravitational accelerations
acc = sp.getAcc(pos, vel, M1, M2, N1, lmbda1, lmbda2, h1, h2, G, k1, k2, n1, n2, nu1, nu2, relax=True)
    
# prep figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
im = ax2.imshow(np.zeros((150,150)))
cbar = fig.colorbar(im, ax=ax2)

# if each particle in stars are with same mass   
m1 = np.full(N1, M1/N1)
m2 = np.full(N2, M2/N2)
m = np.concatenate((m1, m2))

# if each point in stars have the same smoothing length   
h1_arr = np.full(N1, h1)
h2_arr = np.full(N2, h2)
h = np.concatenate((h1_arr, h2_arr))

KE1_old = None
KE2_old = None

relax = True
stable_steps = 0
tol1 = 0.002
tol2 = 0.001
# Simulation Main Loop
for i in range(Nt):

    KE1 = 0.5 * np.sum(m[:N1] * np.sum(vel[:N1]**2, axis=1))
    KE2 = 0.5 * np.sum(m[N1:] * np.sum(vel[N1:]**2, axis=1))

    if relax:

       if KE1_old is not None:

          dKE1 = abs(KE1 - KE1_old) / KE1_old
          dKE2 = abs(KE2 - KE2_old) / KE2_old
          print('dKE1', dKE1)
          print('dKE2', dKE2)
          if (dKE1 < tol1) and (dKE2 < tol2):
             stable_steps += 1
          else:
             stable_steps = 0
         
          if stable_steps >= 5:
             # add orbital velocity to each star
             relax = False
             print(f"Both stars reached equilibrium at t = {t:.2f}")

    KE1_old = KE1
    KE2_old = KE2
    
    vel += acc * dt / 2                 # (1/2) kick
    pos += vel * dt                     # drift

    acc = sp.getAcc(pos, vel, M1, M2, N1, lmbda1, lmbda2, h1, h2, G, k1, k2, n1, n2, nu1, nu2, relax=relax) # update accelerations
    vel += acc * dt / 2                 # (1/2) kick
    
    t += dt                             # update time

    rho = sp.getDensity(pos, pos, m, h) # get density for plotting
    brightness = rho.flatten()          # for visibility
            
    x, y, cv = sp.getSurface(pos, brightness, 0.05)
            
    u = v = np.linspace(-1, 1, 150)
    u, v = np.meshgrid(u, v)
    sig = sp.sqVisi(u, v, x, y, d, lam, cv)     # the square visibility
    
    # plot in real time
    if plotRealTime or (i == Nt - 1):           
       plt.sca(ax1)
       plt.cla()
       plt.scatter(x, y, c=cv, cmap='autumn')
       ax1.set_xlabel("Sky along x axis", fontsize=12, fontweight='bold')
       ax1.set_ylabel("Sky along y axis", fontsize=12, fontweight='bold')
       ax1.set_title("Stars in Sky with Time", fontsize=14, fontweight='bold')
       ax1.set(xlim=(-10, 10), ylim=(-10, 10))
       ax1.set_aspect("equal", "box")
       ax1.set_xticks([-10, 0, 10])
       ax1.set_yticks([-10, 0, 10])
       ax1.set_facecolor("black")
       ax1.set_facecolor((0.1, 0.1, 0.1))
       # Make tick labels bold
       ax1.tick_params(axis='both', labelsize=11)
       for label in ax1.get_xticklabels() + ax1.get_yticklabels():
           label.set_fontweight('bold')

       plt.sca(ax2)
       plt.cla()
       im = ax2.imshow(sig)
       cbar.update_normal(im)
       ax2.set_xlabel("Baseline along u axis", fontsize=12, fontweight='bold')
       ax2.set_ylabel("Baseline along v axis", fontsize=12, fontweight='bold')
       ax2.set_title("Square of the Visibility on Earth", fontsize=14, fontweight='bold')
       # Make tick labels bold
       ax2.tick_params(axis='both', labelsize=11)
       for label in ax2.get_xticklabels() + ax2.get_yticklabels():
           label.set_fontweight('bold')
       plt.pause(0.001)
            
# Save the last figure
plt.savefig("sph.png", dpi=240)
plt.show()

