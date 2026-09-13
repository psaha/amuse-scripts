import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def W(x, y, z, h):
    """
    Gausssian Smoothing kernel (3D): defines the shape of that blob (not a point particle) using a Gaussian distribution.
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        w     is the evaluated smoothing function
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    w = (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)

    return w


def gradW(x, y, z, h):
    """
    Gradient of the Gausssian Smoothing kernel (3D): blob is not uniform
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    n = -2 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3 / 2)
    wx = n * x
    wy = n * y
    wz = n * z

    return wx, wy, wz


def getPairwiseSeparations(ri, rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions: M particles in 3 coordinate system
    rj    is an N x 3 matrix of positions: N particle in 3 coordinate system
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:, 0].reshape((M, 1))          # reshape change the array into column vector
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T #ALL pairwise x-separations.
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


def getDensity(r, pos, m, h):
    """
    Get Density at sampling locations from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx, dy, dz = getPairwiseSeparations(r, pos)
    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho


def getPressure(rho, k, n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """

    P = k * rho ** (1 + 1 / n)

    return P


def getAcc(pos, vel, G, m, h, k, n, eps, nu):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """
    
    N = pos.shape[0]

    # Calculate densities at the position of the particles
    rho = getDensity(pos, pos, m, h)

    # Get the pressures
    P = getPressure(rho, k, n)

    # Get pairwise distances and gradients
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    # Add Pressure contribution to accelerations
    ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))
    
    
    # Newtonian gravity
    r2 = dx**2 + dy**2 + dz**2 + eps**2
    r3 = r2**(-3/2)

    ax -= G * m * np.sum(dx * r3, axis=1).reshape((N, 1))
    ay -= G * m * np.sum(dy * r3, axis=1).reshape((N, 1))
    az -= G * m * np.sum(dz * r3, axis=1).reshape((N, 1))

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    # velocity damping (viscosity)
    a -= nu * vel

    return a

def make_star(N, R):
    """
    Generate the initial positions of SPH particles inside a sphere of radius R
    
    N: Number of particle
    R: Radius of star in ls
    
    pos: containing the coordinates of all particles 
    """
    pos = np.random.randn(N, 3)

    r = np.sqrt(np.sum(pos**2, axis=1))

    pos *= (R / np.max(r))

    return pos
    
def project_particles(pos, m, h):
    """
    Project SPH particles onto plane of sky.

    Returns
    -------
    x, y : projected coordinates in ls
    w    : brightness weights
    """

    x = pos[:,0]
    y = pos[:,1]
    
    rho = getDensity(pos, pos, m, h)

    if rho is None:
        w = np.ones_like(x)
    else:
        w = rho.flatten()

    return x, y, w

def Visi(u, v, x, y, d, w):
    """
    Compute normalized complex visibility.

    Parameters
    ----------
    u,v : baseline coordinates in lambda units
    """

    phase = np.exp(
        -2j*np.pi*(u*x + v*y)/d)

    V = np.sum(w*phase) / np.sum(w)
    return abs(V)**2

def main():
    np.random.seed(42)
    M1 = 2.0
    M2 = 1.0

    R1 = 0.75
    R2 = 0.50

    N1 = 500
    N2 = 300

    d = 3.0
    
    # SPH parameters
    h = 0.10

    k = 0.10
    n = 1.0

    G = 1.0
    eps = 0.05

    nu = 0.01

    dt = 0.005
    tEnd = 10.0
    plotRealTime = True
    
    # Particle masses
    
    N = N1 + N2

    m = (M1 + M2) / N
    
    # Initial stars
    pos1 = make_star(N1, R1)
    pos2 = make_star(N2, R2)
    # separation
    
    pos1[:, 0] -= d/2
    pos2[:, 0] += d/2

    pos = np.vstack((pos1, pos2))
    
    # Circular orbit velocities
    omega = np.sqrt(G * (M1 + M2) / d**3)

    v1 = omega * d * M2/(M1 + M2)
    v2 = omega * d * M1/(M1 + M2)

    vel1 = np.zeros((N1, 3))
    vel2 = np.zeros((N2, 3))

    vel1[:, 1] = +v1
    vel2[:, 1] = -v2

    vel = np.vstack((vel1, vel2))
    
    # Initial acceleration
    acc = getAcc(pos, vel, G, m, h, k, n, eps, nu)
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    t = 0.0

    Nt = int(np.ceil(tEnd/dt))
    
    # MAIN LOOP
    for step in range(Nt):

        # Kick

        vel += 0.5 * acc * dt

        pos += vel * dt

        # New acceleration

        acc = getAcc(pos, vel, m, h, k, n, G, eps, nu)

        # Kick

        vel += 0.5 * acc * dt

        t += dt
        
        x, y, w = project_particles(pos, m, h)
        u = np.linspace(-200, 200, 800)
        v = np.zeros_like(u)

        sig = np.array([Visi(u, v, x, y, d, w) for ui, vi in zip(u, v)])
        if plotRealTime and step % 10 == 0:

           ax1.clear()
           ax2.clear()
           
           ax1.scatter(pos[:N1,0], pos[:N1,1], s=3, alpha=0.7, label='Star 1')

           ax1.scatter(pos[N1:,0], pos[N1:,1], s=3, alpha=0.7, label='Star 2')

           ax1.set_xlim(-60,60)
           ax1.set_ylim(-60,60)
           ax1.set_aspect('equal')
           ax1.set_title(f't = {t:.2f}')
           ax1.legend()

           ax2.plot(u, sig, lw=2)

           ax2.set_xlabel("u")
           ax2.set_ylabel("|V|")
           ax2.set_title("Visibility")

           plt.pause(0.01)

    plt.show()
    
    
if __name__ == "__main__":
    main()
