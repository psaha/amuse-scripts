import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from amuse.community.fi.interface import Fi
from amuse.units import units, nbody_system
from amuse.io import read_set_from_file, write_set_to_file

BINARY_FILE = "amuse_file/binary_polytrope.amuse"
OUTPUT_DIR = "fi_snapshots"
FRAME_DIR = "fi_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

N1 = 50000
N2 = 70000

parts = read_set_from_file(BINARY_FILE, "amuse")

print("Number of particles =", len(parts))

Mtot = 2.5 | units.MSun
Rscale = 1.0 | units.RSun
converter = nbody_system.nbody_to_si(Mtot, Rscale)

fi = Fi(converter, channel_type="sockets")

fi.parameters.timestep = 100.0 | units.s
fi.parameters.periodic_box_size = 20.0 | units.RSun
fi.parameters.verbosity = 0

fi.gas_particles.add_particles(parts)
fi.commit_particles()

E0 = fi.total_energy

times = np.arange(0, 368064, 1000)

for i,t in enumerate(times):

    if t > 0:
        fi.evolve_model(t | units.s)

    snapshot = fi.gas_particles.copy()

    filename = f"{OUTPUT_DIR}/binary_{t:05d}.amuse"
    write_set_to_file(snapshot, filename, "amuse")

    x = snapshot.x.value_in(units.RSun)
    y = snapshot.y.value_in(units.RSun)
    rho = snapshot.rho.value_in(units.kg/units.m**3)

    plt.figure(figsize=(8,8))

    plt.scatter(
        x,
        y,
        c=np.log10(np.maximum(rho, 1e-20)),
        s=0.4,
        cmap="gray",
        marker="."
    )

    plt.xlabel(r"$x\ (R_\odot)$")
    plt.ylabel(r"$y\ (R_\odot)$")
    plt.title(
        f"Binary SPH evolution: "
        f"t = {fi.model_time.value_in(units.s):.0f} s"
    )

    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        f"{FRAME_DIR}/frame_{i:03d}.png",
        dpi=150
    )

    plt.close()

    E = fi.total_energy
    dE = (E-E0)/abs(E0)

    print(
        f"Frame {i:03d}: "
        f"t = {fi.model_time.value_in(units.s):.2f} s, "
        f"dE/E0 = {dE:.3e}"
    )

fi.stop()
