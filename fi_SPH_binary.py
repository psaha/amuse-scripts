from amuse.community.fi.interface import Fi
from amuse.units import units, nbody_system
from amuse.io import read_set_from_file

BINARY_FILE = "amuse_file/binary_polytrope.amuse"

# ============================================================
# READ INITIAL BINARY
# ============================================================

print("==============================================")
print("READING INITIAL BINARY")
print("==============================================")

parts = read_set_from_file(BINARY_FILE, "amuse")

print("Total particles =", len(parts))

print("\nAvailable particle attributes:")
print(parts.get_attribute_names_defined_in_store())


# ============================================================
# BINARY MASS AND SCALE
# ============================================================

M1 = 1.0 | units.MSun
M2 = 1.5 | units.MSun

Mtot = M1 + M2

# Characteristic length scale
Rscale = 1.0 | units.RSun

converter = nbody_system.nbody_to_si(Mtot, Rscale)


# ============================================================
# START FI WITH CONVERTER
# ============================================================

print("\n==============================================")
print("STARTING FI")
print("==============================================")

fi = Fi(converter)


# ============================================================
# SPH / HYDRODYNAMICS
# ============================================================

fi.parameters.use_hydro_flag = True
fi.parameters.self_gravity_flag = True
fi.parameters.conservative_sph_flag = True

fi.parameters.n_smooth = 64
fi.parameters.targetnn = 32


# ============================================================
# ARTIFICIAL VISCOSITY
# ============================================================

fi.parameters.artificial_viscosity_alpha = 0.5
fi.parameters.sph_artificial_viscosity_eps = 0.01
fi.parameters.courant = 0.3


# ============================================================
# ADAPTIVE TIMESTEP
# ============================================================

fi.parameters.acc_timestep_flag = True
fi.parameters.acc_timestep_crit_constant = 0.25


# ============================================================
# CHECK FI UNITS
# ============================================================

print("Fi length unit =", fi.parameters.code_length_unit)
print("Fi mass unit   =", fi.parameters.code_mass_unit)
print("Fi timestep    =", fi.parameters.timestep)


# ============================================================
# CHECK INPUT PARTICLES
# ============================================================

print("\n==============================================")
print("INPUT PARTICLE UNITS")
print("==============================================")

print("Mass     =", parts.mass.unit)
print("Position =", parts.x.unit)
print("Velocity =", parts.vx.unit)
print("Density  =", parts.rho.unit)
print("Pressure =", parts.pressure.unit)
print("u        =", parts.u.unit)


# ============================================================
# ADD BINARY
# ============================================================

print("\n==============================================")
print("ADDING BINARY PARTICLES")
print("==============================================")

fi.gas_particles.add_particles(parts)

print("Particles in Fi =", len(fi.particles))


# ============================================================
# COMMIT
# ============================================================

print("\n==============================================")
print("COMMITTING PARTICLES")
print("==============================================")

fi.commit_particles()

print("Particles after commit =", len(fi.particles))

print("\nFi particle attributes:")
print(fi.particles.get_attribute_names_defined_in_store())

# ============================================================
# CHECK FI PARTICLE ATTRIBUTES
# ============================================================

print("\n==============================================")
print("FI PARTICLE ATTRIBUTES")
print("==============================================")

print(fi.particles.get_attribute_names_defined_in_store())


# ============================================================
# INITIAL ENERGY
# ============================================================

print("\n==============================================")
print("INITIAL ENERGY")
print("==============================================")

print("Kinetic  =", fi.kinetic_energy)
print("Potential =", fi.potential_energy)
print("Thermal  =", fi.thermal_energy)

E0 = (
    fi.kinetic_energy
    + fi.potential_energy
    + fi.thermal_energy
)

print("Total energy =", E0)


# ============================================================
# SHORT SPH EVOLUTION
# ============================================================

print("\n==============================================")
print("STARTING SPH EVOLUTION")
print("==============================================")

print("Initial time =", fi.model_time)

t_end = 5000.0 | units.s

print("Initial time =", fi.model_time)
print("Evolving to  =", t_end)

fi.evolve_model(t_end)

print("Final time   =", fi.model_time)

print("Final time =", fi.model_time)


# ============================================================
# FINAL STATE
# ============================================================

print("\n==============================================")
print("FINAL FI STATE")
print("==============================================")

print("Particles =", len(fi.particles))

print("Total mass =", fi.particles.mass.sum())

print("x range =",
      fi.particles.x.min(),
      fi.particles.x.max())

print("y range =",
      fi.particles.y.min(),
      fi.particles.y.max())

print("z range =",
      fi.particles.z.min(),
      fi.particles.z.max())

print("Final kinetic  =", fi.kinetic_energy)
print("Final potential =", fi.potential_energy)
print("Final thermal  =", fi.thermal_energy)

E1 = (
    fi.kinetic_energy
    + fi.potential_energy
    + fi.thermal_energy
)

print("Final total energy =", E1)

print("Relative energy change =",
      abs((E1 - E0) / E0))
      
# ============================================================
# STOP
# ============================================================
fi.stop()
