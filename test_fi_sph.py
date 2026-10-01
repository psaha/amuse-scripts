from amuse.community.fi.interface import Fi
from amuse.units import units, nbody_system
from amuse.io import read_set_from_file

BINARY_FILE = "amuse_file/binary_polytrope.amuse"

print("==============================================")
print("READING BINARY")
print("==============================================")

parts = read_set_from_file(BINARY_FILE, "amuse")

print("Input particles =", len(parts))

Mtot = 2.5 | units.MSun
Rscale = 1.0 | units.RSun

converter = nbody_system.nbody_to_si(Mtot, Rscale)

print("\n==============================================")
print("STARTING FI")
print("==============================================")

fi = Fi(converter)

fi.parameters.use_hydro_flag = True
fi.parameters.self_gravity_flag = True
fi.parameters.conservative_sph_flag = True

fi.parameters.n_smooth = 64
fi.parameters.targetnn = 32

fi.parameters.artificial_viscosity_alpha = 0.5
fi.parameters.sph_artificial_viscosity_eps = 0.01
fi.parameters.courant = 0.3

fi.parameters.integrate_entropy_flag = False
fi.parameters.sph_dens_init_flag = True

fi.parameters.n_smooth = 64
fi.parameters.targetnn = 32

fi.parameters.artificial_viscosity_alpha = 0.5
fi.parameters.sph_artificial_viscosity_eps = 0.01

fi.parameters.courant = 0.3

# Important for our test:
print("Hydro =", fi.parameters.use_hydro_flag)
print("SPH density initialization =",
      fi.parameters.sph_dens_init_flag)

print("\n==============================================")
print("ADDING AS GAS PARTICLES")
print("==============================================")

print("Gas particles =", len(fi.gas_particles))

print("\n==============================================")
print("COMMIT")
print("==============================================")

fi.gas_particles.add_particles(parts)
fi.commit_particles()

print("Gas particles after commit =",
      len(fi.gas_particles))
      
print("\n==============================================")
print("INPUT vs FI SPH STATE")
print("==============================================")

print("Input u:")
print(parts.u[:10])

print("\nFi u:")
print(fi.gas_particles.u[:10])

print("\nInput h_smooth:")
print(parts.h_smooth[:10])

print("\nFi h_smooth:")
print(fi.gas_particles.h_smooth[:10])

print("\nInput rho:")
print(parts.rho[:10])

print("\nFi rho:")
print(fi.gas_particles.rho[:10])

print("\nInput pressure:")
print(parts.pressure[:10])

print("\nFi pressure:")
print(fi.gas_particles.pressure[:10])

print("\n==============================================")
print("SPH STATE")
print("==============================================")

print("Internal energy:")
print(fi.gas_particles.u[:10])

print("\nSmoothing length:")
print(fi.gas_particles.h_smooth[:10])

print("\nDensity:")
print(fi.gas_particles.rho[:10])

print("\nPressure:")
print(fi.gas_particles.pressure[:10])

print("\n==============================================")
print("ENERGY")
print("==============================================")

print("Kinetic =", fi.kinetic_energy)
print("Potential =", fi.potential_energy)
print("Thermal =", fi.thermal_energy)
print("Total =", fi.total_energy)

fi.stop()
