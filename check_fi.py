from amuse.community.fi.interface import Fi
from amuse.units import units, nbody_system

Mtot = 2.5 | units.MSun
Rscale = 1.0 | units.RSun

converter = nbody_system.nbody_to_si(Mtot, Rscale)

fi = Fi(converter)

print("\n==============================================")
print("FI INTERFACE DIAGNOSTIC")
print("==============================================")

print("\nFi parameters:")
print(fi.parameters.get_attribute_names_defined_in_store())

print("\nFi particle attributes BEFORE commit:")
print(fi.particles.get_attribute_names_defined_in_store())

print("\nFi methods:")

for name in dir(fi):
    if not name.startswith("_"):
        print(name)

fi.stop()
