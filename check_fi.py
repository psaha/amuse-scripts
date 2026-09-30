from amuse.community.fi.interface import Fi
from amuse.units import units, nbody_system

Mtot = 2.5 | units.MSun
Rscale = 1.0 | units.RSun
converter = nbody_system.nbody_to_si(Mtot, Rscale)

fi = Fi(converter)

print("\n==============================================")
print("FI INTERFACE DIAGNOSTIC")
print("==============================================")

# ------------------------------------------------
# PARAMETERS
# ------------------------------------------------
print("\n==============================================")
print("FI PARAMETERS")
print("==============================================")

print(fi.parameters)

# ------------------------------------------------
# PARTICLE ATTRIBUTES
# ------------------------------------------------
print("\n==============================================")
print("FI PARTICLE ATTRIBUTES BEFORE COMMIT")
print("==============================================")

print(fi.particles.get_attribute_names_defined_in_store())

# ------------------------------------------------
# METHODS OF FI OBJECT
# ------------------------------------------------
print("\n==============================================")
print("FI OBJECT METHODS")
print("==============================================")

for name in dir(fi):
    if not name.startswith("_"):
        print(name)

# ------------------------------------------------
# METHODS OF PARTICLE SET
# ------------------------------------------------
print("\n==============================================")
print("FI PARTICLE METHODS")
print("==============================================")

for name in dir(fi.particles):
    if not name.startswith("_"):
        if any(word in name.lower() for word in
               ["energy", "density", "rho", "pressure",
                "smooth", "hydro", "entropy", "internal", "u"]):
            print(name)

# ------------------------------------------------
# STOP
# ------------------------------------------------
fi.stop()
