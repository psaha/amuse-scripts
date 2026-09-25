from amuse.io import read_set_from_file

stars = read_set_from_file("amuse_file/binary_polytrope.amuse", "amuse")

print(stars)
#print()
#print("x:", stars.x)
#print("y:", stars.y)
#print("z:", stars.z)
#print("mass:", stars.mass)

#print("x unit:", stars.x.unit)
#print("mass unit:", stars.mass.unit)
#print(stars.get_attribute_names_defined_in_store())

for name in ['x', 'y', 'z', 'h_smooth', 'mass', 'rho', 'density', 'u', 'pressure', 'radius']:
    q = getattr(stars, name)
    print(f"{name:10s} : {q.unit}")
    print(f"             min = {q.min()}")
    print(f"             max = {q.max()}")
    
print("number of particles =", len(stars))
