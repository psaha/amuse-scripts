import glob
import matplotlib.pyplot as plt
from PIL import Image

#files = sorted(glob.glob("fi_frames/frame_*.png"))
files = sorted(glob.glob("surface_interferometry/surface_visibility_*.png"))

images = [Image.open(f) for f in files]

images[0].save(
    "surface_visibility_.gif",
    save_all=True,
    append_images=images[1:],
    duration=500,
    loop=0
)

print("Saved surface_visibility_.gif")
