import glob
import matplotlib.pyplot as plt
from PIL import Image

files = sorted(glob.glob("fi_frames/frame_*.png"))

images = [Image.open(f) for f in files]

images[0].save(
    "binary_sph_evolution.gif",
    save_all=True,
    append_images=images[1:],
    duration=500,
    loop=0
)

print("Saved binary_sph_evolution.gif")
