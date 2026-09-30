import matplotlib.pyplot as plt
import numpy as np

# USER PARAMETERS
INPUT_FILE = "amuse_file/visible_surface_particles_single.csv"
OUTPUT_FILE = "plot/stellar_surface_single.png"
IMAGE_SIZE = 1024                                    # Number of image pixels in each direction.
KERNEL_RADIUS = 2.5                                  # Number of smoothing lengths used for the Gaussian footprint.
GAUSSIAN_WIDTH = 0.5                                 # Controls the width of the Gaussian.
BRIGHTNESS_MIN = 0.0                                 # Minimum brightness below which particles are ignored.
DPI = 150                                            # Output image DPI.
print("Reading:", INPUT_FILE)                        # READ CSV
data = np.genfromtxt(INPUT_FILE, delimiter=",", names=True)
x = data["x_Rsun"]
y = data["y_Rsun"]
h = data["h_Rsun"]
brightness = data["brightness"]

mask = (np.isfinite(x) & np.isfinite(y) & np.isfinite(h) 
       & np.isfinite(brightness) & (h > 0) & (brightness > BRIGHTNESS_MIN)) # REMOVE INVALID VALUES

x = x[mask]
y = y[mask]
h = h[mask]
brightness = brightness[mask]
print("Particles used:", len(x))

# IMAGE EXTENT
xmin = np.min(x - KERNEL_RADIUS * h)
xmax = np.max(x + KERNEL_RADIUS * h)
ymin = np.min(y - KERNEL_RADIUS * h)
ymax = np.max(y + KERNEL_RADIUS * h)

# Use the same scale in x and y so the star is not distorted.
extent = max(xmax - xmin, ymax - ymin)
xc = 0.5 * (xmin + xmax)
yc = 0.5 * (ymin + ymax)
xmin = xc - extent / 2
xmax = xc + extent / 2
ymin = yc - extent / 2
ymax = yc + extent / 2

# CREATE IMAGE GRID
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
pixel_size = extent / IMAGE_SIZE
print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Pixel size:", pixel_size, "Rsun")

# DEPOSIT PARTICLES
print("Rendering particles...")
for i in range(len(x)):
    xi = x[i]
    yi = y[i]
    hi = h[i]
    bi = brightness[i]
    px = (xi - xmin) / pixel_size                            # Convert particle position to pixel coordinates
    py = (yi - ymin) / pixel_size
    sigma = GAUSSIAN_WIDTH * hi                              # Gaussian smoothing width
    radius = KERNEL_RADIUS * hi                              # Radius of kernel in physical units.
    radius_pixels = int(np.ceil(radius / pixel_size))        # Convert radius to pixels.
    ix = int(px)                                             # Pixel containing particle centre.
    iy = int(py)

    ix0 = max(0, ix - radius_pixels)                         # Image boundaries
    ix1 = min(IMAGE_SIZE - 1, ix + radius_pixels)
    iy0 = max(0, iy - radius_pixels)
    iy1 = min(IMAGE_SIZE - 1, iy + radius_pixels)
    if ix0 > ix1 or iy0 > iy1:
        continue

    xx = (np.arange(ix0, ix1 + 1) + 0.5) * pixel_size + xmin # Local pixel coordinates

    yy = (np.arange(iy0, iy1 + 1) + 0.5) * pixel_size + ymin
    XX, YY = np.meshgrid(xx, yy)
    r2 = ((XX - xi) ** 2 + (YY - yi) ** 2)                   # Distance from particle

    kernel = np.exp(-r2 / (2.0 * sigma * sigma))             # Gaussian kernel
    image[iy0:iy1 + 1, ix0:ix1 + 1] += bi * kernel           # Add particle contribution

maximum = np.max(image)
if maximum > 0:                                              # NORMALIZE
    image /= maximum

print("Saving:", OUTPUT_FILE)                                # SAVE PNG
plt.figure(figsize=(8, 8), facecolor="black")
plt.imshow(image, origin="lower", extent=[xmin, xmax, ymin, ymax],
    cmap="gray", interpolation="bilinear", vmin=0, vmax=1)

plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches="tight", pad_inches=0, facecolor="black")
plt.show()
plt.close()
print("Done.")


