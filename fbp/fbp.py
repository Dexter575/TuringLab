"""
Computed Tomography (CT) Reconstruction
using the Filtered Back Projection (FBP) method,
with visualization using NumPy, SciPy, and Matplotlib.
"""

#Turing Lab :)
# Author: Muhammad Idrees, idreees575@gmail.com (note: it has three e's).

#Step One: Import Librariesimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

# Step 1: Create Phantom Image
image_size = 256
phantom = shepp_logan_phantom()

if phantom.shape[0] < image_size:
    pad_total = image_size - phantom.shape[0]
    pad_width = ((pad_total // 2, pad_total - pad_total // 2),  # rows
                 (pad_total // 2, pad_total - pad_total // 2))  # columns
    phantom_resized = np.pad(phantom, pad_width, mode='constant')
elif phantom.shape[0] > image_size:
    phantom_resized = resize(phantom, (image_size, image_size), anti_aliasing=True)
else:
    phantom_resized = phantom

# Step 2: Generate Sinogram
angles = np.linspace(0., 180., max(phantom_resized.shape), endpoint=False)
sinogram = radon(phantom_resized, theta=angles)

# Step 3: Reconstruction (Filtered Back Projection)
reconstruction_fbp = iradon(sinogram, theta=angles, filter_name='ramp')

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(phantom_resized, cmap='gray')
axes[0].set_title('Original Phantom')
axes[0].axis('off')

axes[1].imshow(sinogram, cmap='gray', aspect='auto')
axes[1].set_title('Sinogram (Projections)')
axes[1].set_xlabel('Projection angle (deg)')
axes[1].set_ylabel('Detector pixel')

axes[2].imshow(reconstruction_fbp, cmap='gray')
axes[2].set_title('Reconstructed Image (FBP)')
axes[2].axis('off')

plt.tight_layout()
plt.show()