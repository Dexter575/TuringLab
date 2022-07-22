# Note: The diffraction pattern changes with:
# a. increasing the z distance or
# b. decreasing the pinhole radius size.

from unicodedata import name
from wave_length2rgb import wavelength_to_rgb
from ctypes import util
import numpy as np
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from matplotlib.animation import PillowWriter
import pint

u = pint.UnitRegistry() # Used for Units

discrete_grid_size = np.linspace(-2,2,1600) * u.mm
xv, yv = np.meshgrid(discrete_grid_size, discrete_grid_size)
R = 0.5*u.mm
U0 = xv**2 + yv**2 < R**2
U0 = U0.astype(float)

# Spectrum:
spectrum_size = 400
spectrum_division = 50
dλ = (780- 380) / spectrum_division

def calculate_U(U0, wave_length, z):
    A = fft2(U0)
    kx = 2*np.pi * fftfreq(len(discrete_grid_size), np.diff(discrete_grid_size)[0])
    kxv, kyv = np.meshgrid(kx, kx)
    k = 2*np.pi/wave_length
    return ifft2(A*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2)))

wavelengths = []
U_all = []

alpha = 0
for idx in range(380, 750, 1):
    wavelengths.append(idx * u.nm)

    print(wavelengths[alpha])
    new_u = calculate_U(U0, wave_length = wavelengths[alpha], z=8*u.cm)
    U_all.append(new_u)
    alpha = alpha + 1
"""
cmaps = [LinearSegmentedColormap.from_list('custom', 
                                         [(0,0,0),wavelength_to_rgb(wl.magnitude)],
                                         N=256) for wl in wavelengths]
"""

new_U_all = np.zeros(shape=(1600, 1600))
for index in range(len(U_all)):
    new_U_all = new_U_all + U_all[index]

new_U_all = new_U_all / dλ

# Next Step: Plot Diffraction Pattern:
fig = plt.figure(figsize=(15, 15), dpi = 250)
fig.canvas.set_window_title('Diffraction Pattern with White Light')
plt.pcolormesh(xv,yv, np.abs(new_U_all), cmap='prism')
plt.xlabel('X-Position [mm]')
plt.ylabel('Y-Position [mm]')
plt.title('Diffraction pattern by integral over entire wave_spectrum')
plt.savefig('output_new/' + 'spectral.png')