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

import cv2
u = pint.UnitRegistry() # Used for Units

discrete_grid_size = np.linspace(-2,2,1600) * u.mm
xv, yv = np.meshgrid(discrete_grid_size, discrete_grid_size)
R = 0.05*u.mm
U0 = xv**2 + yv**2 < R**2
U0 = U0.astype(float)

def calculate_U(U0, wave_length, z):
    A = fft2(U0)
    kx = 2*np.pi * fftfreq(len(discrete_grid_size), np.diff(discrete_grid_size)[0])
    kxv, kyv = np.meshgrid(kx, kx)
    k = 2*np.pi/wave_length
    return ifft2(A*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2)))

wavelengths = []
U_all = []

alpha = 0
for idx in range(507, 750, 1):
    wavelengths.append(idx * u.nm)

    #print(wavelengths[alpha])
    U_all.append(calculate_U(U0, wave_length = wavelengths[alpha], z=6*u.cm))
    alpha = alpha + 1

cmaps = [LinearSegmentedColormap.from_list('custom', 
                                         [(0,0,0),wavelength_to_rgb(wl.magnitude)],
                                         N=256) for wl in wavelengths]

for index in range(len(cmaps)):
    # Next Step: Plot Diffraction Pattern:
    fig = plt.figure(figsize=(15, 15), dpi = 250)
    fig.canvas.set_window_title('Visual Spectrum vs Fringes')
    plt.pcolormesh(xv,yv, np.abs(U_all[index]), cmap=cmaps[index], vmax=np.max(np.abs(U_all[index]))/2)
    plt.xlabel('X-Position [mm]')
    plt.ylabel('Y-Position [mm]')
    plt.title('$\lambda$={} nm'.format(wavelengths[index].magnitude))
    plt.savefig('output/' + str(wavelengths[index].magnitude) + '.png')