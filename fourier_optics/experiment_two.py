import numpy as np
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq

import matplotlib.pyplot as plt
import pint
u = pint.UnitRegistry() # Used for Units

"""
Experiment Number Two: The Double Slit
"""

#First Step: Initialize Parameters
distance_between_slits = 0.2*u.mm   # s
slit_width = 0.05* u.mm             # D
wavelength = 660 * u.nm     # Wavelength

#Second Step: Descritization
discrete_grid_size = np.linspace(-4,4,3200) * u.mm
xv, yv = np.meshgrid(discrete_grid_size, discrete_grid_size)

# Initial Field u(x, y, 0)
U_not = (np.abs(xv-distance_between_slits/2)< slit_width/2) * (np.abs(yv)<2*u.mm) + (np.abs(xv+distance_between_slits/2)< slit_width/2) * (np.abs(yv)<2*u.mm)
U_not = U_not.astype(float) #cast to float type

# Plot U_not, initial field
plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,U_not)
plt.xlabel('X-Position [mm]')
plt.ylabel('Y-Position [mm]')
plt.show()

#----------------------------------------------------------------------------------------
def calculate_U(U0, xv, yv, lam, z):
    A = fft2(U0)
    kx = 2*np.pi * fftfreq(len(discrete_grid_size), np.diff(discrete_grid_size)[0])
    kxv, kyv = np.meshgrid(kx, kx)
    k = 2*np.pi/lam
    return ifft2(A*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2)))

U = calculate_U(U_not, xv, yv, wavelength, z=5*u.cm)

# Next Step: Plot Diffraction Pattern:
plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,np.abs(U), cmap='inferno')
plt.xlabel('X-Position [mm]')
plt.ylabel('Y-Position [mm]')
plt.title("Double Slit Experiment")
plt.show()

# For good observation, we can plot by looking at cross section via center line:
central_line = np.abs(U)[250]
plt.plot(discrete_grid_size, central_line)
plt.xlabel('$x$ [mm]')
plt.ylabel('$u(x,y,z)$ [sqrt of intensity]')
plt.grid()
plt.show()