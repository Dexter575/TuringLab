import numpy as np
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift

import matplotlib.pyplot as plt
import pint
u = pint.UnitRegistry() # Used for Units

#Calculate Inverse Fourier Transform for calculating U
def calculate_U(z, k):
    return ifft2(A*np.exp(1j*z*np.sqrt(k**2-kxv**2-kyv**2)))

"""
Experiment Number One: The Single Slit
"""

#First Step: Initialize Parameters
wavelength = 660 * u.nm     # Wavelength
slit_distance = 0.1 * u.mm  # Slit Distance

k = 2*np.pi / (wavelength) # k = 2 pi / lambda
d = 3* u.cm  # distance from the slit to the screen, z = d

#Second Step: Descritization
# Define the spatial grid [-2mm - 2mm], step size: 1600 for dx and dy.
discrete_grid_size = np.linspace(-2, 2, 1600) * u.mm
xv, yv = np.meshgrid(discrete_grid_size, discrete_grid_size)

#Third Step: Define U not, i.e., u(x, y, 0)
U_not = (np.abs(xv) < slit_distance/2) * (np.abs(yv) < 0.5 * u.mm)
U_not = U_not.astype(float) #cast to float type

# Plot U_not
plt.figure(figsize=( 5, 5) )
plt.pcolormesh(xv,yv,U_not)
plt.xlabel('X-Position [mm]')
plt.ylabel('Y-Position [mm]')
plt.show()

#----------------------------------------------------------------------------------------

#Next, create a mesh grid of k_x and k_y values so we can compute ffts of U_not
A = fft2(U_not)

# Calculate Fourier Frequiences
kx = fftfreq(len(discrete_grid_size), np.diff(discrete_grid_size)[0]) * 2 * np.pi # multiply by 2pi to get angular frequency
kxv, kyv = np.meshgrid(kx, kx)

# Plot the fourier transforms, use fftshift() for positive frequencies only. Sampling Theory :D
plt.figure(figsize=(5,5))
plt.pcolormesh(fftshift(kxv.magnitude), fftshift(kyv.magnitude), np.abs(fftshift(A)))
plt.xlabel('$k_x$ [mm$^{-1}$]')
plt.ylabel('$k_y$ [mm$^{-1}$]')
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()

# Next Step: get U by k and distance from screen and plot it.
U = calculate_U(d,k)

plt.figure(figsize=(5,5))
plt.pcolormesh(xv,yv,np.abs(U), cmap='inferno')
plt.xlabel('$x$ [mm]')
plt.ylabel('$y$ [mm]')
plt.show()

m  = np.arange(1,5,1)
x_min = np.sqrt(m**2 * wavelength**2 * d**2 / (slit_distance**2 - m**2 * wavelength**2)).to('mm')

plt.plot(discrete_grid_size, np.abs(U)[250])
[plt.axvline(discrete_grid_size.magnitude, ls='--', color='r') for discrete_grid_size in x_min]
[plt.axvline(-discrete_grid_size.magnitude, ls='--', color='r') for discrete_grid_size in x_min]
plt.xlabel('$x$ [mm]')
plt.ylabel('$u(x,y,z)$ [sqrt of intensity]')
plt.show()