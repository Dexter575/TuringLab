# Turing_Lab :)
# Author: Muhammad Idrees, idreees575@gmail.com (note: it has three e's).

# Step Zero: Import all libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step One: Simulation Setup and Configurations

npp = 100000  # number of tries to take
ns = 300      # number of samples to keep 
neighbour_range = 0.20   # look only within 20% of the total size of the model_space

# Step Two: Generating a pdf.
#In matlab, we can use "peaks(n)" but here I will write my own function in python:
n = 100 # Number of dimension
pdf = np.zeros( [n , n] )
x = -3.
for i in range(0, n):
    y = -3.
    for j in range(0, n):
        pdf[j , i]=3.*(1 - x)**2*np.exp(-(x**2)-(y+1)**2)-10.*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1./3*np.exp(-(x+1)**2-y**2)
        if pdf[j,i] < 0:
            pdf[j,i] = pdf[j,i] * (-1) #Note: in contrast to the peaks function: all negative values are multiplied by (-1)
        y = y + 6./(n-1)
    x = x + 6./(n-1)

pdf = pdf / pdf.sum()

# Now, we Plot the 3D plot of pdf
X = np.arange(0,100 + 100./(n-1), 100./(n-1))
Y = np.arange(0,100 + 100./(n-1), 100./(n-1))
fig0 = plt.figure()

ax = fig0.gca(projection='3d')
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(Y, X, pdf,rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.1)
plt.gca().invert_xaxis()
plt.title("Probability Density Function")
plt.savefig("pdf_generation.png", dpi=600)
plt.show()

# Step Two: Random walk

# First find an initial vector x
xcur = np.array([np.floor(np.random.uniform(0, n)), np.floor(np.random.uniform(0, n))])

# Let s get moving
iis = 1
npp = 0
xa = np.zeros([ns+1,2])
xa[0,0] = xcur[0]
xa[0,1] = xcur[1]

## Ploting the path 
fig = plt.figure()
f1 = fig.add_subplot(111)  
f1.imshow(pdf.transpose(),aspect='auto',interpolation='none', animated=True)
f1.set_xlim(0,99)
f1.set_ylim(99,0)
f1.set_title('Random Walk with Near Neighbor Sampling')
plt.ion()
plt.show()

Pa = np.zeros(ns+1)
xnew = np.array([0.,0.])

# Now, we iterate over the loop
while iis <= ns:
    npp = npp+1;   
    # make a random choice for the next move
    xnew = xcur + 1
    for i in range(0, 2):
        a = np.around((np.random.uniform(0, 1) - 0.5) * n * neighbour_range) + xcur[i]
        if a <= 0:
            a = 0
        if a >= n:
            a = n-1
        xnew[i] = a

    # compare probabilities
    Pcur = pdf[int(xcur[0]), int(xcur[1])]
    Pnew = pdf[int(xnew[0]), int(xnew[1])]
   
    if Pnew >= Pcur:
        xcur = xnew
        xa[iis,0] = xcur[0]
        xa[iis,1] = xcur[1]
        Pa[iis] = Pnew
        #print('Turing Lab: Metropolis Algo-Made the %i-th move to [%i,%i] (Pnew>Pold) ' %(iis, xcur[0], xcur[1])) 
        f1.plot([xa[iis-1, 0], xa[iis,0]], [xa[iis-1, 1], xa[iis,1]],'k-')
        f1.plot([xa[iis-1, 0], xa[iis,0]], [xa[iis-1, 1], xa[iis,1]],'k+')
        plt.gcf().canvas.draw()
        plt.savefig(str(iis) + '.png', dpi = 600)
        iis = iis + 1
    
    if Pnew < Pcur: 
        P = Pnew / Pcur
        test = np.random.uniform(0,1)
        if test <= P: 
            xcur = xnew
            xa[iis,0] = xcur[0]
            xa[iis,1] = xcur[1]
            Pa[iis] = Pnew
            #print('Turing Lab: Metropolis Algo-Made the %i-th move to [%i,%i] (Pnew<Pold) ' %(iis, xcur[0], xcur[1]) )
            f1.plot([xa[iis-1, 0], xa[iis,0]], [xa[iis-1, 1], xa[iis,1]],'k-')
            f1.plot([xa[iis-1, 0], xa[iis,0]], [xa[iis-1, 1], xa[iis,1]],'k+')
            plt.gcf().canvas.draw()
            plt.savefig(str(iis) + '.png', dpi = 600)
            iis = iis + 1
plt.gcf().canvas.draw()
plt.savefig('final_result.png', dpi = 600)

# Next Step: Final sampling
fig = plt.figure()
plt.imshow(pdf.transpose(),aspect='auto', extent=[0,100,100,0],interpolation='none')
plt.plot(xa[:,0],xa[:,1],'w+')
plt.title('Density Function')
plt.savefig('density_function.png', dpi = 600)
plt.show()