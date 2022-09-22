#Turing Lab

#Step One: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Step Two: Initialize Variables
plate_length = 100 # 100 * 100 m square plate length
max_iter_time = 350 # Maximum Iterations
alpha = 2 #diffusivity constant

# Step Three: Start Discretization of the 2D Grid.
delta_x = 1 #minimum resolution
delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)
u = np.empty((max_iter_time, plate_length, plate_length)) # Initialize solution: the grid of u(k, i, j)

# Step Four: Initial and Boundary Conditions
u_initial = 0 # Initial condition everywhere inside the grid

# Boundary conditions
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Set the initial condition
u.fill(u_initial)

# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right

# Plot Utilities
def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"[Turing Lab] t = {k*delta_t:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

def animate(k):
    plotheatmap(u[k], k)

# Step Five: Implement Solver to calculate solution u for everywhere in x, y as well as over time t.
# TODO: Come up with better loop unrolling methods...
def solver(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]

    return u
# Step Six: Run the Solver
u = solver(u)

#Step Seven: Animate using Matplotlib and plot utitlities.
anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("turing_lab_heat_transfer.gif")
print("Heat Equation Simulation Completed")
#-----------------------------------------------------------------------------------------------------------------------