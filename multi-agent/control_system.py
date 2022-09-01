#Non Linear Model Predictive Controls
#Turing Lab
#Step One: Import all usefull libraries:
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from scipy.optimize import minimize, Bounds

#-------------------------------------------------------------------------
#Helper Functions: Plotting Functions

def plot_robot_and_obstacles(robot, obstacles, robot_radius, num_steps, sim_time, filename):
    fig = plt.figure("Non Linear MPC for Robust Collision Avoidance")
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title("Non Linear MPC for Robust Collision Avoidance, Goal Point: (1, 1)", fontsize = 8)
    line, = ax.plot([], [], '--r')

    robot_patch = Circle((robot[0, 0], robot[1, 0]),
                         robot_radius, facecolor='green', edgecolor='black')
    obstacle_list = []
    for obstacle in range(np.shape(obstacles)[2]):
        obstacle = Circle((0, 0), robot_radius,
                          facecolor='red', edgecolor='black')
        obstacle_list.append(obstacle)

    def init():
        ax.add_patch(robot_patch)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle)
        line.set_data([], [])
        return [robot_patch] + [line] + obstacle_list

    def animate(i):
        robot_patch.center = (robot[0, i], robot[1, i])
        for j in range(len(obstacle_list)):
            obstacle_list[j].center = (obstacles[0, i, j], obstacles[1, i, j])
        line.set_data(robot[0, :i], robot[1, :i])
        return [robot_patch] + [line] + obstacle_list

    init()
    step = (sim_time / num_steps)
    for i in range(num_steps):
        animate(i)
        plt.pause(step)

    if not filename:
        return

    ani = animation.FuncAnimation(
        fig, animate, np.arange(1, num_steps), interval=200,
        blit=True, init_func=init)

    ani.save(filename, "ffmpeg", fps=5, dpi=400)

def plot_robot(robot, timestep, radius=1, is_obstacle=False):
    if robot is None:
        return
    center = robot[:2, timestep]
    x = center[0]
    y = center[1]
    if is_obstacle:
        circle = plt.Circle((x, y), radius, color='red', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], '--r',)
    else:
        circle = plt.Circle((x, y), radius, color='green', ec='black')
        plt.plot(robot[0, :timestep], robot[1, :timestep], 'blue')

    plt.gcf().gca().add_artist(circle)

#------------------------------------------------------------------------

#Step Three:
#Push Back Hard coded obstacles
def create_obstacles(sim_time, num_timesteps):

    v = -2
    p0 = np.array([5, 12])
    obst = create_robot(p0, v, np.pi/2, sim_time,
                        num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = obst

    v = 2
    p0 = np.array([0, 5])
    obst = create_robot(p0, v, 0, sim_time, num_timesteps).reshape(
        4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    v = 2
    p0 = np.array([10, 10])
    obst = create_robot(p0, v, -np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                                num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    v = 2
    p0 = np.array([7.5, 2.5])
    obst = create_robot(p0, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                                num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    return obstacles

def create_robot(p0, v, theta, sim_time, num_timesteps):
    # Creates obstacles starting at p0 and moving at v in theta direction
    t = np.linspace(0, sim_time, num_timesteps)
    theta = theta * np.ones(np.shape(t))
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    v = np.stack([vx, vy])
    p0 = p0.reshape((2, 1))
    p = p0 + np.cumsum(v, axis=1) * (sim_time / num_timesteps)
    p = np.concatenate((p, v))
    return p

#Step Four: Initalize all default simulation parameters
SIM_TIME = 8 
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2

# collision cost parameters
Qc = 5.
kappa = 4.

# nmpc parameters
HORIZON_LENGTH = int(4)
NMPC_TIMESTEP = 0.3
upper_bound = [(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2
lower_bound = [-(1/np.sqrt(2)) * VMAX] * HORIZON_LENGTH * 2

def compute_velocity(robot_state, obstacle_predictions, xref):
    u0 = np.random.rand(2*HORIZON_LENGTH)
    def cost_fn(u): return total_cost(
        u, robot_state, obstacle_predictions, xref)

    bounds = Bounds(lower_bound, upper_bound)

    res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds)
    velocity = res.x[:2]
    return velocity, res.x

def compute_xref(start, goal, number_of_steps, timestep):
    dir_vec = (goal - start)
    norm = np.linalg.norm(dir_vec)
    if norm < 0.1:
        new_goal = start
    else:
        dir_vec = dir_vec / norm
        new_goal = start + dir_vec * VMAX * timestep * number_of_steps
    return np.linspace(start, new_goal, number_of_steps).reshape((2*number_of_steps))

def total_cost(u, robot_state, obstacle_predictions, xref):
    x_robot = update_state(robot_state, u, NMPC_TIMESTEP)
    c1 = tracking_cost(x_robot, xref)
    c2 = total_collision_cost(x_robot, obstacle_predictions)
    total = c1 + c2
    return total

def tracking_cost(x, xref):
    return np.linalg.norm(x-xref)

def total_collision_cost(robot, obstacles):
    total_cost = 0
    for i in range(HORIZON_LENGTH):
        for j in range(len(obstacles)):
            obstacle = obstacles[j]
            rob = robot[2 * i: 2 * i + 2]
            obs = obstacle[2 * i: 2 * i + 2]
            total_cost += collision_cost(rob, obs)
    return total_cost

def collision_cost(x0, x1):
    d = np.linalg.norm(x0 - x1)
    cost = Qc / (1 + np.exp(kappa * (d - 2*ROBOT_RADIUS)))
    return cost

def predict_obstacle_positions(obstacles):
    obstacle_predictions = []
    for i in range(np.shape(obstacles)[1]):
        obstacle = obstacles[:, i]
        obstacle_position = obstacle[:2]
        obstacle_vel = obstacle[2:]
        u = np.vstack([np.eye(2)] * HORIZON_LENGTH) @ obstacle_vel
        obstacle_prediction = update_state(obstacle_position, u, NMPC_TIMESTEP)
        obstacle_predictions.append(obstacle_prediction)
    return obstacle_predictions

def update_state(x0, u, timestep):
    N = int(len(u) / 2)
    lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
    kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

    new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * timestep

    return new_state

# Step Five: Implement Step Next Function for simulation
def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    start = np.array([7, 8])
    p_desired = np.array([1, 1])

    robot_state = start
    robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))

    for i in range(NUMBER_OF_TIMESTEPS):
        # predict the obstacles' position in future
        obstacle_predictions = predict_obstacle_positions(obstacles[:, i, :])
        xref = compute_xref(robot_state, p_desired,
                            HORIZON_LENGTH, NMPC_TIMESTEP)
        # compute velocity using nmpc
        vel, velocity_profile = compute_velocity(
            robot_state, obstacle_predictions, xref)
        robot_state = update_state(robot_state, vel, TIMESTEP)
        robot_state_history[:2, i] = robot_state

    plot_robot_and_obstacles(
        robot_state_history, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)

if __name__ == "__main__":
    simulate("")