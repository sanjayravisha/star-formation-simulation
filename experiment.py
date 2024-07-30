import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tqdm
import cv2

# Constants

N = 5000  # Number of particles
t = 0  # start time of simulation
tEnd = 8  # end time for simulation
dt = 0.01  # timestep
M = 2  # star mass
R = 0.75  # star radius
h = 0.1  # smoothing length
k = 0.1  # equation of state constant
n = 1  # polytropic index
nu = 1  # damping
m = M / N  # single particle mass
lmbda = 2.01  # lambda for gravity
Nt = int(np.ceil(tEnd / dt))  # number of timesteps


# Initial Conditions


def initial():
    # Set random number generator seed

    np.random.seed(42)

    # Set randomly selected positions and velocities

    pos = np.random.randn(N, 2)
    vel = np.zeros(pos.shape)
    return pos, vel


# Gaussian Smoothing Kernel


"""
  Inputs:
    x : matrix of x positions
    y : matrix of y positions
    h : smoothing length

  Output:
    w : evaluated smoothing function
  """


def kernel(x, y, h):
    # Calculate |r|

    r = np.sqrt((x ** 2) + (y ** 2))

    # Calculate value of smoothing function

    w = (1.0 / ((h ** 3) * (np.pi ** 1.5))) * np.exp(-(r ** 2) / (h ** 2))

    return w


# Gaussian Smoothing Kernel Gradient


"""
  Inputs:
    x : matrix of x positions
    y : matrix of y positions
    h : smoothing length

  Output:
    wx, wy : evaluated gradient
  """


def grad_kernel(x, y, h):
    # Calculate |r|

    r = np.sqrt((x ** 2) + (y ** 2))

    # Calculate scalar part of gradient

    n = (-2.0 / ((h ** 5) * (np.pi ** 1.5))) * np.exp(-(r ** 2) / (h ** 2))

    # Calculate vector parts of gradient

    wx = n * x
    wy = n * y

    return wx, wy


# Magnitude of Distance in Density Equation


"""
  Inputs:
    ri : M x 2 matrix of positions
    rj : N x 2 matrix of positions

  Output:
    dx, dy : M x N matrix of separations
  """


def magnitude(ri, rj):
    M = ri.shape[0]
    N = rj.shape[0]

    # Calculate x, y of ri

    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))

    # Calculate x, y of rj

    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))

    # Calculate separations

    dx = rix - rjx.T
    dy = riy - rjy.T

    return dx, dy


# Density Equation


"""
  Inputs:
    r : M x 3 matrix of sampling locations positions
    pos : N x 3 matrix of particle positions
    m : particle mass
    h :smoothing length

  Output:
    rho : M x 1 vector of densities
  """


def density(r, pos, m, h):
    M = r.shape[0]

    # Calculate density

    dx, dy = magnitude(r, pos)
    rho = np.sum(m * kernel(dx, dy, h), 1).reshape((M, 1))

    return rho


# Pressure Equation


"""
  Inputs:
    rho : vector of densities
    k : equation of state constant
    n : polytropic index

  Output:
    P : pressure
  """


def pressure(rho, k, n):
    # Calculate pressure

    P = k * (rho ** (1 + (1 / n)))

    return P


# Acceleration Equation


"""
  Inputs:
    pos : N x 2 matrix of positions
    vel : N x 2 matrix of velocities
    m : particle mass
    h : smoothing length
    k : equation of state constant
    n : polytropic index
    lmbda : external force constant
    nu : viscosity

  Output:
    a : N x 2 matrix of accelerations
  """


def acceleration(pos, vel, m, h, k, n, lmbda, nu):
    N = pos.shape[0]

    # Calculate densities

    rho = density(pos, pos, m, h)

    # Calculate pressures

    P = pressure(rho, k, n)

    # Calculate pairwise distances and gradients

    dx, dy = magnitude(pos, pos)
    dWx, dWy = grad_kernel(dx, dy, h)

    # Add pressure contribution to acceleration

    ax = -np.sum((m * ((P / (rho ** 2)) + (P.T / (rho.T ** 2))) * dWx), 1).reshape((N, 1))
    ay = -np.sum((m * ((P / (rho ** 2)) + (P.T / (rho.T ** 2))) * dWy), 1).reshape((N, 1))

    # Pack acceleration components

    a = np.hstack((ax, ay))

    # Add external forces

    a += -(lmbda * pos) - (nu * vel)

    return a


# Folder Creation

if not os.path.exists('output'):
    os.mkdir('output')
else:
    files = glob.glob('output/*.png')
    for f in files:
        os.remove(f)

# Plot

pos, vel = initial()

for i in tqdm.tqdm(range(Nt)):
    acc = acceleration(pos, vel, m, h, k, n, lmbda, nu)
    vel += acc * dt
    pos += vel * dt
    rho = density(pos, pos, m, h)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.sca(ax)
    plt.cla()

    cval = np.minimum((rho - 3) / 3, 1).flatten()
    plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn, s=5, alpha=0.75)

    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
    ax.set_aspect('equal', 'box')
    ax.set_facecolor('black')
    ax.set_facecolor((.1, .1, .1))

    plt.savefig(f'output/{i}.png')
    plt.close()

# Animation

img_array = []

print("Reading Frames")

imgs_list = glob.glob('output/*.png')
lsorted = sorted(imgs_list, key=lambda x: int(os.path.splitext(x[7:])[0]))

for filename in tqdm.tqdm(lsorted):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('simulation-experiment-N9.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

print("Writing Frames")
for i in tqdm.tqdm(range(len(img_array))):
    out.write(img_array[i])

print("rho: ", sum(rho) / len(rho))
print("pressure: ", pressure(sum(rho) / len(rho), k, n))

out.release()
