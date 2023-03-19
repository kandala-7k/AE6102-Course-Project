# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 02:23:23 2023

@author: vk777
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define computational domain
nx = 41    # Number of points in x-direction
ny = 41    # Number of points in y-direction
nz = 41    # Number of points in z-direction
L = 4      # Length of the domain in x,y,z-direction
x = np.linspace(-L/2, L/2, nx)
y = np.linspace(-L/2, L/2, ny)
z = np.linspace(-L/2, L/2, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define parameters
V_inf = 1   # Magnitude of the freestream velocity
R = 1       # Radius of the circular cylinder
gamma = 4*np.pi*V_inf*R   # Circulation strength

# Initialize potential and stream function arrays
phi = np.zeros((nz, ny, nx))
psi = np.zeros((nz, ny, nx))

# Define boundary conditions
phi[:, :, 0] = V_inf*X[:, :, 0]    # Left boundary: uniform freestream
phi[:, :, -1] = V_inf*X[:, :, -1]    # Right boundary: uniform freestream
phi[0, :, :] = V_inf*Y[0, :, :]    # Top boundary: uniform freestream
phi[-1, :, :] = V_inf*Y[-1, :, :]    # Bottom boundary: uniform freestream
phi[:, 0, :] = V_inf*Z[:, 0, :]    # Front boundary: uniform freestream
phi[:, -1, :] = V_inf*Z[:, -1, :]    # Back boundary: uniform freestream

# Solve for potential and stream function using finite differences
for k in range(1, nz-1):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = np.sqrt((X[k, j, i]-R)**2 + Y[k, j, i]**2)
            theta = np.arctan2(Y[k, j, i], X[k, j, i]-R)
            phi[k, j, i] = V_inf*X[k, j, i] + gamma/(4*np.pi*r)*np.sin(theta)    # Potential
            psi[k, j, i] = -V_inf*Y[k, j, i] + gamma/(4*np.pi)*np.log(r)    # Stream function

# Plot potential and stream function contours
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contourf(X[:, :, 20], Y[:, :, 20], psi[:, :, 20], levels=50, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('phi')
plt.show()
