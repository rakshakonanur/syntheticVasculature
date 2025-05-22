import pyvista as pv
import svcco
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import trange
import numpy as np
from svcco.implicit.load import load3d_pv
from time import perf_counter
from datetime import datetime
import os
import vtk
import sys
import subprocess

# direction_vector = np.array([1,0,0])
# direction = direction_vector.reshape(-1,1)
# print(direction)
# print(direction.shape)

# rom = int(input('Enter the order of ROM (0, 1): '))
# num_branches = int(input('Enter the number of branches: '))
# q = 4
# resolution = 80

# Use a raw string for the file path
cu = pv.Cube().triangulate().subdivide(5)
cube = svcco.surface()
cube.set_data(cu.points,cu.point_normals)
cube.solve()
cube.build()
print('domain constructed')

t = svcco.tree()
t.set_boundary(cube)
t.convex = True
t.set_root()
t.n_add(50)

t.export_1d_simulation()

# # Simulation Settings
# steady = True
# outdir = 'H:\\My Drive\\Shadden Lab Research\\Synthetic Vessel Generation\\Synthetic Vessel Generation\\Spring 2025\\1D Output\\020525\\Run7_1branches\\'
# # # now = datetime.now()
# # #folder = 'H:\\My Drive\\Shadden Lab Research\\Synthetic Vessel Generation\\Synthetic Vessel Generation\\Spring 2025\\0D Output\\012225\\' + now.strftime("%m%d%Y %H:%M:%S") + '_numBranches' + str(num_branches)
# # # os.mkdir(outdir+'\\'+folder)
# number_cardiac_cycles = 1
# number_time_pts_per_cycle = 5

# # 0D Simulation Settings
# #cerm_tree.export_0d_simulation()
# # cerm_tree.export_0d_simulation(get_0d_solver=True, path_to_0d_solver='C:\\Program Files\\SimVascular\\SimVascular\\2023-03-27\\Python3.5\\Lib\\site-packages\\svZeroDSolver\\')
