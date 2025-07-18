import pyvista as pv
import svcco
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import trange
import numpy as np
import pandas as pd
from svcco.implicit.load import load3d_pv
from time import perf_counter
from datetime import datetime
import os
import vtk
import sys
import subprocess

class CFD:
    def __init__(self):
        pass

    def set_parameters(self,**kwargs):
        self.parameters = {}
        self.parameters['k']    = kwargs.get('k',2)
        self.parameters['q']   = kwargs.get('q',4)
        self.parameters['resolution']       = kwargs.get('resolution',50)
        self.parameters['buffer']       = kwargs.get('buffer',5)
        self.parameters['inlet_normal'] = kwargs.get('inlet_normal',np.array([0,-1,0]))#.reshape(-1,1))
        self.parameters['outlet_normal'] = kwargs.get('outlet_normal',np.array([1,0,0]))
        self.parameters['inlet'] = kwargs.get('inlet',np.array([0,0.41,0.34])) #old - [2.6,3.05,3.4], [.3,.305,.34]
        self.parameters['outlet'] = kwargs.get('outlet',np.array([-.3,-.305,.34]))
        self.parameters['num_branches'] = kwargs.get('num_branches',10)
        self.parameters['path_to_0d_solver'] = kwargs.get('path_to_0d_solver',r'/usr/local/sv/svZeroDSolver/2025-07-02/bin')
        self.parameters['path_to_1d_solver'] = kwargs.get('path_to_1d_solver',r'/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver')
        self.parameters['outdir'] = kwargs.get('outdir',"/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output")
        self.parameters['folder'] = kwargs.get('folder','tmp')
        self.parameters['geom'] = kwargs.get('geom',"/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/syntheticVasculature/files/geometry/cermRaksha_scaled_clipped_big.stl")

    def set_assumptions(self,**kwargs):
        self.homogeneous = kwargs.get('homogeneous',True)
        self.convex      = kwargs.get('convex',False)

    def implicit(self, plotVolume=False): # compute implicit domain
        pathName = self.parameters['geom']
        cermSurf = svcco.surface()
        cermSurf.load(pathName)
        cermSurf.solve()
        cermSurf.build(q=self.parameters['q'],resolution=self.parameters['resolution'],k=self.parameters['k'],buffer=self.parameters['buffer'])
        if plotVolume:
            svcco.plot_volume(cermSurf)
        print('domain constructed')
        self.cermSurf = cermSurf
    
    def tree_build(self): # build vascular tree
        cermSurf = self.cermSurf
        root = self.parameters['inlet']
        direction = self.parameters['inlet_normal']
        num_branches = self.parameters['num_branches']
        cerm_tree = svcco.tree()
        cerm_tree.set_parameters(Qterm = 0.05)
        cerm_tree.set_boundary(cermSurf)
        cerm_tree.convex = self.convex
        cerm_tree.set_root(start=root, direction=direction)
        cerm_tree.n_add(num_branches)
        self.cerm_tree = cerm_tree
    
    def forest_build(self, number_of_networks,trees_per_network): # build vascular forest
        cermSurf = self.cermSurf
        num_branches = self.parameters['num_branches']
        folder = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep
        cerm_forest = svcco.forest(boundary = cermSurf, number_of_networks = number_of_networks, trees_per_network = trees_per_network, start_points = [[self.parameters['inlet'], self.parameters['outlet']]], 
                                   directions=  [[self.parameters['inlet_normal'], self.parameters['outlet_normal']]], 
                                   root_lengths_low=[[1,1]],root_lengths_high=[[5,5]])
        cerm_forest.set_roots()
        cerm_forest.add(num_branches)
        cerm_forest.connect()
        
        p = cerm_forest.forest_copy.show()
        p.show(screenshot=folder + 'forest.png')

        cerm_forest.assign()
        #_,_,_,P,unioned =cerm_forest.export_solid(folder + 'forest')
        #unioned.save(folder + 'forest.vtk')
        # p.save_graphic(folder + 'forest.png')
        self.cerm_forest = cerm_forest


    def export_tree_0d_files(self, num_cardiac_cycles = 1, num_time_pts_per_cycle = 5, distal_pressure = 0.0): # export 0d files required for simulation
        cerm_tree = self.cerm_tree
        path_to_0d_solver = self.parameters['path_to_0d_solver']
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']
        cerm_tree.export_0d_simulation(get_0d_solver=False, path_to_0d_solver=path_to_0d_solver,outdir=outdir,folder=folder,number_cardiac_cycles=num_cardiac_cycles,number_time_pts_per_cycle=num_time_pts_per_cycle,distal_pressure=distal_pressure)

    def export_forest_0d_files(self, num_networks, num_cardiac_cycles = 1, num_time_pts_per_cycle = 5, distal_pressure = 0.0): # export 0d files required for simulation
        cerm_forest = self.cerm_forest
        path_to_0d_solver = self.parameters['path_to_0d_solver']
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']
        inlet = self.parameters['inlet']
        outlet = self.parameters['outlet']
        cerm_forest.export_0d_simulation(network_id = 0, inlets = [[inlet],[None]], get_0d_solver=True, path_to_0d_solver=path_to_0d_solver,outdir=outdir,folder=folder,number_cardiac_cycles=num_cardiac_cycles,number_time_pts_per_cycle=num_time_pts_per_cycle,distal_pressure=distal_pressure)
        self.save_data()

    def run_tree_0d_simulation(self): # run 0d simulation
        import pysvzerod
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'run.py'
        replace = "sys.path.append('None')"
        path_to_0d_solver = r"sys.path.append('/usr/local/sv/svZeroDSolver/2025-07-02/bin/svzerodsolver')"
        # # Read the file and store the modified content
        # with open(fileName, "r") as file:
        #     lines = file.readlines()  # Read all lines

        # # Modify the target line
        # with open(fileName, "w") as file:
        #     for line in lines:
        #         if line.strip() == replace:  # Match the line (strip to ignore spaces)
        #             file.write(new_path + "\n")  # Write the new line
        #         else:
        #             file.write(line)  # Keep the other lines unchanged
        # print('0D solver path changed successfully.')

        # # Run the 0D simulation
        # subprocess.run(['python', fileName])

        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([path_to_0d_solver, fileName], 
                    cwd= self.parameters['outdir'] ,
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer 

    def plot_0d_results_to_3d(self): # export 0d results to 3d
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'])
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'plot_0d_results_to_3d.py'
        subprocess.run(['python', fileName])

    def export_tree_1d_files(self,number_cardiac_cycles = 5,num_points=1000): # export 1d files required for simulation
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']
        cerm_tree = self.cerm_tree
        print(outdir)
        _,_ = cerm_tree.export_3d_solid(outdir=outdir,folder="3d_tmp",watertight=False)
        import logging

        logging.basicConfig(
            level=logging.INFO,  # or logging.DEBUG for more details
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.basicConfig(level=logging.DEBUG)
        _,_,self.data = cerm_tree.export_1d_simulation(steady = True, outdir=outdir, folder=folder,number_cariac_cycles=number_cardiac_cycles,num_points=num_points)
        self.save_data()


    def run_tree_1d_simulation(self): # run 1d simulation
        import shutil
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'])
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + '1d_simulation_input.json'

        backup_path = fileName.replace(".json", "_backup.json")  # Create a backup file path

        # Create a backup of the original file
        shutil.copy(fileName, backup_path)
        print(f"Backup saved at: {backup_path}")

        replace = "OUTPUT TEXT"
        new_path = "OUTPUT VTK 0"
        # Read the file and store the modified content
        with open(fileName, "r") as file:
            lines = file.readlines()  # Read all lines

        # Modify the target line
        with open(fileName, "w") as file:
            for line in lines:
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Keep the other lines unchanged

        # Replace the number of finite elements in each segment

        # Replace only the 5th entry after SEGMENT
        with open(fileName, "w") as file:
            for line in lines:
                if line.startswith("SEGMENT"):  # Check if the line starts with SEGMENT
                    parts = line.split()  # Split the line into parts
                    if len(parts) > 5 and parts[4] == "5":  # Ensure the 4th entry exists and is "5"
                        parts[4] = "100"  # Replace the 4th entry
                    line = " ".join(parts) + "\n"  # Reconstruct the line
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Write the modified line
        
        path_to_1d_solver = self.parameters['path_to_1d_solver']
        
        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([path_to_1d_solver, fileName], 
                    cwd= self.parameters['outdir'] ,
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer 


    def create_directory(self, rom, num_branches, is_forest): # create directory for output files
        """Creates a directory if it doesn't exist."""

        current_time = datetime.now()
        date = f"{str(current_time.month).zfill(2)}{str(current_time.day).zfill(2)}{current_time.year%2000}"

        folder = ''
        if is_forest == 1:
            folder = 'Forest_Output/'

        if rom == 1:
            folder += '1D_Output'
        elif rom == 0:
            folder += '0D_Output'

        # Current path
        dir = self.parameters['outdir']
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Directory '{dir}' created successfully.")
        directory_path = dir + '/' + folder + '/' + str(date)

        count = 0
        if os.path.exists(directory_path):
            for entry in os.scandir(directory_path):
                if entry.is_dir():
                    count += 1

        path_create = directory_path + '/' + 'Run' + str(count+1) + '_' + str(num_branches) + 'branches'
        if rom == 0:
            path_create += '/' + '0D_Input_Files'
        elif rom == 1:
            path_create += '/' + '1D_Input_Files'

        os.makedirs(path_create)
        print(f"Directory '{path_create}' created successfully.")

        self.parameters['outdir'] = directory_path + '/' + 'Run' + str(count+1) + '_' + str(num_branches) + 'branches'
        if rom == 0:
            self.parameters['folder'] = '0D_Input_Files'
        elif rom == 1:
            self.parameters['folder'] = '1D_Input_Files'

    def save_data(self):
        """" From Zach's code...
        data : ndarray
            This is the contiguous 2d array of vessel data forming the vascular
            tree. Each row represents a single vessel within the tree. The array
            has a shape (N,31) where N is the current number of vessel segments
            within the tree.
            The following descibe the organization and importance of the column
            indices for each vessel.

            Column indicies:
                    index: 0:2   -> proximal node coordinates
                    index: 3:5   -> distal node coordinates
                    index: 6:8   -> unit basis U
                    index: 9:11  -> unit basis V
                    index: 12:14 -> unit basis W (axial direction)
                    index: 15,16 -> children (-1 means no child)
                    index: 17    -> parent
                    index: 18    -> proximal node index (only real edges)
                    index: 19    -> distal node index (only real edges)
                    index: 20    -> length (path length)
                    index: 21    -> radius
                    index: 22    -> flow
                    index: 23    -> left bifurcation
                    index: 24    -> right bifurcation
                    index: 25    -> reduced resistance
                    index: 26    -> depth
                    index: 27    -> reduced downstream length
                    index: 28    -> root radius scaling factor
                    index: 29    -> edge that subedge belongs to
                    index: 30    -> self identifying index
        """
        

        fileName = self.parameters['outdir'] + '/' + self.parameters['folder'] + '/' + 'branchingData.csv'
        columnNames = ["proximalCoordsX","proximalCoordsY","proximalCoordsZ","distalCoordsX","distalCoordsY","distalCoordsZ",
                       "U1","U2","U3","V1","V2","V3","W1","W2","W3","Child1","Child2","Parent","ProximalNodeIndex","DistalNodeIndex",
                       "Length","Radius","Flow","LeftBifurcation","RightBifurcation","ReducedResistance","Depth",
                       "ReducedDownstreamLength","RootRadiusScalingFactor","Edge","Index"]
        # convert array into dataframe 
        DF = pd.DataFrame(self.data, columns=columnNames)
        # save the dataframe as a csv file 
        DF.to_csv(fileName)
        






