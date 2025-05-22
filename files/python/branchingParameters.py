import vtk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.optimize import curve_fit

class branchingParameters:
    def __init__(self):
        self.parameters = {}
        self.data = np.zeros((1,31))

        """ From Zach's code...
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
    
    def load_csv_file(self, filename):
        """Load a CSV file from past simulations."""
        self.data = np.genfromtxt(filename, delimiter=',')
        data = np.delete(self.data, 0, axis = 0)
        self.data = np.delete(data, 0, axis = 1)

    def murray_law(self):
        """Compute the Murray's law for each segment."""
        data = self.data
        parent_radius = np.power(data[:,21],3)
        children_radius = np.zeros(np.shape(data)[0])
        for i in range(np.shape(data)[0]):
            leftDaughter, rightDaughter = data[i,15], data[i,16] 
            # print(leftDaughter, rightDaughter)
            if leftDaughter != -1:
                children_radius[i] = np.power(data[int(leftDaughter),21],3) + np.power(data[int(rightDaughter),21],3)
        return parent_radius, children_radius
    
    def plot_murray_law(self):
        parent_radius, children_radius = self.murray_law()
        parent_radius = parent_radius[children_radius != 0]
        children_radius = children_radius[children_radius != 0]
        branchIDs = np.arange(np.shape(parent_radius)[0])
        plt.figure()
        plt.plot(branchIDs, children_radius, 'o', linestyle="-")
        plt.plot(branchIDs, parent_radius, 's', linestyle="--")
        plt.xlabel('Parent Branch ID')
        plt.ylabel('Radius^3')
        plt.title('Murray Law')
        plt.legend(['Children', 'Parent'])
        plt.show()

    def branching_angles(self, show = True):
        """Compute the branching angles for each segment."""
        data = self.data
        numBranches = np.shape(data)[0]
        theta1 = np.zeros(numBranches)
        theta2 = np.zeros(numBranches)
        alpha = np.zeros(numBranches) 
        radiusRatio = np.zeros(numBranches)
        theoretical1 = np.zeros(numBranches)
        theoretical2 = np.zeros(numBranches)
        
        for i in range(numBranches):
            parentVec = data[i,12:14]
            leftDaughter, rightDaughter = data[i,15], data[i,16]
            if leftDaughter != -1:
                leftDaughterRadius, rightDaughterRadius = data[int(leftDaughter),21], data[int(rightDaughter),21]
                leftVec = data[int(leftDaughter),12:14]
                rightVec = data[int(rightDaughter),12:14]
                leftDotProd = np.dot(parentVec, leftVec)
                rightDotProd = np.dot(parentVec, rightVec)
                if leftDaughterRadius >= rightDaughterRadius:
                    theta1[i] = math.acos(leftDotProd)
                    theta2[i] = math.acos(rightDotProd)
                    radiusRatio[i] = rightDaughterRadius/leftDaughterRadius
                    alpha[i] = (rightDaughterRadius/leftDaughterRadius)**2
                else:
                    theta1[i] = math.acos(rightDotProd)
                    theta2[i] = math.acos(leftDotProd)
                    radiusRatio[i] = leftDaughterRadius/rightDaughterRadius
                    alpha[i] = (leftDaughterRadius/rightDaughterRadius)**2

                theta1[i] = math.degrees(theta1[i])
                theta2[i] = math.degrees(theta2[i])

        theta1 = theta1[theta1 != 0]
        theta2 = theta2[theta2 != 0]
        radiusRatio = radiusRatio[radiusRatio != 0]
        alpha = alpha[alpha != 0]

        radiusRatioTheoretical = np.linspace(0.01, 1, np.shape(theta1)[0])  
        alphaTheoretical = radiusRatioTheoretical**2

        def thetaDrag1(alpha): # Miniumum drag and lumen surface
            return np.degrees(np.arccos(((1 + alpha**(3/2))**(2/3) + 1 - alpha) / (2 * (1 + alpha**(3/2))**(1/3))))
        
        def thetaDrag2(alpha): # Miniumum drag and lumen surface
            return np.degrees(np.arccos(((1 + alpha**(3/2))**(2/3) - 1 + alpha) / (2 * alpha**(1/2) * (1 + alpha**(3/2))**(1/3))))
        
        def thetaPower1(alpha): # Minimum pumping power and lumen volume
            return np.degrees(np.arccos(((1 + alpha**(3/2))**(4/3) + 1 - alpha**2) / (2 * (1 + alpha**(3/2))**(2/3))))
        
        def thetaPower2(alpha): # Minimum pumping power and lumen volume
            return np.degrees(np.arccos(((1 + alpha**(3/2))**(4/3) - 1 + alpha**2) / (2 * alpha * (1 + alpha**(3/2))**(2/3))))

        # Compute branching angles for optimal values (Zamir, 1978)
        theoreticalDrag1 = thetaDrag1(alpha)
        theoreticalDrag2 = thetaDrag2(alpha)
        theoreticalPower1 = thetaPower1(alpha)
        theoreticalPower2 = thetaPower2(alpha)
                                 
        # Comput R^2 values                         
        ss_res = np.sum((theta1 - theoreticalDrag1) ** 2)
        ss_tot = np.sum((theta1 - np.mean(theta1)) ** 2)
        r2Drag1 = 1 - (ss_res / ss_tot)

        ss_res = np.sum((theta2 - theoreticalDrag2) ** 2)
        ss_tot = np.sum((theta2 - np.mean(theta2)) ** 2)
        r2Drag2 = 1 - (ss_res / ss_tot)

        ss_res = np.sum((theta1 - theoreticalPower1) ** 2)
        ss_tot = np.sum((theta1 - np.mean(theta1)) ** 2)
        r2Power1 = 1 - (ss_res / ss_tot)

        ss_res = np.sum((theta2 - theoreticalPower2) ** 2)
        ss_tot = np.sum((theta2 - np.mean(theta2)) ** 2)
        r2Power2 = 1 - (ss_res / ss_tot)

        print(theta1, theoreticalDrag1, theoreticalPower1)
        print(f'R^2 Drag1: {r2Drag1}, R^2 Drag2: {r2Drag2}, R^2 Power1: {r2Power1}, R^2 Power2: {r2Power2}')
                
        if show:
            
            plt.figure()

            plt.subplot(2,1,1)
            plt.plot(radiusRatio, theta1, 'o')
            plt.plot(radiusRatioTheoretical, thetaDrag1(alphaTheoretical), label=r'$\cos \theta_1$', color='blue')
            plt.plot(radiusRatioTheoretical, thetaPower1(alphaTheoretical), label=r'$\cos \theta_1$', color='red')
            plt.ylabel('Theta1')

            plt.subplot(2,1,2)
            plt.plot(radiusRatio, theta2, 'o')
            plt.plot(radiusRatioTheoretical, thetaDrag2(alphaTheoretical), label=r'$\cos \theta_2$', color='blue')
            plt.plot(radiusRatioTheoretical, thetaPower2(alphaTheoretical), label=r'$\cos \theta_2$', color='red')
            plt.xlabel('Ratio of Radii')
            plt.ylabel('Theta2')
            plt.show()

    def radius_depth(self, show = True):

        data = self.data
        numBranches = np.shape(data)[0]
        length = data[:,20]
        radius = data[:,21]
        rootRadiusScaling = data[:,28]
        depth = data[:,26]
        radius_ratio = np.zeros(numBranches)
        branchingConnectivity = np.zeros((numBranches, 4)) # stores the depth of parent, parent radius_ratio, left daughter radius_ratio, right daughter radius_ratio
        branchingConnectivity[:,1] = rootRadiusScaling

        for i in range(numBranches):
            length[i] = data[i,20]
            radius[i] = data[i,21]
            depth[i] = data[i,26]
            parent = data[i,17]
            if parent != -1:
                parentRadius = data[int(parent),21]
                radius_ratio[i] = radius[i]/parentRadius
                branchingConnectivity[i,0] = depth[i]
            leftDaughter, rightDaughter = data[i,15], data[i,16]
            if leftDaughter != -1:
                leftDaughterScaling, rightDaughterScaling = data[int(leftDaughter),28], data[int(rightDaughter),28]
                branchingConnectivity[i,2] = leftDaughterScaling
                branchingConnectivity[i,3] = rightDaughterScaling

        branchingConnectivity[0,0] = 0
        branchingConnectivity = sorted(branchingConnectivity, key=lambda x: x[0])
        branchingConnectivity = np.array(branchingConnectivity)
        print(branchingConnectivity)

        if show:
            plt.figure()
            plt.subplot(4,1,1)
            plt.plot(depth,rootRadiusScaling, 'o')
            plt.xticks(np.arange(0, max(depth)+1, 1))
            plt.ylabel('r/r_0')

            plt.subplot(4,1,2)
            plt.plot(depth,radius_ratio, 'o')
            plt.xticks(np.arange(0, max(depth)+1, 1))
            plt.ylabel('r/r_p')

            plt.subplot(4,1,3)
            plt.plot(depth,radius, 'o')
            plt.xticks(np.arange(0, max(depth)+1, 1))
            plt.ylabel('Radius')    

            plt.subplot(4,1,4)
            plt.plot(depth,length, 'o')
            plt.xticks(np.arange(0, max(depth)+1, 1))
            plt.xlabel('Depth')
            plt.ylabel('Length')
            plt.show()

            plt.figure()
            plt.plot(0,1, 'o', color='black')
            # plt.plot(branchingConnectivity[:,0],branchingConnectivity[:,1], color='black')

            colors = cm.plasma(np.linspace(0, 1, int(np.max(depth))))
            cnt = 0
            order = int(branchingConnectivity[0,0])

            for i in range(numBranches):
                if branchingConnectivity[i,2] != 0:
                    if order != branchingConnectivity[i,0]:
                        cnt += 1
                        order = int(branchingConnectivity[i,0])
                    plt.plot(branchingConnectivity[i,0]+1,branchingConnectivity[i,2], 'o', color = 'blue')
                    plt.plot(branchingConnectivity[i,0]+1,branchingConnectivity[i,3], 'o', color = 'blue')
                    plt.plot([branchingConnectivity[i,0],branchingConnectivity[i,0]+1],[branchingConnectivity[i,1],branchingConnectivity[i,2]], color=colors[cnt], label = f'Depth {i+1}')
                    plt.plot([branchingConnectivity[i,0],branchingConnectivity[i,0]+1],[branchingConnectivity[i,1],branchingConnectivity[i,3]], color=colors[cnt])
    
            plt.xticks(np.arange(0, max(depth)+1, 1))
            plt.xlabel('Depth')
            plt.ylabel('r/r_0')
            # plt.legend()
            plt.show()
    
    # def load_vtp_file(self, filename): 
    #     """Load a VTP (VTK PolyData) file from past simulations."""
    #     reader = vtk.vtkXMLPolyDataReader()
    #     reader.SetFileName(filename)
    #     reader.Update()
    #     self.polydata = reader.GetOutput()

    # def extract_branch_data(self):
    #     """Extract branch lengths, diameters, and angles from the VTP file."""
    #     branch_lengths = self.extract_cell_data("length")
    #     branch_area = self.extract_cell_data("area")
    #     # branch_diameters = [2 * (area[0] / np.pi) ** 0.5 for area in branch_area]

    #     points = self.extract_points()
    #     branch_angles = self.compute_branch_angles(points)
    #     self.parameters['branch_lengths'] = branch_lengths
    #     self.parameters['branch_area'] = branch_area
    #     self.parameters['branch_angles'] = branch_angles
    #     return branch_lengths, branch_area, branch_angles
        
    
    # def extract_cell_data(self, array_name):
    #     """Compute the length of each segment."""
    #     polydata = self.polydata

    #     numBranches = polydata.GetCellData().GetArray("BranchId").GetRange()[1] + 1
    #     branch = [None]*int(numBranches)
        
    #     for i in range(int(numBranches)):
    #         branch[i] = polydata.GetCellData().GetArray(array_name).GetValue(i)

    #     return branch
    
    # def extract_points(self):
    #     """Extract point coordinates from PolyData."""
    #     polydata = self.polydata
    #     points = polydata.GetPoints()
    #     return [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]

    # def compute_branch_angles(self, points):
    #     """Compute angles between branches based on vector directions."""

    #     polydata = self.polydata
    #     lines = polydata.GetLines()
    #     angles = []

    #     id_list = vtk.vtkIdList()
    #     for i in range(polydata.GetNumberOfCells()):
    #         polydata.GetCellPoints(i, id_list)
    #         if id_list.GetNumberOfIds() >= 2:
    #             # Compute vector direction
    #             p1 = np.array(points[id_list.GetId(0)])
    #             p2 = np.array(points[id_list.GetId(1)])
    #             direction = p2 - p1

    #             # Compute angle with the previous branch if applicable
    #             if len(angles) > 0:
    #                 prev_direction = angles[-1]["vector"]
    #                 angle_rad = np.arccos(np.dot(direction, prev_direction) /
    #                                     (np.linalg.norm(direction) * np.linalg.norm(prev_direction)))
    #                 angles.append({"branch": i, "angle_deg": np.degrees(angle_rad), "vector": direction})
    #             else:
    #                 angles.append({"branch": i, "angle_deg": None, "vector": direction})
    #     print(angles)
    #     return [a["angle_deg"] for a in angles if a["angle_deg"] is not None]
    
    # def plot_branch_data(self):

    #     branchIDs = np.arange(len(self.parameters['branch_lengths']))
    #     plt.figure()
        
    #     plt.subplot(3,1,1)
    #     plt.bar(branchIDs,self.parameters['branch_lengths'], color='blue', alpha=0.7, edgecolor='black')
    #     plt.title('Branch Lengths')

    #     # Second plot
    #     plt.subplot(3, 1, 2)
    #     plt.bar(branchIDs,self.parameters['branch_area'], color='green', alpha=0.7, edgecolor='black')
    #     plt.title('Branch Area')

    #     # Third plot
    #     plt.subplot(3, 1, 3)
    #     plt.bar(branchIDs[:-1],self.parameters['branch_angles'], color='red', alpha=0.7, edgecolor='black')
    #     plt.title('Branch Angles')

    #     # Adjust layout
    #     plt.tight_layout()
    #     plt.show()
    

# Example usage
bp = branchingParameters()
filename = "1D Output\\021425\\Run1_25branches\\1D Input Files\\branchingData.csv"  
polydata = bp.load_csv_file(filename)
bp.plot_murray_law()
bp.branching_angles()
bp.radius_depth()

# segment_lengths, segment_area, segment_angles = bp.extract_branch_data()
# # segment_angles = bp.compute_segment_angles()
# # diameters = bp.extract_diameters()

# print("Branch Lengths:", segment_lengths)
# print("Branch Area:", segment_area)
# print("Segment Bifurcation Angles:", segment_angles)

# bp.plot_branch_data()

