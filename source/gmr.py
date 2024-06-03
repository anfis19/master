import numpy as np
import sys
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import control
import os

sys.path.append('/home/anton/master')

# Riepybdlib stuff:
from riepybdlib import statistics as rs
from riepybdlib import manifold as rm
import riepybdlib.data as pbddata  # Some data to showcase the toolbox
import riepybdlib.plot as pbdplt   # Plot functionality (relies on matplotlib)
import riepybdlib.s2_fcts as s2_fcts

from admittance import *
import roboticstoolbox as rtb
from spatialmath import *
from spatialmath.base import q2r, r2q, angvec2r
import swift

# Define manifolds:
m_e1 = rm.get_euclidean_manifold(1)
m_e2 = rm.get_euclidean_manifold(2)
m_e3 = rm.get_euclidean_manifold(3)

m_s1 = rm.get_s1_manifold()
m_s2 = rm.get_s2_manifold()
m_s3  = rm.get_quaternion_manifold()
m_s3 = m_e1 * m_s3
m1_pos = m_e3
m1_ori = m_s3

# For m2, we need a representation of S1 manifold.
m2_pos = m_s1 * m_e2
m2_pos_time = m_e1 * m2_pos

m3_pos = m_s2 * m_e1
m3_ori = m_s3
m3_pos_time = m_e1 * m3_pos

m_time = m_e1 * m_e3
m_t_s2 = m_e1 * m_s2

# 2D manifolds
m1_2d_pos_time = m_e1 * m_e2
m2_2d_pos = m_s1 * m_e1
m2_2d_pos_time = m_e1 * m2_2d_pos

class Gmr:
    def __init__(self, points):
        ''' Initialisation of the class
            Takes the data as input.
        '''
        self.points = points

    def create_gaussians(self, manifold, cylindrical_axis=None):
        ''' Create a gaussian distribution for each point on a given manifold
            Adds mean and sigma of gaussians to vectors.
        '''
        self.mean = []
        self.sigma = [] 

        gmm = rs.GMM(manifold, 10)
        print(type(self.points), type(m2_2d_pos_time.id_elem))
        gmm.kmeans(self.points, maxsteps=500)
        lik,avglik = gmm.fit(self.points, reg_lambda=1e-2, maxsteps = 500)
        results = []
        i_in  = 0 # Input manifold index
        i_out = [1,2] # Output manifold index
        x_in = self.points
        

        if manifold == m_time:
            i_out = [1] # Output manifold index
            for idx, p in enumerate(list(self.points)):
                results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                self.sigma.append(results[idx].sigma)
                # print("Sigma", results[idx].sigma)
                mu = (p[i_in], results[idx].mu)
                self.mean.append(mu)

        if manifold == m3_pos_time:
                i_out = [1,2] # Output manifold index
                for idx, p in enumerate(list(self.points)):
                    results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                    self.sigma.append(results[idx].sigma)
                    mu = (p[i_in], results[idx].mu)
                    self.mean.append(mu)

                self.mean = M3toR3(self.mean)

        if manifold == m2_pos_time:
                i_out = [1,2] # Output manifold index
                for idx, p in enumerate(list(self.points)):
                    results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                    self.sigma.append(results[idx].sigma)
                    mu = (p[i_in], results[idx].mu)
                    self.mean.append(mu)
                self.mean = M2toR3(self.mean, axis=cylindrical_axis)

        if manifold == m1_2d_pos_time:
            i_out = [1] # Output manifold index
            for idx, p in enumerate(list(self.points)):
                results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                self.sigma.append(results[idx].sigma)
                mu = (p[i_in], results[idx].mu)
                self.mean.append(mu)

        if manifold == m2_2d_pos_time:
            i_out = [1,2] # Output manifold index
            for idx, p in enumerate(list(self.points)):
                results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                self.sigma.append(results[idx].sigma)
                mu = (p[i_in], results[idx].mu)
                self.mean.append(mu)
            self.mean = M2toR2_2D(self.mean)

    def get_mean(self, demonstration, manifold, point):
        ''' Takes a list of points at a time step, and returns the mean on a specified manifold.
        '''
        g = rs.Gaussian(manifold).mle(demonstration)
        mu = g.mu
        e = manifold.log(point, mu)
        Q = np.eye(3)
        e = e[1:].reshape(3,1)
        c = np.matmul(np.matmul(e.T, Q), e)
        return mu

    def best_fit(self):
        # Returns the index of the gaussian with the smallest determinant
        min = np.linalg.det(self.gaussians[0])
        idx_min = 0
        for idx, cov in enumerate(self.gaussians):
            det = np.linalg.det(cov)
            if det < min:
                min = det
                idx_min = idx
        return idx
    # Construct gaussians in different manifolds

    def lqr_path(self, dim = 3):
        '''Computes LQR at each timestep on manifold
        '''
        A = np.array([np.zeros(shape=(dim,dim)), np.zeros(shape=(dim,dim)), np.zeros(shape=(dim,dim)),np.zeros(shape=(dim,dim))]).reshape(2*dim,2*dim)
        A[:dim,dim:] = np.identity(n=dim,dtype=float)
        B = np.array([np.zeros(shape=(dim,dim)), np.identity(n=dim,dtype=float)]).reshape(2*dim,dim)
        R = np.identity(n=dim,dtype=float) * 0.01
        Q = []
        for i in range(len(self.sigma)):
            tmp = np.array(self.sigma[i])
            tmp = tmp[0:dim,0:dim]
            # copy upper half and insert into lower half
            upper = np.triu(tmp)
            tmp = upper + upper.transpose() - np.diag(np.diag(tmp))

            tmp = np.linalg.inv(tmp)
            Z = np.zeros((dim,dim),dtype=int) # Create off-diagonal zeros array
            tmp = np.asarray(np.bmat([[tmp, Z], [Z, tmp]]))
            Q.append(tmp)

        Q = np.array(Q).round(5)

        self.D = []
        self.K = []
        self.u = []
        for i in range(len(Q)):
            k_temp, S, E = control.lqr(A, B, Q[i], R)
            self.K.append(k_temp[0:dim,0:dim] * np.identity(n=dim))
            self.D.append(k_temp[0:dim,dim:2*dim] * np.identity(n=dim))
        np.linalg.cholesky(self.K)
        np.linalg.cholesky(self.D)


def R2toM2_2D(data):
    '''Convert euclidean R^2 list of data to the format of M2 manifold (polar coordinates)
    Input: (t, point)
    Output: (t, (rho), (phi))'''
    output = []
    for point in data:
        t = point[0]
        x = point[1][0]
        y = point[1][1]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        if x == 0:
            if y > 0:
                phi = np.pi / 2
            else:
                phi = -np.pi / 2
        rho = np.array([rho])
        phi = np.array([phi])
        new_point = [t, phi, rho]
        output.append(tuple(new_point))
    return output

def M2toR2_2D(data):
    '''Convert polar list of data to the format of R2 manifold
    Input: (t, (rho),(phi))
    Output: (t, point)
    ''' 
    output = []
    for point in data:
        point = np.array(point) 
        t = point[0]
        rho = point[-1][-1]
        phi = point[-1][0]
        if rho == 0:
            print('Warning')
            x = 0  
            y = 0
        else:
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
        new_point = [t, np.array([x[0],y[0]])]
        output.append(tuple(new_point))
    return output

def R3toM3(data):
    '''Convert euclidean R^3 list of data to the format of M3 manifold
    Input: (t, point)
    Output: (t, direction 3x1, radius 1x1)'''
    output = []
    for point in data:
        t = point[0]
        radius = np.linalg.norm(point[1])
        direction = point[1]/radius
        radius = np.linalg.norm(point[1]) + np.random.uniform(-0.01, 0.01)
        radius = np.array([radius])
        new_point = [t, direction, radius]
        output.append(tuple(new_point))
    return output

        
def M3toR3(data):
    '''Convert M3 list of data to the format of R3
    Input: (t, direction 3x1, radius 1x1)
    Output: (t, point)
    '''
    output = []
    for point in data:
        point = np.array(point) 
        t = point[0]
        radius = point[-1][-1]
        vector = point[1][0]*radius
        new_point = [t, vector]
        output.append(tuple(new_point))
    return output

def R3toM2(data, axis='z'):
    """
    Converts time-series data with Cartesian coordinates (t, x, y, z) to cylindrical coordinates (t, rho, phi, z).

    Args:
        data: A list of tuples, each containing:
            - A NumPy array of shape (1,) representing the time point.
            - A NumPy array of shape (3,) representing the Cartesian coordinates (x, y, z).
        axis: The axis of rotation ('x', 'y', or 'z'). Default is 'z'.

    Returns:
        A list of tuples, each containing:
            - The original time point (NumPy array).
            - A NumPy array of shape (3,) representing the cylindrical coordinates (rho, phi, z).
    """
    
    cylindrical_data = []

    for t, xyz in data:
        if axis == 'z':
            x, y, z = xyz[0], xyz[1], xyz[2]
        elif axis == 'x':
            z, y, x = xyz[0], xyz[1], xyz[2]  # Swap axes
        elif axis == 'y':
            x, z, y = xyz[0], xyz[1], xyz[2]  # Swap axes
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        if axis == 'z':
            cylindrical_coords = np.array([phi, z])
        elif axis == 'x':
            cylindrical_coords = np.array([phi, z])
        else:  # axis == 'y'
            cylindrical_coords = np.array([phi, z])

        cylindrical_data.append((t, rho, cylindrical_coords))  # Output format: (t, rho, [phi, z/x/y])print(cylindrical_data)
    return cylindrical_data

def M2toR3(cylindrical_data, axis='z'):
    """
    Converts time-series data with cylindrical coordinates (t, rho, [phi, z]) to Cartesian coordinates (t, x, y, z).

    Args:
        cylindrical_data: A list of tuples, each containing:
            - A NumPy array of shape (1,) representing the time point.
            - The radial distance rho (float).
            - A NumPy array of shape (2,) representing [phi, z] (or the appropriate coordinates depending on the axis).
        axis: The axis of rotation ('x', 'y', or 'z'). Default is 'z'.

    Returns:
        A list of tuples, each containing:
            - The original time point (NumPy array).
            - A NumPy array of shape (3,) representing the Cartesian coordinates (x, y, z).
    """
    
    cartesian_data = []

    for t, coords in cylindrical_data:
        rho, phi, z_or_x_or_y = coords[0], coords[1][0], coords[1][1]

        if axis == 'z':
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            z = z_or_x_or_y
        elif axis == 'x':
            z = rho * np.cos(phi)
            y = rho * np.sin(phi)
            x = z_or_x_or_y
        elif axis == 'y':
            x = rho * np.cos(phi)
            z = rho * np.sin(phi)
            y = z_or_x_or_y
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

        cartesian_data.append((t, np.array([x, y, z])))

    return cartesian_data

def generateTrajectory(gmrObject):
    '''Generates a trajectory using an admittance controller.
    Needs a Gmr object where the GMR has been done.
    '''
    freq = 500.0

    mass = np.diag([1, 1, 1])
    stiff = np.diag([250, 250, 250])
    damp = np.diag([100, 100, 100])

    robot = rtb.models.UR5()
    adm = Admittance(robot, freq)

    T = np.eye(4)
    T[:3, 3] = gmrObject.mean[0][1]
    T[3, :] = [0, 0, 0, 1]

    trajectory = []
    for i in range(len(gmrObject.mean)):
        T[:3, 3] = gmrObject.mean[i][1]
        damp = gmrObject.D[i]
        stiff = gmrObject.K[i]
        wrench = np.zeros(shape=(6,1))
        rotatedWrench = wrench
        compliant = adm.controller(SE3(T), rotatedWrench, mass=mass, damp=damp, stiff=stiff)
        trajectory.append(compliant[:3, 3])
    return np.array(trajectory)
   
def main():
    dimensions = '3D' # or '2D'
    stacked = False # Should 3D data be stacked letters
    letters = ['A','S','C','J'] # Letters for testing
    filepath = "source/experiments_gmr/ur5/" # Folder for experiment 
    robot_data = True
    if robot_data:
        letters = ['A'] # To only run loop once
        demo_path = 'source/Demos_ur5/'

    for letter in letters:
        if dimensions == '3D':
            dems = pbddata.get_letter_dataS2(letter=letter,n_samples=7,use_time=True)
            data = [point for dem in dems for point in dem]
            data = sorted(data, key=lambda x: x[0][0])

            if robot_data:
                demos = []
                for file in os.listdir(demo_path):
                    filename = os.fsdecode(file)
                    if filename.endswith("short.csv"): 
                        print(filename) 
                        array = np.genfromtxt(demo_path+filename, delimiter=',', skip_header=1)
                        array = array[:,1:5]
                        demos.append(array)
                combined_demos = np.concatenate(demos, axis=0)
                data = [(np.array([row[0]]), row[1:]) for row in combined_demos]
                data = sorted(data, key=lambda x: x[0][0])


            if stacked:
                dems = pbddata.get_letter_data(letter=letter,n_samples=1,use_time=True)
                data = [point for dem in dems for point in dem]

                num_demos = 25  # Number of demonstrations

                # Create incrementing z values
                z_values = np.arange(0, num_demos * 0.1, 0.1)  # [0, 0.1, 0.2, ..., 4.9]

                result = []
                for z in z_values:
                    for row in data:
                        rand = np.random.uniform(-0.1,0.1)
                        new_row = np.append(row, z)  # Add z as a new element
                        result.append(new_row)  # No need for extra dimension

                data = [(np.array([row[0]]), row[1:]*0.1) for row in result]
                data = sorted(data, key=lambda x: x[0][0])

        if dimensions == '2D':
            dems = pbddata.get_letter_data(letter=letter,n_samples=7,use_time=True)
            data = [point for dem in dems for point in dem]
            data = [(np.array([row[0]]), row[1:]*0.1) for row in data]
            data = sorted(data, key=lambda x: x[0][0])

        # -------------------- GMM -> GMR -> LQR ------------------------
        if dimensions == '3D':
            manifold = m_time
            gmr_test_m1 = Gmr(data)
            gmr_test_m1.create_gaussians(manifold)
            gmr_test_m1.lqr_path()

            m2_data = R3toM2(data, axis='x')
            manifold = m2_pos_time
            gmr_test_m2 = Gmr(m2_data)
            gmr_test_m2.create_gaussians(manifold, 'x')
            gmr_test_m2.lqr_path()

            m2_y_data = R3toM2(data, axis='y')
            manifold = m2_pos_time
            gmr_test_y_m2 = Gmr(m2_y_data)
            gmr_test_y_m2.create_gaussians(manifold, 'y')
            gmr_test_y_m2.lqr_path()

            m2_z_data = R3toM2(data, axis='z')
            manifold = m2_pos_time
            gmr_test_z_m2 = Gmr(m2_z_data)
            gmr_test_z_m2.create_gaussians(manifold, 'z')
            gmr_test_z_m2.lqr_path()

            m3_data = R3toM3(data)
            manifold = m3_pos_time
            gmr_test_m3 = Gmr(m3_data)
            gmr_test_m3.create_gaussians(manifold)
            gmr_test_m3.lqr_path()

        if dimensions == '2D':
            manifold = m1_2d_pos_time
            gmr_test_m1_2d = Gmr(data)
            gmr_test_m1_2d.create_gaussians(manifold)
            gmr_test_m1_2d.lqr_path(dim=2)

            m2_data = R2toM2_2D(data)
            manifold = m2_2d_pos_time
            gmr_test_m2_2d = Gmr(m2_data)
            gmr_test_m2_2d.create_gaussians(manifold)
            gmr_test_m2_2d.lqr_path(dim=2)

        # -------------------- Generate trajectories ------------------------
        if dimensions == '3D':
            trajectory_m1 = generateTrajectory(gmr_test_m1)
            trajectory_m2 = generateTrajectory(gmr_test_m2)
            trajectory_y_m2 = generateTrajectory(gmr_test_y_m2)
            trajectory_z_m2 = generateTrajectory(gmr_test_z_m2)
            trajectory_m3 = generateTrajectory(gmr_test_m3)

            all_trajectories = [trajectory_m1, trajectory_m2,trajectory_y_m2, trajectory_z_m2, trajectory_m3]
            all_variance = [gmr_test_m1.sigma, gmr_test_m2.sigma, gmr_test_y_m2.sigma, gmr_test_z_m2.sigma, gmr_test_m3.sigma]
            manifold_names = ['m1', 'm2x','m2y','m2z','m3']
            for traj, var, manifold in zip(all_trajectories, all_variance, manifold_names):
                sigma_diag = np.array(var).diagonal(axis1=1, axis2=2)
                traj_var = np.concatenate((np.array(traj),sigma_diag),axis=1)
                print(traj_var.shape)
                filename = letter + '_' + manifold + '_mu_cov.csv'
                np.savetxt(filepath+filename, traj_var, delimiter=",")

        if dimensions == '2D':
            trajectory_m1 = np.array([np.concatenate((t, coord)) for t, coord in gmr_test_m1_2d.mean])[:,1:]
            print(trajectory_m1[:,1:].shape)
            print(gmr_test_m1_2d.mean[0])
            print(gmr_test_m2_2d.mean[0])
            trajectory_m2 = np.array([np.concatenate((t, coord)) for t, coord in gmr_test_m2_2d.mean])[:,1:]
            all_trajectories = [trajectory_m1, trajectory_m2]
            all_variance = [gmr_test_m1_2d.sigma, gmr_test_m2_2d.sigma]
            manifold_names = ['m1', 'm2']
            for traj, var, manifold in zip(all_trajectories, all_variance, manifold_names):
                sigma_diag = np.array(var).diagonal(axis1=1, axis2=2)
                print("Sigma: ", sigma_diag.shape, " Traj: ", traj.shape)
                traj_var = np.concatenate((np.array(traj),sigma_diag),axis=1)
                print(traj_var.shape)
                filename = letter + '_' + manifold + '_mu_cov.csv'
                print(filename)
                np.savetxt(filepath+filename, traj_var, delimiter=",")


        if dimensions == '3D':
            #syntax for 3-D projection
            plt.figure(figsize=(5,5) )
            ax = plt.subplot(111,projection='3d')
            s2_fcts.plot_manifold(ax)
            plt_data = m_t_s2.swapto_tupleoflist(data)
            print(plt_data[1].shape)
            plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],'.', label='Demonstration')     # Original Data
            plt_data = m_t_s2.swapto_tupleoflist(gmr_test_m1.mean)
            plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],'.', label='Mean')     # Original Data
            label = 'Gaussian'


            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1,projection='3d')
            sigma_diag = np.array(gmr_test_m1.sigma).diagonal(axis1=1, axis2=2)
            print(np.array(gmr_test_m1.sigma).shape)
            print(np.concatenate((np.array(trajectory_m1),sigma_diag),axis=1).shape)
            ax.plot(trajectory_m1[:,0],trajectory_m1[:,1],trajectory_m1[:,2], 'blue', linewidth=2)
            ax.plot(trajectory_m2[:,0],trajectory_m2[:,1],trajectory_m2[:,2], 'black', linewidth=2)
            ax.plot(trajectory_y_m2[:,0],trajectory_y_m2[:,1],trajectory_y_m2[:,2], 'orange', linewidth=2)
            ax.plot(trajectory_z_m2[:,0],trajectory_z_m2[:,1],trajectory_z_m2[:,2], 'brown', linewidth=2)
            ax.plot(trajectory_m3[:,0],trajectory_m3[:,1],trajectory_m3[:,2], 'green', linewidth=2)
            demo = np.array(list(map(lambda x: x[1], data)))
            filename = letter + '_demonstrations.csv'
            time = np.array(list(map(lambda x: x[0], gmr_test_m1.mean)))
            print(time.shape)
            np.savetxt(filepath+filename, np.array(demo), delimiter=",")
            ax.scatter3D(demo[:,0],demo[:,1],demo[:,2], 'blue')
            ax.legend(['M1','M2_x', 'M2_y','M2_z','M3', 'Demonstration', 'Demonstrstaion after m2'])
            plt.show()

        if dimensions == '2D':
            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(trajectory_m1[:,0],trajectory_m1[:,1], 'blue', linewidth=2)
            ax.plot(trajectory_m2[:,0],trajectory_m2[:,1], 'green', linewidth=2)
            # plot_covariance_snake(trajectory_m1, gmr_test_m1.sigma)
            # for i in range(0, len(gmr_test_m1.mean), 20):
            #     # print("Mean: ", mean[1])
            #     # print("Sigma: ", sigma)
            #     plot_covariance_circle_in_3d(gmr_test_m1.mean[i][1], gmr_test_m1.sigma[i], ax)
            demo = np.array(list(map(lambda x: x[1], data)))
            filename = letter + '_demonstrations.csv'
            np.savetxt(filepath+filename, np.array(demo), delimiter=",")
            ax.scatter(demo[:,0],demo[:,1], c='red')
            ax.legend(['M1','M2_x', 'M2_y','M2_z','M3', 'Demonstration', 'Demonstrstaion after m2'])
            plt.show()




if __name__ == "__main__":
    main()