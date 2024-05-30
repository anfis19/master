import numpy as np
import sys
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import control

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

class Calinon:
    def __init__(self, points):
        ''' Initialisation of the class
            Takes the data as input.
        '''
        self.points = points

        # gmm_m1 = rs.GMM(m1_pos, 6)
        # print(m1_pos.np_to_manifold(points))
        # gmm_m1.kmeans(points)
        # lik,avglik = gmm_m1.fit(points, reg_lambda=1e-2, maxsteps = 500)
        # gmm_m3 = rs.GMM(m3_pos, 6)
        # gmm_m3.kmeans(points)
        # lik,avglik = gmm_m3.fit(points, reg_lambda=1e-2, maxsteps = 500)


    def create_gaussians(self, manifold, cylindrical_axis=None):
        ''' Create a gaussian distribution for each point on a given manifold
            Adds mean and sigma of gaussians to vectors.
        '''
        self.mean = []
        self.sigma = [] 
        # for i in range(0, int(len(self.points)),4):
        #     print(i)
        #     g = rs.Gaussian(m_time).mle(self.points[i:i+4])
        #     self.mean.append(g.mu)
        #     self.sigma.append(g.margin(1).sigma)
        # # print("Sigma", self.sigma[0])
        # # print("Mu: ", self.mean[0])¨
        # g = rs.Gaussian(manifold).mle(self.points)
        gmm = rs.GMM(manifold, 10)
        print(type(self.points), type(m2_2d_pos_time.id_elem))
        gmm.kmeans(self.points, maxsteps=500)
        lik,avglik = gmm.fit(self.points, reg_lambda=1e-2, maxsteps = 500)
        results = []
        i_in  = 0 # Input manifold index
        i_out = [1,2] # Output manifold index
        x_in = self.points
        
        #x_in = self.points[0:200:10]
        # print(x_in)
        # results = g.condition(x_in,i_in=i_in,i_out=i_out)
        # for idx, r in enumerate(results):
        #     self.sigma.append(r.sigma)
        #     mu = (x_in[idx][0], r.mu)
        #     self.mean.append(mu)
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
                    # print(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0].sigma)
                    self.sigma.append(results[idx].sigma)
                    # print("Sigma", results[idx].sigma)
                    mu = (p[i_in], results[idx].mu)
                    self.mean.append(mu)
                    # print(mu)
                self.mean = M3toR3(self.mean)
                # print(self.mean)

        if manifold == m2_pos_time:
                i_out = [1,2] # Output manifold index
                for idx, p in enumerate(list(self.points)):
                    results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                    # print(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0].sigma)
                    self.sigma.append(results[idx].sigma)
                    # print("Sigma", results[idx].sigma)
                    mu = (p[i_in], results[idx].mu)
                    self.mean.append(mu)
                self.mean = M2toR3(self.mean, axis=cylindrical_axis)
                # print(self.mean)

        if manifold == m1_2d_pos_time:
            i_out = [1] # Output manifold index
            for idx, p in enumerate(list(self.points)):
                results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                self.sigma.append(results[idx].sigma)
                # print("Sigma", results[idx].sigma)
                mu = (p[i_in], results[idx].mu)
                self.mean.append(mu)

        if manifold == m2_2d_pos_time:
            i_out = [1,2] # Output manifold index
            for idx, p in enumerate(list(self.points)):
                results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])
                self.sigma.append(results[idx].sigma)
                # print("Sigma", results[idx].sigma)
                mu = (p[i_in], results[idx].mu)
                self.mean.append(mu)
            self.mean = M2toR2_2D(self.mean)

    def get_mean(self, demonstration, manifold, point):
        ''' Takes a list of points at a time step, and returns the mean on a specified manifold.
        '''
        # print(demonstration)
        g = rs.Gaussian(manifold).mle(demonstration)
        mu = g.mu
        # print("Point: ", point)
        # print("Mu: ", mu)
        e = manifold.log(point, mu)
        Q = np.eye(3)
        e = e[1:].reshape(3,1)
        # print("Log: ",np.transpose(e).shape, " Point: ", e.shape)
        c = np.matmul(np.matmul(e.T, Q), e)
        # print(c)
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
        # A = np.array([np.zeros(shape=(3,3)), np.zeros(shape=(3,3)), np.zeros(shape=(3,3)),np.zeros(shape=(3,3))]).reshape(6,6)
        # A[:3,3:] = np.identity(n=3,dtype=float)
        # B = np.array([np.zeros(shape=(3,3)), np.identity(n=3,dtype=float)]).reshape(6,3)
        # R = np.identity(n=3,dtype=float) * 0.01
        A = np.array([np.zeros(shape=(dim,dim)), np.zeros(shape=(dim,dim)), np.zeros(shape=(dim,dim)),np.zeros(shape=(dim,dim))]).reshape(2*dim,2*dim)
        A[:dim,dim:] = np.identity(n=dim,dtype=float)
        B = np.array([np.zeros(shape=(dim,dim)), np.identity(n=dim,dtype=float)]).reshape(2*dim,dim)
        R = np.identity(n=dim,dtype=float) * 0.01
        Q = []
        for i in range(len(self.sigma)):
            tmp = np.array(self.sigma[i])
            # print(tmp)
            tmp = tmp[0:dim,0:dim]
            # copy upper half and insert into lower half
            upper = np.triu(tmp)
            tmp = upper + upper.transpose() - np.diag(np.diag(tmp))
            # print(tmp)

            # tmp[abs(tmp)<1e-18] = 0
            # print(tmp)
            tmp = np.linalg.inv(tmp)
            # tmp[tmp>1000] = 1000
            Z = np.zeros((dim,dim),dtype=int) # Create off-diagonal zeros array
            # print(tmp)
            # print("z", Z)
            tmp = np.asarray(np.bmat([[tmp, Z], [Z, tmp]]))
            # print("Q", tmp)
            Q.append(tmp)

        Q = np.array(Q).round(5)
        print(Q[0])

        self.D = []
        self.K = []
        self.u = []
        for i in range(len(Q)):
            # print(Q[i])
            print(Q[i])
            k_temp, S, E = control.lqr(A, B, Q[i], R)
            # self.K.append(k_temp[0:3,0:3] * np.identity(n=3))
            # self.D.append(k_temp[0:3,3:6] * np.identity(n=3))
            self.K.append(k_temp[0:dim,0:dim] * np.identity(n=dim))
            self.D.append(k_temp[0:dim,dim:2*dim] * np.identity(n=dim))
            # print("Mean",self.mean[i])
            # print("Point: ", self.points[i])
            # e = m_time.log(self.points[i], self.mean[i])
            # Q = np.eye(3)
            # e = e[1:].reshape(dim,1)
            # e = e[0:dim]
            # self.u.append(-np.matmul(self.K[i],e))
            # print("U: ", self.u[i])
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
        # print(radius)
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
        
# def R3toM2(data, axis='x'):
#     '''Converts to cylindrical coordinates, axis specifies which axis cylinder is around
#     '''
#     output = []
#     if axis == 'x':
#         for point in data:
#             t = point[0]
#             x = point[-1][0]
#             y = point[-1][1]
#             z = point[-1][2]
#             theta = np.arctan2(z, y)
#             radius = np.sqrt((pow(y,2)+pow(z,2)))
#             vector = np.array([x,radius])
#             new_point = [t, np.array([theta]), vector]
#             output.append(tuple(new_point))
#         return (output)
#     elif axis == 'y':
#         for point in data:
#             t = point[0]
#             x = point[-1][0]
#             y = point[-1][1]
#             z = point[-1][2]
#             theta = np.arctan2(z, x)
#             radius = np.sqrt((pow(x,2)+pow(z,2)))
#             vector = np.array([y,radius])
#             new_point = [t, np.array([theta]), vector]
#             output.append(tuple(new_point))
#         return (output)
#     elif axis == 'z':
#         for point in data:
#             t = point[0]
#             x = point[-1][0]
#             y = point[-1][1]
#             z = point[-1][2]
#             theta = np.arctan2(x, y)
#             radius = np.sqrt((pow(x,2)+pow(y,2)))
#             vector = np.array([z,radius])
#             print(theta)
#             new_point = [t, np.array([theta]), vector]
#             output.append(tuple(new_point))
#         return (output)
#     else:
#         "Print please provide axis x y or z"

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

# def M2toR3(data, axis='x'):
#     """Converts from S1 x R2 to R3, handling different S1 axis orientations.

#     Args:
#         data: (The angle on the S1 circle, (height, radius): The coordinates in the R2 plane.)
#         axis: The axis around which the S1 circle is oriented. 
#               Can be 'x', 'y', or 'z' (default is 'x').
#     Returns:
#         The corresponding (x, y, z) coordinates in R3 space.
#     """
#     output = []
#     norm = 0
#     if axis == 'x':
#         for point in data:
#             t = point[0]
#             theta = point[-1].mean()
#             # print(theta)
#             x = point[-1][0]
#             radius = point[-1][1]
#             # print("point", point, " theta: ", theta, " x: ", x, " radius: ", radius)
#             # radius = np.sqrt(y**2 + z**2)  
#             new_y = radius * np.cos(theta)
#             new_z = radius * np.sin(theta)
#             new_point = [t, np.array([x, new_y, new_z])]
#             output.append(tuple(new_point))
#         return output

#     elif axis == 'y':
#         for point in data:
#             t = point[0]
#             theta = point[-1][0][0]
#             y = point[-1][-1][0]
#             radius = point[-1][-1][1]
#             print("theta: ", theta, " y ", y, " radius: ", radius)
#             # radius = np.sqrt(y**2 + z**2)  
#             new_x = radius * np.cos(theta)
#             new_z = radius * np.sin(theta)
#             new_point = [t, np.array([new_x, y, new_z])]
#             output.append(tuple(new_point))
#         return output

#     elif axis == 'z':
#         for idx, point in enumerate(data):
#             t = point[0]
#             theta = point[-1][0]
#             z = point[-1][0]
#             radius = point[-1][1]
#             # print("theta: ", theta, " z: ", z, " radius: ", radius)
#             # radius = np.sqrt(x**2 + y**2)
#             new_x = radius * np.sin(theta)
#             new_y = radius * np.cos(theta)
#             new_point = [t, np.array([new_x, new_y, z])]
#             if(idx == 0):
#                 norm = np.linalg.norm(new_point[1])
#                 output.append(tuple(new_point))
#             else:
#                 new_norm = np.linalg.norm(new_point[1])
#                 if(new_norm-norm < 0.001):
#                     norm = new_norm
#                     output.append(tuple(new_point))
#         return output

#     else:
#         raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'")
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
    Needs a calinon object where GMR has been done.
    '''
    freq = 500.0

    mass = np.diag([1, 1, 1])
    stiff = np.diag([250, 250, 250])
    damp = np.diag([100, 100, 100])

    robot = rtb.models.UR5()
    adm = Admittance(robot, freq)

    T = np.eye(4)
    # T[:3, :3] = angvec2r(np.linalg.norm(start[3:]), start[3:])
    T[:3, 3] = gmrObject.mean[0][1]
    T[3, :] = [0, 0, 0, 1]

    trajectory = []
    for i in range(len(gmrObject.mean)):
        T[:3, 3] = gmrObject.mean[i][1]
        damp = gmrObject.D[i]
        stiff = gmrObject.K[i]
        wrench = np.zeros(shape=(6,1))
        #wrench[0:3] = gmrObject.u[i]
        rotatedWrench = wrench
        compliant = adm.controller(SE3(T), rotatedWrench, mass=mass, damp=damp, stiff=stiff)
        trajectory.append(compliant[:3, 3])
        # if i == 0:
        #     print(compliant)
    return np.array(trajectory)

def plot_covariance_circle_in_3d(mean, cov, ax, confidence=0.95, num_points=100):
    """
    Plots a circle (or ellipse) representing the uncertainty in 3D data based on mean and covariance.

    Args:
        mean: A NumPy array of shape (3,) representing the mean.
        cov: The covariance matrix (3x3 NumPy array).
        ax: The Matplotlib 3D axis to plot on.
        confidence: The confidence level (e.g., 0.95 for 95% confidence).
        num_points: The number of points to use for drawing the circle/ellipse.
    """

    # Find and sort eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Calculate radii based on confidence level
    chi_sqr_val = np.sqrt(np.percentile(np.random.chisquare(3, num_points), 100 * confidence))
    radii = np.sqrt(eigvals) * chi_sqr_val

    # Generate points on a unit circle in the 2D subspace defined by the first two eigenvectors
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radii[1] * np.cos(theta)
    y = radii[0] * np.sin(theta)
    z = np.zeros(num_points)  # Start in the 2D plane 

    # Transform points to 3D
    points = np.vstack([x, y, z])
    points_3d = np.dot(eigvecs, points) + mean[:, np.newaxis]  

    # Plot the circle/ellipse
    ax.plot(points_3d[0, :], points_3d[1, :], points_3d[2, :])
    ax.scatter(mean[0], mean[1], mean[2], color='red', marker='o')  # Mark the center

def plot_covariance_snake(means, covs, confidence=0.95, num_points=100):
    """
    Plots covariance ellipses/circles resembling a snake around a series of means.

    Args:
        means: A list of NumPy arrays, each of shape (3,) representing the mean points.
        covs: A list of 3x3 NumPy arrays representing the covariance matrices for each point.
        confidence: The confidence level (e.g., 0.95 for 95% confidence).
        num_points: The number of points to use for drawing the circles/ellipses.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array(means)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="blue")  # Plot the line connecting means
    
    for mean, cov in zip(means, covs):
        plot_covariance_circle_in_3d(mean, cov, ax, confidence, num_points)

    plt.show()
   
def main():
    dimensions = '3D' # or '2D'
    stacked = True # Should 3D data be stacked letters
    letters = ['A','S','C','J'] # Letters for testing
    letters = ['J'] # Letters for testing
    filepath = "calinon/experiments_gmr/stacked/" # Folder for experiment 
    for letter in letters:
        if dimensions == '3D':
            dems = pbddata.get_letter_dataS2(letter=letter,n_samples=7,use_time=True)
            data = [point for dem in dems for point in dem]
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
                        rand = 0 #np.random.uniform(-0.1,0.1)
                        new_row = np.append(row, z+rand)  # Add z as a new element
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
            calinon_m1 = Calinon(data)
            calinon_m1.create_gaussians(manifold)
            calinon_m1.lqr_path()

            m2_data = R3toM2(data, axis='x')
            manifold = m2_pos_time
            calinon_m2 = Calinon(m2_data)
            calinon_m2.create_gaussians(manifold, 'x')
            calinon_m2.lqr_path()

            m2_y_data = R3toM2(data, axis='y')
            manifold = m2_pos_time
            calinon_y_m2 = Calinon(m2_y_data)
            calinon_y_m2.create_gaussians(manifold, 'y')
            calinon_y_m2.lqr_path()

            m2_z_data = R3toM2(data, axis='z')
            manifold = m2_pos_time
            calinon_z_m2 = Calinon(m2_z_data)
            calinon_z_m2.create_gaussians(manifold, 'z')
            calinon_z_m2.lqr_path()

            m3_data = R3toM3(data)
            manifold = m3_pos_time
            calinon_m3 = Calinon(m3_data)
            calinon_m3.create_gaussians(manifold)
            calinon_m3.lqr_path()

        if dimensions == '2D':
            manifold = m1_2d_pos_time
            calinon_m1_2d = Calinon(data)
            calinon_m1_2d.create_gaussians(manifold)
            calinon_m1_2d.lqr_path(dim=2)

            m2_data = R2toM2_2D(data)
            manifold = m2_2d_pos_time
            calinon_m2_2d = Calinon(m2_data)
            calinon_m2_2d.create_gaussians(manifold)
            calinon_m2_2d.lqr_path(dim=2)

        # best_manifold = calinon.best_fit()
        # print(best_manifold)

        # -------------------- Admittance control ------------------------
        if dimensions == '3D':
            trajectory_m1 = generateTrajectory(calinon_m1)
            trajectory_m2 = generateTrajectory(calinon_m2)
            trajectory_y_m2 = generateTrajectory(calinon_y_m2)
            trajectory_z_m2 = generateTrajectory(calinon_z_m2)
            trajectory_m3 = generateTrajectory(calinon_m3)

            all_trajectories = [trajectory_m1, trajectory_m2,trajectory_y_m2, trajectory_z_m2, trajectory_m3]
            all_variance = [calinon_m1.sigma, calinon_m2.sigma, calinon_y_m2.sigma, calinon_z_m2.sigma, calinon_m3.sigma]
            manifold_names = ['m1', 'm2x','m2y','m2z','m3']
            for traj, var, manifold in zip(all_trajectories, all_variance, manifold_names):
                sigma_diag = np.array(var).diagonal(axis1=1, axis2=2)
                traj_var = np.concatenate((np.array(traj),sigma_diag),axis=1)
                print(traj_var.shape)
                filename = letter + '_' + manifold + '_mu_cov.csv'
                np.savetxt(filepath+filename, traj_var, delimiter=",")

        if dimensions == '2D':
            trajectory_m1 = np.array([np.concatenate((t, coord)) for t, coord in calinon_m1_2d.mean])[:,1:]
            print(trajectory_m1[:,1:].shape)
            print(calinon_m1_2d.mean[0])
            print(calinon_m2_2d.mean[0])
            trajectory_m2 = np.array([np.concatenate((t, coord)) for t, coord in calinon_m2_2d.mean])[:,1:]
            all_trajectories = [trajectory_m1, trajectory_m2]
            all_variance = [calinon_m1_2d.sigma, calinon_m2_2d.sigma]
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
            plt_data = m_t_s2.swapto_tupleoflist(calinon_m1.mean)
            plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],'.', label='Mean')     # Original Data
            label = 'Gaussian'
            #for r, s in zip(calinon.mean, calinon.sigma):
            #    s2_fcts.plot_gaussian(ax,r[1],s, showtangent=False,
            #                     linealpha=0.3,color='yellow',label=label)
            #    label=''
            # plt.legend(); plt.show()

            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1,projection='3d')
            sigma_diag = np.array(calinon_m1.sigma).diagonal(axis1=1, axis2=2)
            print(np.array(calinon_m1.sigma).shape)
            print(np.concatenate((np.array(trajectory_m1),sigma_diag),axis=1).shape)
            ax.plot(trajectory_m1[:,0],trajectory_m1[:,1],trajectory_m1[:,2], 'blue', linewidth=2)
            ax.plot(trajectory_m2[:,0],trajectory_m2[:,1],trajectory_m2[:,2], 'black', linewidth=2)
            ax.plot(trajectory_y_m2[:,0],trajectory_y_m2[:,1],trajectory_y_m2[:,2], 'orange', linewidth=2)
            ax.plot(trajectory_z_m2[:,0],trajectory_z_m2[:,1],trajectory_z_m2[:,2], 'brown', linewidth=2)
            ax.plot(trajectory_m3[:,0],trajectory_m3[:,1],trajectory_m3[:,2], 'green', linewidth=2)
            # plot_covariance_snake(trajectory_m1, calinon_m1.sigma)
            # for i in range(0, len(calinon_m1.mean), 20):
            #     # print("Mean: ", mean[1])
            #     # print("Sigma: ", sigma)
            #     plot_covariance_circle_in_3d(calinon_m1.mean[i][1], calinon_m1.sigma[i], ax)
            demo = np.array(list(map(lambda x: x[1], data)))
            filename = letter + '_demonstrations.csv'
            np.savetxt(filepath+filename, np.array(demo), delimiter=",")
            ax.scatter3D(demo[:,0],demo[:,1],demo[:,2], 'blue')
            ax.legend(['M1','M2_x', 'M2_y','M2_z','M3', 'Demonstration', 'Demonstrstaion after m2'])
            plt.show()

        if dimensions == '2D':
            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(trajectory_m1[:,0],trajectory_m1[:,1], 'blue', linewidth=2)
            ax.plot(trajectory_m2[:,0],trajectory_m2[:,1], 'green', linewidth=2)
            # plot_covariance_snake(trajectory_m1, calinon_m1.sigma)
            # for i in range(0, len(calinon_m1.mean), 20):
            #     # print("Mean: ", mean[1])
            #     # print("Sigma: ", sigma)
            #     plot_covariance_circle_in_3d(calinon_m1.mean[i][1], calinon_m1.sigma[i], ax)
            demo = np.array(list(map(lambda x: x[1], data)))
            filename = letter + '_demonstrations.csv'
            np.savetxt(filepath+filename, np.array(demo), delimiter=",")
            ax.scatter(demo[:,0],demo[:,1], c='red')
            ax.legend(['M1','M2_x', 'M2_y','M2_z','M3', 'Demonstration', 'Demonstrstaion after m2'])
            plt.show()

        # plt.plot(trajectory)
        # plt.legend(['x','y','z'])
        # plt.show()
        # start = trajectory[0]
        # start = SE3.Trans(start)
        # #print(robot.ik_LM(start))
        # robot = rtb.models.UR5()
        # robot.q = robot.ik_LM(start)[0]#robot.ik_LM(start)

        # Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
        # sol = robot.ik_LM(SE3.Trans(trajectory[-1]))         # solve IK
        # #print(sol)
        # q_pickup = sol[0]
        # qt = rtb.jtraj(robot.qr, q_pickup, 50)
        # robot.plot(qt.q, backend='pyplot')




if __name__ == "__main__":
    main()