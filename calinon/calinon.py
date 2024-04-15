import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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
m_s2 = rm.get_s2_manifold()

m_s3  = rm.get_quaternion_manifold()
m_e1xe3xq  = m_e1*m_e3*m_s3

m1_pos = m_e3
m1_ori = m_s3

m3_pos = m_s2 * m_e1
m_time = m_e1 * m_e3
m_t_s2 = m_e1 * m_s3

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


    def create_gaussians(self, manifold):
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

        g = rs.Gaussian(manifold).mle(self.points)
        results = []
        i_in  = 0 # Input manifold index
        i_out = 1 # Output manifold index
        x_in = self.points
        # print(x_in)
        results = g.condition(x_in,i_in=i_in,i_out=i_out)
        for idx, r in enumerate(results):
            self.sigma.append(r.sigma)
            mu = (x_in[idx][0], r.mu)
            self.mean.append(mu)

    



    def get_mean(self, demonstration, manifold, point):
        ''' Takes a list of points at a time step, and returns the mean on a specified manifold.
        '''
        # print(demonstration)
        g = rs.Gaussian(manifold).mle(demonstration)
        mu = g.mu
        print("Point: ", point)
        print("Mu: ", mu)
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

    def lqr_path(self):
        '''Computes LQR at each timestep on manifold
        '''
        # A = np.array([np.zeros(shape=(3,3)), np.zeros(shape=(3,3)), np.zeros(shape=(3,3)),np.zeros(shape=(3,3))]).reshape(6,6)
        # A[:3,3:] = np.identity(n=3,dtype=float)
        # B = np.array([np.zeros(shape=(3,3)), np.identity(n=3,dtype=float)]).reshape(6,3)
        # R = np.identity(n=3,dtype=float) * 0.01
        A = np.array([np.zeros(shape=(2,2)), np.zeros(shape=(2,2)), np.zeros(shape=(2,2)),np.zeros(shape=(2,2))]).reshape(4,4)
        A[:2,2:] = np.identity(n=2,dtype=float)
        B = np.array([np.zeros(shape=(2,2)), np.identity(n=2,dtype=float)]).reshape(4,2)
        R = np.identity(n=2,dtype=float) * 0.01
        Q = []
        for i in range(len(self.sigma)):
            tmp = np.array(self.sigma[i])
            tmp = tmp[0:2,0:2]
            #copy upper half and insert into lower half
            # upper = np.triu(tmp)
            # tmp = upper + upper.transpose() - np.diag(np.diag(tmp))
            # print(tmp)
            tmp = np.linalg.inv(tmp)
            Z = np.zeros((2,2),dtype=int) # Create off-diagonal zeros array
            tmp = np.asarray(np.bmat([[tmp, Z], [Z, tmp]]))
            Q.append(tmp)

        Q = np.array(Q).round(5)


        self.D = []
        self.K = []
        self.u = []
        for i in range(len(Q)):
            k_temp, S, E = control.lqr(A, B, Q[i], R)
            # self.K.append(k_temp[0:3,0:3] * np.identity(n=3))
            # self.D.append(k_temp[0:3,3:6] * np.identity(n=3))
            self.K.append(k_temp[0:2,0:2] * np.identity(n=2))
            self.D.append(k_temp[0:2,2:4] * np.identity(n=2))
            # print("Mean",self.mean[i])
            # print("Point: ", self.points[i])
            e = m_time.log(self.points[i], self.mean[i])
            # Q = np.eye(3)
            e = e[1:].reshape(3,1)
            e = e[0:2]
            self.u.append(-np.matmul(self.K[i],e))
        np.linalg.cholesky(self.K)
        np.linalg.cholesky(self.D)
        print(self.u)



        

        
        
   
def main():
    startpts = np.array([0, 0, 0])
    endpts = np.array([12, 3, 8])
    dems = pbddata.get_letter_dataS2(letter='S',n_samples=4,use_time=True)
    data = [point for dem in dems for point in dem]
    data = sorted(data, key=lambda x: x[0][0])
    print(data[0])
    line = np.vstack(pbddata.get_letter_dataS2(letter='S',n_samples=4,use_time=False))
    time = np.linspace(0,1,800)
    # print("Line shape: ", line.shape)
    line = (line, np.flip(line, 0), time.reshape(800,1))
    # print("Data", sorted(data, key=lambda x: x[0][0])[:10])

    print("S2 Time: ", m_s2.id_elem, " NdimT: ", m_t_s2.n_dimT, " NdimM; ", m_t_s2.n_dimM)
    print("M time: ", m_e3.id_elem, " NdimT: ", m_time.n_dimT, " NdimM; ", m_time.n_dimM)

    calinon = Calinon(data)
    calinon.create_gaussians(m_time)
    # print(calinon.get_mean(points, m_time,data[50]))
    calinon.lqr_path()
    # best_manifold = calinon.best_fit()
    # print(best_manifold)

    # -------------------- Admittance control ------------------------
    freq = 500.0

    mass = np.diag([1, 1, 1])
    stiff = np.diag([250, 250, 250])
    damp = np.diag([100, 100, 100])

    robot = rtb.models.UR5()
    adm = Admittance(robot, freq)

    T = np.eye(4)
    # T[:3, :3] = angvec2r(np.linalg.norm(start[3:]), start[3:])
    T[:3, 3] = calinon.mean[0][1]
    T[3, :] = [0, 0, 0, 1]

    trajectory = []

    for i in range(len(calinon.mean)):
        T[:3, 3] = calinon.mean[i][1]
        damp = calinon.D[i]
        stiff = calinon.K[i]
        wrench = np.zeros(shape=(6,1))
        wrench[0:3] = calinon.u[i]
        rotatedWrench = wrench
        compliant = adm.controller(SE3(T), rotatedWrench, mass=mass, damp=damp, stiff=stiff)
        trajectory.append(compliant[:3, 3])
        # if i == 0:
        #     print(compliant)
    trajectory = np.array(trajectory)

    fig = plt.figure()
 
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    print("Data", np.array(list(map(lambda x: x[1], data))))
    ax.plot3D(trajectory[:,0],trajectory[:,1],trajectory[:,2], 'red')
    # trajectory = np.array(list(map(lambda x: x[1], data)))
    # ax.plot3D(trajectory[:,0],trajectory[:,1],trajectory[:,2], 'green')
    plt.show()
    # return 0
    plt.plot(trajectory)
    plt.legend(['x','y','z'])
    plt.show()
    start = trajectory[0]
    start = SE3.Trans(start)
    print(robot.ik_LM(start))
    robot.q = robot.ik_LM(start)[0]#robot.ik_LM(start)

    Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ik_LM(SE3.Trans(trajectory[-1]))         # solve IK
    print(sol)
    q_pickup = sol[0]
    qt = rtb.jtraj(robot.qr, q_pickup, 50)
    robot.plot(qt.q, backend='pyplot')
    dt = 0.05
    qt_test = []
    env = swift.Swift()
    # env.launch(realtime=True)
    arrived = False
    # for i in range(len(trajectory)-1):
    #     arrived = False
    # while not arrived:
    #     current_pos = robot.fkine(robot.q)
    #     v, arrived = rtb.p_servo(current_pos, SE3.Trans(trajectory[-1]), 1)
    #     print(np.linalg.pinv(robot.jacobe(robot.q)) @ v)
    #     robot.qd = np.linalg.pinv(robot.jacobe(robot.q)) @ v
    # env.step(dt)
        # qt_test.append(robot.q)
    # print(robot.qd)
    # print(np.array(qt_test))
    # robot.plot(np.array(qt_test), backend='pyplot')



if __name__ == "__main__":
    main()