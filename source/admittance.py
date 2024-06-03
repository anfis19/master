import numpy as np
from spatialmath import *
from spatialmath.base import q2r, r2q

class Admittance:
    startq = np.zeros(shape=6)  # Initial joint configuration
    p_c = np.zeros(shape=3)  # Set compliant position
    q_c = UnitQuaternion(1, [0, 0, 0])

    # For translation
    delta_p = np.zeros(shape=(3, 1))
    delta_dp = np.zeros(shape=(3, 1))
    delta_ddp = np.zeros(shape=(3, 1))

    # For rotation
    delta_q = UnitQuaternion([1, 0, 0, 0])
    delta_w = np.zeros(shape=(3, 1))
    delta_dw = np.zeros(shape=(3, 1))

    def __init__(self, robot, f):
        self.robot = robot
        self.ts = 1 / f

    def setStartQ(self, q):
        self.startq = q

    def controller(self, pose_d: SE3, wrench, mass, stiff, damp):
        p_d = pose_d.t.reshape(3, 1)
        q_d = UnitQuaternion(r2q(pose_d.R))

        F = np.array(wrench[:3]).reshape(3, 1)
        mu = np.array(wrench[3:]).reshape(3, 1)

        # Gain matrices
        M = mass  # np.diag([1,1,1])
        invM = np.linalg.inv(M)  # We need this for the control loop
        K_p = stiff  # np.diag([25,25,25])
        D_p = damp  # np.diag([10,10,10])

        prev_ddp = self.delta_ddp
        prev_dp = self.delta_dp
        prev_w = self.delta_w
        prev_dw = self.delta_dw
        # print((F-np.matmul(D_p, self.delta_dp)-np.matmul(K_p, self.delta_p)))
        self.delta_ddp = np.matmul(invM, (F - np.matmul(D_p, self.delta_dp) - np.matmul(K_p, self.delta_p)))
        self.delta_dw = np.matmul(invM, (mu - np.matmul(D_p, self.delta_w) - np.matmul(self.K_prime(self.delta_q, K_p),
                                                                                       self.delta_q.v.reshape((3, 1)))))

        self.delta_dp = np.multiply((self.delta_ddp + prev_ddp) / 2, self.ts) + prev_dp
        # print(self.delta_dp)
        # print(np.multiply((self.delta_dp+prev_dp)/2, self.ts))
        self.delta_p = np.multiply((self.delta_dp + prev_dp) / 2, self.ts) + self.delta_p

        self.delta_w = np.multiply((self.delta_dw + prev_dw) / 2, self.ts) + prev_w
        self.delta_q = self.expMap(self.ts, self.delta_w) * self.delta_q

        self.p_c = self.delta_p + p_d
        self.q_c = (self.delta_q * q_d)

        T = np.empty((4, 4))
        T[:3, :3] = q2r(self.q_c.vec)
        # print(self.p_c)
        T[:3, 3] = self.p_c.reshape(3, )
        T[3, :] = [0, 0, 0, 1]
        return T

    def skew(self, vector):
        """
        this function returns a numpy array with the skew symmetric cross product matrix for vector.
        the skew symmetric cross product matrix is defined such that
        np.cross(a, b) = np.dot(skew(a), b)

        :param vector: An array like vector to create the skew symmetric cross product matrix for
        :return: A numpy array of the skew symmetric cross product vector
        """

        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    def expMap(self, ts, omega):
        r = (ts / 2) * omega
        if np.linalg.norm(r) == 0:
            epsilon = np.zeros(3)
        else:
            epsilon = np.multiply(np.divide(r, np.linalg.norm(r)), np.sin(np.linalg.norm(r)))
        epsilon = epsilon.reshape(3)
        q = UnitQuaternion([np.cos(np.linalg.norm(r)), epsilon[0], epsilon[1], epsilon[2]])
        return q

    def K_prime(self, q: UnitQuaternion, Ko):
        I = q.s * np.identity(3)
        E = I - self.skew(q.v)
        return 2 * np.matmul(E.transpose(), Ko)
