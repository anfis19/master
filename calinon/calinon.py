import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/anton/master')

# Riepybdlib stuff:
from riepybdlib import statistics as rs
from riepybdlib import manifold as rm
import riepybdlib.data as pbddata  # Some data to showcase the toolbox
import riepybdlib.plot as pbdplt   # Plot functionality (relies on matplotlib)
import riepybdlib.s2_fcts as s2_fcts

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


class Calinon:
    def __init__(self, points):
        ''' Initialisation of the class
            Takes the data as input.
        '''
        self.points = points

        gmm_m1 = rs.GMM(m1_pos, 6)
        print(m1_pos.np_to_manifold(points))
        gmm_m1.kmeans(points)
        lik,avglik = gmm_m1.fit(points, reg_lambda=1e-2, maxsteps = 500)
        gmm_m3 = rs.GMM(m3_pos, 6)
        gmm_m3.kmeans(points)
        lik,avglik = gmm_m3.fit(points, reg_lambda=1e-2, maxsteps = 500)

    def create_gaussians(self):
        ''' Create a gaussian distribution for the given manifold
        '''
        g = rs.Gaussian(m3_pos).mle(self.points)
        g2 = rs.Gaussian(m1_pos).mle(self.points)
        self.gaussians = [g.sigma, g2.sigma]

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
        
   
def main():
    startpts = np.array([0, 0, 0])
    endpts = np.array([12, 3, 8])

    line = np.vstack(pbddata.get_letter_dataS2(letter='I',n_samples=4,use_time=False))
    new = np.linspace(0,1,800)
    line = (line, new.reshape(800,1))

    calinon = Calinon(line)
    calinon.create_gaussians()
    best_manifold = calinon.best_fit()
    print(best_manifold)

if __name__ == "__main__":
    main()