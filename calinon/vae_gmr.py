import matplotlib.pyplot as plt
import numpy as np
import hyperspherical_vae.distributions.von_mises_fisher as vmf
import torch
import torch.distributions as td
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import glob
import pickle
from GeodesicMotionSkills.Experiments.Utils import auxiliary_tests, discretized_manifold
from GeodesicMotionSkills.Experiments.Utils.environment import Environment
from stochman.manifold import EmbeddedManifold
from stochman.curves import DiscreteCurve
from stochman import nnj
import copy

from enum import Enum

class RegularizationType(Enum):
    NONE      = None
    SHRINKAGE = 1
    DIAGONAL  = 2

from vae import *




class VAEgmr():
    def __init__(self, model, demonstrations, n_components, base=None):
        """
        Takes a VAE model of the manifold as input, as well as the demonstrations
        used for training VAE
        """
        mu0 = np.zeros(2)
        sigma0 = np.eye(3)

        self.gaussians = []


        self.model = model
        self.demo = demonstrations
        self.n_components = n_components
        self.priors = np.ones(n_components)/n_components
        

    def log(self, p0, p1, discrete_model=None):
        ''' Computes the logarithmic map on a Discretized Manifold.
            Model : an instance of the class Toy_example.VAE()
            p0 and p1 : numpy array
        '''
        self.model.eval()
        # Encode points to manifold
        print("p0", p0, "p1", p1)    
        p0 = self.model.encode(torch.tensor(p0).float(), train_rbf=True)[1]
        p1 = self.model.encode(torch.tensor(p1).float(), train_rbf=True)[1]
        points = torch.cat((p0.view(1,-1),p1.view(1,-1)), 0)

        curve, dist = discrete_model.shortest_path(p0, p1)
        curve = discrete_model.connecting_geodesic(p0.view(1,-1), p1.view(1,-1), self.model, self.model.time_step) # Cubic spline
        derivative_t0 = self.model.decode(curve.deriv(torch.zeros(1)), train_rbf=True)[0]
        derivative_t0_var = derivative_t0.stddev.detach().numpy()
        derivative_t0 = derivative_t0.mean.detach().numpy()
        mu,cov = np.array_split(self.model.embed(p0.view(1,-1), False).detach().numpy()[0],2)
        cov = np.diag(cov)
        # cov = self.model.embed(p0.view(1,-1), False).detach().numpy()
        print("Spline derive variance at t=0:\n", cov)#, derivative_t0_var)
        derivative_t = self.model.decode(curve.deriv(torch.ones(1)*2), train_rbf=True)[0].mean.detach().numpy()
        print("\nSpline derive at t=1: ", derivative_t)
        log = (derivative_t0/np.linalg.norm(derivative_t0))*dist
        print("Log",log)
        return log
        # Derivative of curve

    def exp(self, x, g): # Only needed for the GMR step
        '''
        Uses stochman method for the Exponential map
        '''
        pass

    def expectation(self, data):
        lik = []
        for i, guass in enumerate(self.gaussians):
            lik.append(gauss.prob(data)*self.priors[i])
        lik = np.vstack(lik)
        return lik
        pass

    def predict():
        pass

    def fit():
        pass

    def kmeans():
        pass

    # Only needed for the GMR step

    def parallel_transport():
        pass

    def margin():
        pass

    def gmr():
        pass



class Gaussian():
    def __init__(self, model, mu=None, sigma=None):
        self.model = model

        if mu is None:
            self.mu = np.zeros(2)
        else:
            self.mu = mu
        
        if sigma is None:
            self.sigma = np.eye(2)
        else:
            self.sigma = sigma

    def log(self, x, g):
        pass

    def prob(self, data):
        d = 2
        reg = np.sqrt( ( (2*np.pi)**d )*np.linalg.det(self.sigma) ) + 1e-200

        dist = self.log(data, self.mu)
        
        # Correct size:
        if dist.ndim==2:
            # Correct dimensions
            pass
        elif dist.ndim==1 and dist.shape[0]==self.manifold.n_dimT:
            # Single element
            dist=dist[None,:]
        elif dist.ndim==1 and self.manifold.n_dimT==1:
            # Multiple elements
            dist=dist[:,None]

        dist = ( dist * np.linalg.solve(self.sigma,dist.T).T ).sum(axis=(dist.ndim-1))
        probs =  np.exp( -0.5*dist )/reg 

        return probs
    
    def margin(self):
    
    def mle(self, x, h=None, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):
        self.mu = self.__empirical_mean(x,h)
        self.sigma = self.__empirical_covariance(x, h, reg_lambda, reg_type)
        return self
    
    def __empirical_mean(self, x, h=None):
        mu = self.mu
        diff = 1.0
        it = 0;
        while (diff > 1e-8):
            delta = self.__get_weighted_distance(x, mu, h)
            mu = self.exp(delta, mu)
            diff = sum(delta*delta)
            it+=1
            if it > 50:
                raise RuntimeWarning('Gaussian mle not converged in 50 iterations.')
                break
        return mu

    def __get_weighted_distance(self, x, base, h=None):
        if h is None:
            # No weights given, equal weight for all points
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray :
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data)/n_data

        dtmp = self.log(x, base)
        d = h.dot(self.log(x, base))
        return d
    
    def __empirical_covariance(self, x, h=None, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):
        if h is None:
            # No weights given, equal weight for all points
            # Determine dimension of input
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray :
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data)/n_data

        tmp = self.log(x, self.mu)
        sigma = tmp.T.dot(np.diag(h).dot(tmp))

        # Perform Shrinkage regularizaton:
        if (reg_type == RegularizationType.SHRINKAGE):
            return reg_lambda*np.diag(np.diag(sigma)) + (1-reg_lambda)*sigma
        elif (reg_type == RegularizationType.DIAGONAL):
            return sigma + reg_lambda*np.eye(len(sigma))
        elif reg_type==None:
            return sigma
        else:
            raise ValueError('Unknown regularization type for covariance regularization')

    def condition(self, val):


