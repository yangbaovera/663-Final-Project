
import numpy as np
from scipy.special import digamma, polygamma

text_ = [np.array([0., 3., 2., 1., 4., 2., 3., 0., 5.]),
         np.array([ 5., 11.,  9., 12.,  8.,  1.,  6.,  7., 10.]),
         np.array([16., 15., 20.,  8., 18., 14., 17., 21., 13., 19.]),
         np.array([25., 23., 19., 26., 29., 27.,  5., 24., 28.,  8.,  1., 22.]),
         np.array([16., 30., 31.,  0.,  3., 16.])]
np.random.seed(64528)
M = 5
k = 4
N = np.array(list(map(len, text_)))
V = 32
V_words = range(V)
alpha = np.random.dirichlet(10*np.ones(k),1)[0]
beta = np.random.dirichlet(np.ones(V),k)

phi = np.array([1/k*np.ones([N[m],k]) for m in range(M)])
gamma = np.tile(alpha,(M,1)) + np.tile(N/k,(k,1)).T


def test_optimize_vp():
    K = k
    words = text_
    
    for t in range(10):
        phi_old = phi
        gamma_old = gamma
        #update phi
        for m in range(M):
            for n in range(N[m]):
                for i in range(K):
                    phi[m][n,i] = beta[i,np.int(words[m][n])] * np.exp(digamma(gamma[m,i]))
                #nomalize to 1)
                phi[m][n,:] = phi[m][n,:]/np.sum(phi[m][n,:])
        phi_new = phi
        #update gamma
        for i in range(M):
            gamma[i,:]  = alpha + np.sum(phi[i], axis = 0)
        gamma_new = gamma

        assert np.min(list(map(np.min, phi)))>0
        assert all(np.sum(phi[1], axis = 1))==1
        assert np.min(gamma)>0
