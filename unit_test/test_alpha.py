
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
alpha = np.random.dirichlet(10*np.ones(k),1)[0]
phi = np.array([1/k*np.ones([N[m],k]) for m in range(M)])
gamma = np.tile(alpha,(M,1)) + np.tile(N/k,(k,1)).T

def test_alpha():
    K = k
    alpha = np.random.dirichlet(10*np.ones(k),1)[0]
    for t in range(10):
        alpha_old = alpha
        
        g = np.zeros(K)
        h = np.zeros(K)
        for i in range(K):
            g1 = M*(digamma(np.sum(alpha))-digamma(alpha[i]))
            g2 = 0
            for d in range(M):
                g2 += digamma(gamma[d,i])-digamma(np.sum(gamma[d,:]))
            g[i] = g1 + g2
            
            h[i] = -M*polygamma(1, alpha[i])
        
        z = M*polygamma(1, np.sum(alpha))
        c = (np.sum(g/h))/(z**(-1) + np.sum(h**(-1)))
                   
        alpha -= (g-c)/h
            
        assert np.min(alpha)>0
