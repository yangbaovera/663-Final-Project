
import numpy as np
from scipy.special import digamma, polygamma

N = np.array([ 9,  9, 10, 12,  6])
M = 5
k = 4
V = 32
alpha = np.random.dirichlet(10*np.ones(k),1)[0]
beta = np.random.dirichlet(np.ones(V),k)
gamma = np.tile(alpha,(M,1)) + np.tile(N/k,(k,1)).T

def test_converge2():
    tol = 10**(-2)
    
    loss1 = np.sqrt(list(map(np.sum,np.square(beta - beta))))
    loss2 = np.sqrt(list(map(np.sum,np.square(gamma - gamma))))
    assert np.max(loss1) <= tol and np.max(loss2) <= tol