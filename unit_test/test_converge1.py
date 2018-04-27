
import numpy as np
from scipy.special import digamma, polygamma

N = np.array([ 9,  9, 10, 12,  6])
M = 5
k = 4

phi = np.array([1/k*np.ones([N[m],k]) for m in range(M)])

def test_converge1():
    tol = 10**(-2)
    
    loss = np.sqrt(list(map(np.sum,np.square(phi - phi))))
    assert np.max(loss) <= tol