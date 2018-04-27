
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


def test_beta():
    K = k
    D = text_
    V = len(V_words)
    beta = np.ones((K,V))
    # first obtain the propotion values
    for j in range(V):
        word = V_words[j]
        # give a TRUE or FALSE "matrix", remember w_mnj should have the same shape with phi
        w_mnj = [np.repeat(w==word, K).reshape((len(w),K)) for w in D]
        # compute the inner sum over number of words
        sum1 = list(map(lambda x: np.sum(x,axis=0),phi*w_mnj))
        # compute the outer sum over documents
        beta[:,j] = np.sum(np.array(sum1), axis = 0)
    
    # then normalize each row s.t. the row sum is one
    for i in range(K):
        beta[i,:] = beta[i,:]/sum(beta[i,:])
        
    assert np.min(beta)>0 
    assert all(np.sum(beta, axis = 1)==1)
