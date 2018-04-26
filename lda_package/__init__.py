
import numpy as np
from scipy.special import digamma, polygamma


##################################### Algorithm Implementation########################################33

# convergence function
def is_convergence1(old, new, tol = 10**(-2)):
    """
    output:
    TRUR or FALSE
    """
    loss = np.sqrt(list(map(np.sum,np.square(old - new))))
    return np.max(loss) <= tol

def is_convergence2(old, new, tol = 10**(-2)):
    """
    output:
    TRUR or FALSE
    """
    loss = np.sqrt(np.sum(np.square(old - new)))
    return np.max(loss) <= tol

def optimize_vp(phi, gamma, alpha, beta, words, M, N, K, max_iter=500):
    '''
    optimize the variational parameter
    
    Parameters
    ----------
    phi:   ndarray
           An array of topic-word matrix
    gamma: ndarray
           A matrix of doc-topic
    alpha: ndarray
           the parameter of doc-topic dirichlet distribution
    beta:  ndarray
           the parameter of topic-word dirichlet distribution
    words: list 
           the list of lists of words in all 
    M : int, the number of documents
    N : ndarraay, the number of words in each document
    K : int, the number of topics in the corpus
    Returns
    -------
    out : list of ndarray
          the optimized and normalized(sum to 1) phi 
    '''
    
    for t in range(max_iter):
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
        
        if is_convergence1(phi_old, phi_new) == True and is_convergence2(gamma_old, gamma_new) == True:
            break
   
    return phi, gamma

# estimate alpha
def alpha_estimate(gamma, alpha_initial, K, M, max_iter = 100):
    """
    This is an estimation function, especially used in the process of LDA algorithm.
    digamma function and polygamma function are used in the following process.
    
    input:
    alpha_initial: the initial setting of alpha, it is an 1*K vector
    K: the number of topics
    M: the number of documents
    gamma: the result from another update function (see gamma_update())
    """
    
    alpha = alpha_initial
    for t in range(max_iter):
        alpha_old = alpha
        
        # compute the gradient vector and the diagonal part of the Hessian matrix
        g = np.zeros(K)
        h = np.zeros(K)
        for i in range(K):
            g1 = M*(digamma(np.sum(alpha))-digamma(alpha[i]))
            g2 = 0
            for d in range(M):
                g2 += digamma(gamma[d,i])-digamma(np.sum(gamma[d,:]))
            g[i] = g1 + g2
            
            h[i] = -M*polygamma(1, alpha[i])
        
        # compute the constant part
        z = M*polygamma(1, np.sum(alpha))
        c = (np.sum(g/h))/(z**(-1) + np.sum(h**(-1)))
                           
        # update alpha                   
        alpha -= (g-c)/h
        
        if is_convergence2(alpha_old, alpha):
            break
            
    return alpha

# estimate beta
def beta_estimate(K, V_words, phi, D):
    
    """
    This is an estimation function, especially used in the process of LDA algorithm
    
    input:
    K: the number of topics
    V_words: a vector of all unique words in the vocabulary
    D: D = (w_1,w_2,...w_M), contains all words in all documents
    phi: the result from another update function (see phi_update())
    
    output:
    beta: the estimate parameter for LDA, it is a K*V matrix
    """
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
        
    return beta

# Variation EM
def variation_EM(M, K, D, N, V_words, alpha_initial, beta_initial, gamma_initial, phi_initial, iteration = 1000):
    
    phi_gamma = optimize_vp(phi_initial, gamma_initial, alpha_initial, beta_initial, D, M, N, K)
    phi = phi_gamma[0]
    gamma = phi_gamma[1]
    
     
    (alpha, beta) = (alpha_initial, beta_initial)
    
    for t in range(iteration):
        
        (phi_old, gamma_old) = (phi, gamma)
        
        alpha = alpha_estimate(gamma, alpha, K, M)
        beta = beta_estimate_opt(K, V_words, phi, D)
        
        phi_gamma1 = optimize_vp(phi, gamma, alpha, beta, D, M, N, K)
        phi = phi_gamma1[0]
        gamma = phi_gamma1[1]
        
        if is_convergence2(gamma_old, gamma) and is_convergence1(phi_old, phi):
            break
    
    return alpha, beta, gamma, phi



# a new function to calculate log of sum
def log_sum(log_a, log_b):
    """
    input: log(a), log(b)
    output: log(a+b)
    """
    return log_a + np.log(1+np.exp(log_b - log_a))



def optimize_vp_opt(phi, gamma, alpha, beta, words, M, N, K, max_iter=500):
    '''
    optimize the variational parameter
    
    Parameters
    ----------
    phi:   ndarray
           An array of topic-word matrix
    gamma: ndarray
           A matrix of doc-topic
    alpha: ndarray
           the parameter of doc-topic dirichlet distribution
    beta:  ndarray
           the parameter of topic-word dirichlet distribution
    words: list 
           the list of lists of words in all 
    M : int, the number of documents
    N : ndarraay, the number of words in each document
    K : int, the number of topics in the corpus
    Returns
    -------
    out : list of ndarray
          the optimized and normalized(sum to 1) phi 
    '''
    
    for t in range(max_iter):
        phi_old = phi
        
        # we use log(phi) here and following processes
        log_phi = np.array(list(map(np.log, phi)))
        gamma_old = gamma
       
        for m in range(M):
            for n in range(N[m]):
                
                logsum = 0
                for i in range(K):
                    
                    # use new method in log form to update phi
                    log_phi[m][n,i] = np.log(beta[i,np.int(words[m][n])]) + digamma(gamma[m,i])
                    
                    logsum = log_sum(logsum, log_phi[m][n,i])
                # use new metohd to implement nomalization
                log_phi_mn = log_phi[m][n,:] - logsum
                log_phi[m][n,:] = log_phi_mn
                
                phi[m][n,:] = np.exp(log_phi_mn)
        
            # instead of alpha, use old phi and new phi to iterative
            d_phi = phi[m] - phi_old[m]
            gamma[m,:]  = gamma[m,:] + np.sum(d_phi, axis = 0)
            
        phi_new = phi
        gamma_new = gamma
        
        if is_convergence1(phi_old, phi_new) == True and is_convergence2(gamma_old, gamma_new) == True:
            break
   
    return phi, gamma


# estimate alpha
def alpha_estimate_opt(gamma, alpha_initial, K, M, max_iter = 100):
    """
    This is an estimation function, especially used in the process of LDA algorithm.
    digamma function and polygamma function are used in the following process.
    
    input:
    alpha_initial: the initial setting of alpha, it is an 1*K vector
    K: the number of topics
    M: the number of documents
    gamma: the result from another update function (see gamma_update())
    """
    from scipy.special import digamma, polygamma
    
    alpha = alpha_initial
    for t in range(max_iter):
        alpha_old = alpha
        
        # we use vector instead of calculating in loop
        g = M*(digamma(np.sum(alpha))-digamma(alpha)) 
        + np.sum(digamma(gamma) -np.tile(digamma(np.sum(gamma,axis=1)),(K,1)).T,axis=0)
        h = -M*polygamma(1,alpha)
        
        z = M*polygamma(1, np.sum(alpha))
        c = (np.sum(g/h))/(z**(-1) + np.sum(h**(-1)))
                           
        # update alpha                   
        alpha -= (g-c)/h
        
        if is_convergence2(alpha_old, alpha):
            break
            
    return alpha


# estimate beta
def beta_estimate_opt(K, V_words, phi, D):
    
    """
    This is an estimation function, especially used in the process of LDA algorithm
    
    input:
    K: the number of topics
    V_words: a vector of all unique words in the vocabulary
    D: D = (w_1,w_2,...w_M), contains all words in all documents
    phi: the result from another update function (see phi_update())
    
    output:
    beta: the estimate parameter for LDA, it is a K*V matrix
    """
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
    
    # then normalize each row s.t. the row sum is one, in vector method
    beta= beta/ np.sum(beta, axis = 1).reshape((-1,1))
        
    return beta


def variation_EM_new(M, K, D, N, V_words, alpha_initial, beta_initial, gamma_initial, phi_initial, iteration = 1000):
    
    phi_gamma = optimize_vp_opt(phi_initial, gamma_initial, alpha_initial, beta_initial, D, M, N, K)
    phi = phi_gamma[0]
    gamma = phi_gamma[1]
    
     
    (alpha, beta) = (alpha_initial, beta_initial)
    
    for t in range(iteration):
        
        (phi_old, gamma_old) = (phi, gamma)
        
        alpha = alpha_estimate_opt(gamma, alpha, K, M)
        beta = beta_estimate_opt(K, V_words, phi, D)
        
        phi_gamma1 = optimize_vp_opt(phi, gamma, alpha, beta, D, M, N, K)
        phi = phi_gamma1[0]
        gamma = phi_gamma1[1]
        
        if is_convergence2(gamma_old, gamma) and is_convergence1(phi_old, phi):
            break
    
    return alpha, beta, gamma, phi



############################################### Data Process #######################################################

def data_clean(doc_set, stop_word = None):
    
    """
    REF: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
    
    input: 
    doc_set: a list of documets, the elements are context of that document
    stop_word: if 'stop_word = None', this function will give a sample
    
    output: 
    texts:a list of array(documents), each element contains all words in that document
    dictionary: a dictionary, key is the id of words, values are unique words
    corpus: a list of list, each inner list represents a document. In the inner list, each tuple is (word_id, word_count)
    
    """
    
    from nltk.tokenize import RegexpTokenizer
    from stop_words import get_stop_words
    from nltk.stem.porter import PorterStemmer
    from gensim import corpora, models
    import gensim
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    # create English stop words list
    if stop_word==None:
        stop_word = get_stop_words('en')
        
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stop_word]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
        # add tokens to list
        texts.append(stemmed_tokens)

        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
    
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
    
    return texts, dictionary, corpus

def data_process(texts, dictionary):

    """
    transform the texts from word-formation into id-formation, which can be used in function variantion_EM()
    
    """
    import numpy as np
    
    text_ = []
    for i in range(len(texts)):
        text_i = []
        for l in range(len(texts[i])):
            text_i = np.append(text_i, dictionary.token2id[texts[i][l]])
        text_.append(text_i)
            
    return text_



