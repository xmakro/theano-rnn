import numpy as np
import theano
import theano.tensor as T
import scipy

def Tparam(init):
    param = theano.shared(np.array(init, dtype=theano.config.floatX))
    return param

def zeros(shape):
    return Tparam(np.zeros(shape))

def uniform(shape, range=0.01):
    return Tparam(np.random.uniform(-range, +range, shape))
    
def normal(shape, scale=0.01):
    return Tparam(np.random.normal(0, scale, shape))
    
def sample_weights(shape, sparsity=-1, scale=0.01):
    """
    Initialization that fixes the largest singular value.
    """
    sizeX = int(shape[0])
    sizeY = int(shape[1])
    
    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
        
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in range(sizeX):
        perm = np.random.permutation(sizeY)
        new_vals = np.random.uniform(low=-scale, high=scale, size=(sparsity,))
        vals_norm = np.sqrt((new_vals**2).sum())
        new_vals = scale*new_vals/vals_norm
        values[dx, perm[:sparsity]] = new_vals
    _,v,_ = np.linalg.svd(values)
    values = scale * values/v[0]
    return Tparam(values.astype(theano.config.floatX))

def sample_weights_orth(shape, sparsity=-1, scale=0.01):
    sizeX = int(shape[0])
    sizeY = int(shape[1])

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = np.minimum(sizeY, sparsity)
        
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in range(sizeX):
        perm = np.random.permutation(sizeY)
        new_vals = np.random.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    u,s,v = np.linalg.svd(values)
    values = u * scale

    return Tparam(values.astype(theano.config.floatX))
 
def sample_weights_sutskever(shape):
    n_out = shape[1]
    W_hh_init = np.random.normal(0, 1, (n_out, n_out)) * 0.1
    for i in range(n_out):
        W_hh_init[i][np.random.choice(n_out, n_out-16)] = 0
    pWhh = np.max(abs(scipy.linalg.eigvals(W_hh_init)))
    W_hh_init = W_hh_init / pWhh * 1.1 
	