import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
import dA

rng = np.random.RandomState(401)

A_initial = np.reshape(rng.normal(0,1,4), (2,2))
A = theano.shared(value=A_initial, name='A')
B = matrix_inverse(A)

f = theano.function([], B)


# inverting some mlp transition functions
import theano
import theano.tensor as T
import numpy as np
from numpy.linalg import pinv

def inv_sigmoid( x):
    return T.log( (1-x) / x )

def affine( x, W, b):
    return np.dot( x, W) + b

def inv_affine( y, W, b):
    ''' Assume mult by W on right '''
    Z = W.T
    
    return np.dot(y - b, W_inv)

# This is not going to work; no pseudo-inverse exists



