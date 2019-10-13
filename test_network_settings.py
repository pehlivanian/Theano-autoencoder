import dA
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import gd_solver, rmsprop_solver, neg_feedback_solver

def base_desc(seed=1348):
    numpy_rng = numpy.random.RandomState(seed)
    
    return dict(
        n_visible = 10,
        n_hidden = 5,
        tie_weights = False,
        tie_biases = False,
        encoder_activation_fn = 'tanh',
        decoder_activation_fn = 'tanh',
    
        initialize_W_as_identity=False,
        initialize_W_prime_as_W_transpose = False,
        add_noise_to_W = False,
        noise_limit = 0.0,

        numpy_rng = numpy_rng,
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30)),
        )
        
def gen_autoencoder(**desc):
    return dA.dA(**desc)

def test_base_creation():
    desc = base_desc()
    autoencoder = gen_autoencoder(**desc)

    assert autoencoder

def test_W_identity_initialization():
    desc = base_desc()

    dim = 25

    desc['n_visible'] = dim
    desc['n_hidden'] = dim
    desc['initialize_W_as_identity'] = True
    autoencoder = gen_autoencoder(**desc)

    W = autoencoder.W.get_value()
    assert W.shape == (dim, dim)
    assert numpy.equal(W, numpy.identity(dim)).all()
    
def test_tied_weights():
    desc = base_desc()
    dim = 25

    desc['n_visible'] = dim
    desc['n_hidden'] = dim
    desc['initialize_W_as_identity'] = False
    desc['tie_weights'] = True

    autoencoder = gen_autoencoder(**desc)
    W = autoencoder.initial_W
    W_prime = autoencoder.initial_W_prime

    assert numpy.equal(W.transpose(), W_prime).all()

def test_negative_feedback()
    desc = base_desc()
    dim = 5
    desc['n_visible'] = dim
    desc['n_hidden'] = dim
    desc['initialize_W_as_identity'] = False
    desc['tie_weights'] = False

    autoencoder = gen_autoencoder(**desc)
    solver = 

    

