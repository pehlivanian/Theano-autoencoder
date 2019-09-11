import os
from six.moves import cPickle
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from DL_utils import tile_raster_images
from theano_utils import gd_solver, rmsprop_solver
import dA

def test_dA_SdA(
    train_set_x0=None,                  # training set
    training_epochs=1000,               # training epochs
    pretraining_epochs=1000,            # pretraining epochs    
    n_visible=10,                       # num visible layers ~ input dim
    dA_layers_sizes=[8, 6, 4,],         # dA layer sizes, also mlp layer sizes
    symmetric=False,                    # symmetric 
    corruption_levels=[0.],             # corruption leves for pretraining
    n_trials=1000,                      # num trials in input data
    batch_size=20,                      # batch size for mini-batches
    tie_weights=True,                   # tie weights in pretraining step
    tie_biases=False,                   # tie biases in pretraining step
    encoder_activation_fn='tanh',       # encoder activation fns, by layer       
    decoder_activation_fn='tanh',       # decoder activation fns, by layer
    initialize_W_as_identity=False,     # initial W is identity, if n_visible=dA_training_layers[-1]
    add_noise_to_W=True,                # add noise to initial W
    noise_limit=0.01,                   # noise limit for uniform W0 noise
    pretrain_solver='rmsprop',          # pretraining solver type
    finetune_solver='gd',               # finetuning solver type
    pretrain_solver_kwargs=dict(eta=1.e-2,beta=.8,epsilon=1.e-6),  # for rmsprop, gd solvers
    finetune_solver_kwargs=dict(learning_rate=0.001)               # for rmsprop, gd solvers
    ):

    ######################
    # SYMMETRICSDA TESTS #
    ######################
    numpy_rng = numpy.random.RandomState(1349)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    if train_set_x0 is None:
        train_set_x0 = numpy.asarray(
            numpy_rng.uniform(
                low=-1,
                high=1,
                size=(n_trials, n_visible)
                ),
            dtype=theano.config.floatX
            )
        
    train_set_x = theano.shared(value=train_set_x0,
                                name='train_set_x',
                                borrow=True)
        
    n_batches = int( n_trials / batch_size )

    index = T.iscalar('index')
    x = T.matrix('x')

    train_set_x0 = numpy.asarray(
        numpy_rng.uniform(
            low=-1,
            high=1,
            size=(n_trials, n_visible)
            ),
        dtype=theano.config.floatX
    )

    train_set_x = theano.shared(value=train_set_x0,
                                name='train_set_x',
                                borrow=True
                                )

    multidA = dA.symmetricSdA(
        numpy_rng=numpy_rng,
        theano_rng=theano_rng,
        input=x,
        symmetric=symmetric,
        n_visible=n_visible,
        dA_layers_sizes=dA_layers_sizes,
        tie_weights=tie_weights,
        tie_biases=tie_biases,
        encoder_activation_fn=encoder_activation_fn,
        decoder_activation_fn=decoder_activation_fn,
        initialize_W_as_identity=initialize_W_as_identity,
        add_noise_to_W=add_noise_to_W,
        noise_limit=noise_limit,
        solver_type=pretrain_solver,
        solver_kwargs=pretrain_solver_kwargs)

    ############
    # PRETRAIN #
    ############

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    
    pretraining_fns = multidA.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('Pretraining step...')
    for num in range(multidA.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(pretraining_fns[num](index=batch_index))
            if not epoch % 10:
                print('Pre-training layer %i, epoch %d, cost ' % (num, epoch), end=' ')
                print(numpy.mean(c))
        print('Pre-training layer %i finished, cost ' % (num, ), end=' ')
        print(numpy.mean(c))

    # pretraining_fns = multidA.pretraining_functions(train_set_x=train_set_x,
    #                                             batch_size=train_set_x.get_value().shape[0])
    # pretraining_costs = [pretraining_fn(0) for pretraining_fn in pretraining_fns]

    ################
    # END PRETRAIN #
    ################

    ############
    # FINETUNE #
    ############

    if finetune_solver == 'gd':
        solver = gd_solver(multidA, **finetune_solver_kwargs)
    elif finetune_solver == 'rmsprop':
        solver = rmsprop_solver(multidA, **finetune_solver_kwargs)
    else:
        raise RuntimeError( 'finetune_solver type %s not supported' % (finetune_solver,))

    cost, updates = solver.compute_cost_updates()
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens= { x: train_set_x[index*batch_size : (index+1)*batch_size] }
        )

    for epoch in range(training_epochs):
        costs = []
        for batch_index in range(n_batches):
            costs.append(train_da(batch_index))
        print('Training epoch %d, cost ' % epoch, numpy.mean(costs))

    y_hat = T.matrix('y_hat')
    y_hat = multidA.predict()
    
    predict = theano.function(
        [],
        y_hat,
        givens={
            x: train_set_x
            }
    )

    # debug
    #
    # Change weight matrix in middle layer, check appropriate outputs affected
    #
    
    mlp_output_fns = multidA.mlp_layerOutputs(train_set_x)
    dA_output_fns = multidA.dA_layerOutputs(train_set_x)

    mlp_outputs = [output_fn() for output_fn in mlp_output_fns]
    dA_outputs = [output_fn() for output_fn in dA_output_fns]

    # W3 = multidA.mlp_layers[3].W.get_value()
    # multidA.mlp_layers[3].W.set_value(W3+0.05)

    # mlp_new_outputs = [output_fn() for output_fn in mlp_output_fns]
    # dA_new_outputs = [output_fn() for output_fn in dA_output_fns]    

    y_hat = T.matrix('y_hat')
    y_hat = multidA.predict()
    
    predict = theano.function(
        [],
        y_hat,
        givens={
                x: train_set_x
            }
    )

    mse = numpy.mean( numpy.sqrt( numpy.sum( ( train_set_x.get_value() - predict()) ** 2, axis=1)) )

    stacked_outputs, stacked_mse = mlp_outputs, mse

    ##########################
    # END SYMMETRICSDA TESTS #
    ##########################

    ############
    # DA TESTS #
    ############

    n_hidden = dA_layers_sizes[-1]
    corruption_level = corruption_levels[-1]
    
    da = dA.dA(
        numpy_rng=numpy_rng,
        theano_rng = theano_rng,
        input=x,
        n_visible=n_visible,
        n_hidden=n_hidden,
        tie_weights=tie_weights,
        tie_biases=tie_biases,
        encoder_activation_fn=encoder_activation_fn,
        decoder_activation_fn=decoder_activation_fn,
        initialize_W_as_identity=initialize_W_as_identity,
        add_noise_to_W=add_noise_to_W,
        noise_limit=noise_limit,
        )

        
    if finetune_solver == 'gd':
        solver = gd_solver(da, **finetune_solver_kwargs )
    elif finetune_solver == 'rmsprop':
        solver = rmsprop_solver(da, **finetune_solver_kwargs )
    else:
        raise RuntimeError( 'finetune_sovler type %s not supported' % (finetune_solver,))

    cost, updates = solver.compute_cost_updates(corruption_level=corruption_level)
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens= { x: train_set_x[index*batch_size : (index+1)*batch_size] } )
        
    for epoch in range(training_epochs):
        costs = []
        for batch_index in range(n_batches):
            costs.append(train_da(batch_index))
        print('Training epoch %d, cost ' % epoch, numpy.mean(costs))

    y_hat = T.matrix('y_hat')
    y_hat = da.predict()
    
    predict = theano.function(
        [],
        y_hat,
        givens={
                x: train_set_x
            }
    )

    mse = numpy.mean( numpy.sqrt( numpy.sum( ( train_set_x.get_value() - predict()) ** 2, axis=1)) )

    unstacked_output, unstacked_mse = predict(), mse

    ################
    # END DA TESTS #
    ################

    # Return (
    #    outputs from stacked model,
    #    cost from stacked model,
    #    output from single-layer model
    #    cost from single-layer model
    
    return stacked_outputs, stacked_mse, unstacked_output, unstacked_mse
    
    
if __name__ == '__main__':

    n_visible = 20
    n_trials = 1000

    '''
       Test non-linear model
       Gupta DK, Arora Y. Singh UK, Gupta JP (@012) Recursive and colony
       optimization for estimation of parameters of a function
       
    '''

    numpy_rng = numpy.random.RandomState(5492)
    
    t     = numpy.linspace(0.05, 1.0001, 20)
    z     = numpy_rng.uniform(low=0.05, high=4.0, size=(n_trials,1))
    theta = numpy_rng.uniform(low=0.3, high=1.3, size=(n_trials,1))
    x     = (z * numpy.sin(theta)) * t + (t**2 / z) * numpy.cos(theta)

    res = test_dA_SdA(
        train_set_x0=x,
        training_epochs=1000,
        n_visible=20,
        dA_layers_sizes=[18, 16],
        corruption_levels=[0.],
        n_trials=n_trials,
        symmetric=False,
        batch_size=50,
        tie_weights=True,
        tie_biases=False,
        encoder_activation_fn='sigmoid',
        decoder_activation_fn='sigmoid',
        initialize_W_as_identity=False,
        add_noise_to_W=False,
        noise_limit=0.01,
        pretrain_solver='rmsprop',
        finetune_solver='rmsprop',
        pretraining_epochs=1000,
        pretrain_solver_kwargs=dict(eta=1.e-2,beta=.8,epsilon=1.e-6),
        finetune_solver_kwargs=dict(eta=1.e-2,beta=.8,epsilon=1.e-6)
        )
    
