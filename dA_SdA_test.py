import os
from six.moves import cPickle
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from theano_utils import gd_solver, rmsprop_solver
import dA

def test_dA_SdA(
    train_set_x0=None,                         # training set
    training_epochs=1000,                      # training epochs
    pretraining_epochs=1000,                   # pretraining epochs    
    n_visible=10,                              # num visible layers ~ input dim
    dA_layers_sizes=[8, 6, 4,],                # dA layer sizes, also mlp layer sizes
    symmetric=False,                           # symmetric 
    corruption_levels=[0.],                    # corruption leves for pretraining
    n_trials=1000,                             # num trials in input data
    batch_size=20,                             #  batch size for mini-batches
    tie_weights=True,                          # tie weights in pretraining step
    tie_biases=False,                          # tie biases in pretraining step
    encoder_activation_fn='tanh',              # encoder activation fns, by layer       
    decoder_activation_fn='tanh',              # decoder activation fns, by layer
    global_decoder_activation_fn='tanh',       # decoder activation for final step, to reconstruct input
    initialize_W_as_identity=False,            # initial W is identity, if n_visible=dA_training_layers[-1]
    initialize_W_prime_as_W_transpose=False,   # initial W_prime is W.T, mostly for debugging
    add_noise_to_W=True,                       # add noise to initial W
    noise_limit=0.01,                          # noise limit for uniform W0 noise
    pretrain_solver='rmsprop',                 # pretraining solver type
    finetune_solver='gd',                      # finetuning solver type
    pretrain_solver_kwargs=dict(eta=1.e-2,beta=.8,epsilon=1.e-6),  # for rmsprop, gd solvers
    finetune_solver_kwargs=dict(learning_rate=0.001)               # for rmsprop, gd solvers
    ):

    ####################################
    # SYMMETRIC/NONSYMMETRIC SDA TESTS #
    ####################################
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

    multidA = dA.SdA(
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
        global_decoder_activation_fn=global_decoder_activation_fn,
        initialize_W_as_identity=initialize_W_as_identity,
        initialize_W_prime_as_W_transpose=initialize_W_prime_as_W_transpose,
        add_noise_to_W=add_noise_to_W,
        noise_limit=noise_limit,
        solver_type=pretrain_solver,
        solver_kwargs=pretrain_solver_kwargs)

    if multidA.n_layers == 1:
        assert multidA.mlp_layers[0] == multidA.mlp_layers[-1]
        dA_initial_W = multidA.mlp_layers[-1].initial_W
        dA_initial_b = multidA.mlp_layers[-1].initial_b

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
            decoder_activation_fn=global_decoder_activation_fn,
            initialize_W_as_identity=initialize_W_as_identity,
            initialize_W_prime_as_W_transpose=True,
            initial_W=dA_initial_W,
            initial_b=dA_initial_b,
            add_noise_to_W=add_noise_to_W,
            noise_limit=noise_limit,
            )

        y_hat = da.predict()
        predict = theano.function( [], y_hat, givens={ da.x: train_set_x })
        mse = numpy.mean( numpy.sqrt(numpy.sum((train_set_x.get_value() - predict()) ** 2, axis=1)))

        y_hat2 = multidA.dA_layers[0].predict()
        predict2 = theano.function( [], y_hat2, givens={ multidA.dA_layers[0].x: train_set_x })
        mse2 = numpy.mean( numpy.sqrt(numpy.sum((train_set_x.get_value() - predict2()) ** 2, axis=1)))

        import pdb
        pdb.set_trace()
    
    ########################################
    # END SYMMETRIC/NONSYMMETRIC SDA TESTS #
    ########################################

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
        if pretraining_epochs:
            print('Pre-training layer %i finished, cost ' % (num, ), end=' ')
            print(numpy.mean(c))

    postpretrain_desc = multidA.describe(train_set_x, title='POST PRETRAIN')

    ################
    # END PRETRAIN #
    ################

    ############
    # FINETUNE #
    ############

    import pdb
    pdb.set_trace()

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


    postfinetune_desc = multidA.describe(train_set_x, title='POST FINETUNE')
    

    mlp_output_fns = multidA.mlp_layerOutputs(train_set_x)
    dA_output_fns = multidA.dA_layerOutputs(train_set_x)

    mlp_outputs = [output_fn() for output_fn in mlp_output_fns]
    dA_outputs = [output_fn() for output_fn in dA_output_fns]

    # debug
    #
    # Change weight matrix in middle layer, check appropriate outputs affected
    #
    
    # W3 = multidA.mlp_layers[3].W.get_value()
    # multidA.mlp_layers[3].W.set_value(W3+0.05)

    # mlp_new_outputs = [output_fn() for output_fn in mlp_output_fns]
    # dA_new_outputs = [output_fn() for output_fn in dA_output_fns]    


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
        decoder_activation_fn=global_decoder_activation_fn,
        initialize_W_as_identity=initialize_W_as_identity,
        initialize_W_prime_as_W_transpose=initialize_W_prime_as_W_transpose,
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

    da_desc = da.describe(train_set_x, 'DA')
    
    ################
    # END DA TESTS #
    ################

    # Return (
    #    outputs from stacked model,
    #    cost from stacked model,
    #    output from single-layer model
    #    cost from single-layer model

    print(postpretrain_desc)
    print(postfinetune_desc)
    print(da_desc)
    
    
if __name__ == '__main__':
    
    n_visible = 10
    n_trials = 1000
    batch_size = 100
    pretraining_epochs = 10000
    training_epochs = 10000

    dA_layers_sizes = [8, 6]
    corruption_levels=[0.]
    symmetric=False

    tie_weights = False
    tie_biases = False
    encoder_activation_fn = 'tanh'
    decoder_activation_fn = 'tanh'
    global_decoder_activation_fn = 'tanh'

    initialize_W_as_identity=False
    initialize_W_prime_as_W_transpose = False
    add_noise_to_W = False
    noise_limit = 0.0

    pretrain_solver = 'rmsprop'
    finetune_solver = 'rmsprop'
    pretrain_solver_kwargs = dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
    finetune_solver_kwargs = dict(eta=1.e-4,beta=.7,epsilon=1.e-6)
    

    '''
       Test case non-linear model
       Gupta DK, Arora Y. Singh UK, Gupta JP (@012) Recursive and colony
       optimization for estimation of parameters of a function
       
    '''

    numpy_rng = numpy.random.RandomState(5492)
    
    t     = numpy.linspace(0.05, 1.0001, n_visible)
    z     = numpy_rng.uniform(low=0.05, high=4.0, size=(n_trials,1))
    theta = numpy_rng.uniform(low=0.3, high=1.3, size=(n_trials,1))
    x     = (z * numpy.sin(theta)) * t + (t**2 / z) * numpy.cos(theta)

    test_dA_SdA(
        train_set_x0=x,
        training_epochs=training_epochs,
        n_visible=n_visible,
        dA_layers_sizes=dA_layers_sizes,
        corruption_levels=corruption_levels,
        n_trials=n_trials,
        symmetric=symmetric,
        batch_size=batch_size,
        tie_weights=tie_weights,
        tie_biases=tie_biases,
        encoder_activation_fn=encoder_activation_fn,
        decoder_activation_fn=decoder_activation_fn,
        global_decoder_activation_fn=global_decoder_activation_fn,
        initialize_W_as_identity=initialize_W_as_identity,
        initialize_W_prime_as_W_transpose=initialize_W_prime_as_W_transpose,
        add_noise_to_W=add_noise_to_W,
        noise_limit=noise_limit,
        pretrain_solver='rmsprop',
        finetune_solver='rmsprop',
        pretraining_epochs=pretraining_epochs,
        pretrain_solver_kwargs=pretrain_solver_kwargs,
        finetune_solver_kwargs=pretrain_solver_kwargs
        )
