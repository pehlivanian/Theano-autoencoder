import theano
import theano.tensor as T
import abc

def elu(x, alpha=1.0):
    return T.switch(x > 0, x, T.exp(x) - 1)

###########
# SOLVERS #
###########

class solver(object):
    def __init__(self, decorated, *a, **k):
        self.decorated = decorated

        # required
        assert self.decorated.x
        assert self.decorated.y
        assert self.decorated.params
        assert self.decorated.predict

    @abc.abstractmethod
    def compute_cost_updates(self, *a, **k):
        pass

class gd_solver( solver ):
    ''' Gradient descent solver decorator.

        :type learning_Rate: float
        :param parameters: learning rate for gradient descent
        
        :cost_fn: str
        :param cost_fn: one of 'mse', 'cross-entropy'
    '''

    def __init__(self,
                 decorated,
                 learning_rate=0.01,
                 cost_fn='MSE',
                 *args,
                 **kwargs
                 ):

        super( gd_solver, self).__init__(decorated)
        
        self.x = self.decorated.x
        self.y = self.decorated.y
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def compute_cost_updates(self, *a, **k):

        '''

        Gradient descent update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''

        z = self.decorated.predict(*a, **k)
        if self.cost_fn.upper() == 'MSE':
            L = T.sqrt( T.sum( (self.y - z) ** 2, axis=1))
        elif self.cost_fn.upper() in [ 'CROSS-ENTROPY', 'CROSS_ENTROPY' ]:
            L = T.sum( self.y * T.log(z) + (1-self.y) * T.log(1-z), axis=1)
        cost = T.mean(L)

        grads = T.grad(cost, self.decorated.params)

        updates = [ (param, param - self.learning_rate * grad)
                    for param, grad in zip(self.decorated.params, grads) ]
        
        return (cost, updates)
        
        
class rmsprop_solver( solver ):
    ''' Rmsprop solver decorator.

        :type beta: theano.tensor.scalar
        :param beta: initial learning rate
        
        :type eta: theano.tensor.scalar
        :param eta: decay rate
        
        :type parameters: theano variable
        :param parameters: model parameters to update
        
        :type grads: theano variable
        :param grads: gradients of const wrt parameters
    '''
    
    def __init__(self,
                 decorated,
                 eta=1.e-2,
                 beta=.75,
                 epsilon=1.e-6,
                 cost_fn='MSE',
                 *args,
                 **kwargs
                 ):

        super(rmsprop_solver, self).__init__(decorated)
        
        self.x = self.decorated.x
        self.y = self.decorated.y
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon
        self.cost_fn = cost_fn
    
    def compute_cost_updates(self, *a, **k):

        '''

        RMSProp update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''

        z = self.decorated.predict(*a, **k)
        if self.cost_fn.upper() == 'MSE':
            L = T.sqrt( T.sum( (self.y - z) ** 2, axis=1 ))
        elif self.cost_fn.upper() in [ 'CROSS-ENTROPY', 'CROSS_ENTROPY' ]:
            L = -T.sum(self.y * T.log(z) + (1 - self.y) * T.log(1-z), axis=1)
        else:
            raise RunTimeError('Cost function not defined')

        cost = T.mean(L)

        grads = T.grad(cost, self.decorated.params)
        
        one = T.constant(1.0)

        def _updates(param, cache, df, beta=0.9, eta=1.e-2, epsilon=1.e-6):
            cache_val = beta * cache + (one-beta) * df**2
            x = T.switch( T.abs_(cache_val) <
                          epsilon, cache_val, eta * df /
                          (T.sqrt(cache_val) + epsilon))
            updates = (param, param-x), (cache, cache_val)

            return updates

        caches = [theano.shared(name='c_%s' % param,
                                value=param.get_value() * 0.,
                                broadcastable=param.broadcastable)
                  for param in self.decorated.params]

        updates = []

        for param, cache, grad in zip(self.decorated.params, caches, grads):
            param_updates, cache_updates = _updates(param,
                                                   cache,
                                                   grad,
                                                   beta=self.beta,
                                                   eta=self.eta,
                                                   epsilon=self.epsilon)
            updates.append(param_updates)
            updates.append(cache_updates)

        return (cost, updates)

class neg_feedback_solver( solver ):
    ''' Negative feedback solver decorator.

        :type learning_Rate: float
        
        :cost_fn: str
        :param cost_fn: one of 'mse', 'cross-entropy'
    '''

    def __init__(self,
                 decorated,
                 learning_rate=0.01,
                 cost_fn='MSE',
                 *args,
                 **kwargs
                 ):

        super( neg_feedback_solver, self).__init__(decorated)
        
        self.x = self.decorated.x
        self.y = self.decorated.y
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn

    def compute_cost_updates(self, *a, **k):

        '''

        Negative feedback update
        
        Parameters
        ----------

        Returns
        -------
        tuple
        returns cost and a list of updates for the required parameters
        
        
        '''

        # Works
        # z = self.predict_from_input(self.x)
        # L = T.sqrt( T.sum( (self.y - z) ** 2, axis=1))
        # cost = T.mean(L)
        # grads = T.grad(cost, self.decorated.params)

        # XXX
        # Works?
        z = self.decorated.predict_from_input(self.x - self.decorated.predict(*a, **k)) - self.decorated.predict(*a, **k)
        L = T.sqrt( T.sum( (self.y - z) ** 2, axis=1))
        cost = T.mean(L)
        grads = T.grad(cost, self.decorated.params)

        
        if self.cost_fn.upper() == 'MSE':
            L = T.sqrt( T.sum( (self.y - z) ** 2, axis=1))
        elif self.cost_fn.upper() in [ 'CROSS-ENTROPY', 'CROSS_ENTROPY' ]:
            L = T.sum( self.y * T.log(z) + (1-self.y) * T.log(1-z), axis=1)
        cost = T.mean(L)
        
        grads = T.grad(cost, self.decorated.params)

        updates = [ (param, param - self.learning_rate * grad)
                    for param, grad in zip(self.decorated.params, grads) ]
        
        return (cost, updates)
        
