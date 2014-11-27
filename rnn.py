import numpy as np
import theano
import theano.tensor as T
import pickle
import scipy
import time
from rnn_layers import *

mode = theano.Mode(linker='cvm')

class RNN(object):
    def __init__(self, n_input, n_hidden, n_output, architecture):
        self.structure = locals()
        del self.structure['self']
        print("RNN Dimensions: " + str(n_input) + "->" + str(n_output))
        
        self.x = T.matrix(name='x')
        self.z = T.scalar(name='z', dtype='int32')
        self.rng = np.random.RandomState(1234)
        
        # architecture layers
        if architecture == 'dblstm':
            l1_forw = LSTMLayer(n_input, n_hidden)
            l2_forw = LSTMLayer(n_hidden, n_hidden)
            l1_back = LSTMLayer(n_input, n_hidden, go_backwards=True)
            l2_back = LSTMLayer(n_hidden, n_hidden)
            l_merge = MergeLayer()
            l_final = SigmoidMeanLayer(2 * n_hidden, n_output)
            
            y = l_final(l_merge(l2_forw(l1_forw(self.x)), l2_back(l1_back(self.x))))        
            self.params = l1_forw.params + l2_forw.params + l1_back.params + l2_back.params + l_final.params

        elif architecture == 'ff2drop':
            l1_drop = DropoutLayer(0.5, self.rng)
            l2_drop = DropoutLayer(0.5, self.rng)
            l3_drop = DropoutLayer(0.5, self.rng)
            l1_forw = ForwardLayer(n_input, n_hidden)
            l2_forw = ForwardLayer(n_hidden, n_hidden)
            l3_forw = SigmoidMeanLayer(n_hidden, n_output)
            self.layers = [ l1_drop, l1_forw, l2_drop, l2_forw, l3_drop, l3_forw ]
                    
        elif architecture == 'ff2':
            self.layers = [
                ForwardLayer(n_input, n_hidden, activation=T.tanh),
                ForwardLayer(n_hidden, n_hidden, activation=T.tanh),
                ForwardLayer(n_hidden, 1, activation=T.nnet.sigmoid),
                PoolLayer(),
                lambda x: T.as_tensor_variable([ x, 1-x ])
            ]
            
        elif architecture == 'recti':
            self.layers = [
                ForwardLayer(n_input, n_hidden, activation=RectifiedLinear),
                DropoutLayer(0.5, self.rng),
                ForwardLayer(n_hidden, n_hidden, activation=RectifiedLinear),
                DropoutLayer(0.5, self.rng),
                ForwardLayer(n_hidden, n_output, activation=Identity),
                SoftmaxLayer(),
                lambda x: T.mean(x, 0)
            ]

        elif architecture == 'rnn':
            self.layers = [
                RecurrentLayer(n_input, n_hidden),
                ForwardLayer(n_hidden, n_output, activation=Identity),
                PoolSoftmaxLayer()
            ]
            
        elif architecture == 'gatedrnn':
            self.layers = [
                RecurrentLayer(n_input, n_hidden),
                PoolSoftmaxLayer(n_hidden, n_output)
            ]
        
        elif architecture == 'fastdrop':
            l1 = FastdropRecurrentLayer(n_input, n_hidden, 0.5)
            lf = SigmoidMeanLayer(n_hidden, n_output)
            self.layers = [ l1, lf ]
        
        elif architecture == 'lstm':
            self.layers = [
                LSTMLayer2(n_input, n_hidden),
                PoolSoftmaxLayer(n_hidden, n_output)
            ]

        # connect layers together
        if not hasattr(self, 'params'):
            self.params = [ p for l in self.layers if hasattr(l, 'params') for p in l.params ]
        
        if 'y_pred' not in locals() or 'y_train' not in locals():
            if 'y' in locals():
                y_pred = y_train = y
            else:
                y_pred = y_train = self.x
                for l in self.layers:
                    if isinstance(l, DropoutLayer):
                        y_train = l(y_train, True)
                        y_pred = l(y_pred, False)
                    else:
                        y_train = l(y_train)
                        y_pred = l(y_pred)
        
        # train function
        #theano.printing.debugprint(y_train)
        loss = -T.log(y_train[self.z] * 0.9999 + 0.00005)
        grad = T.grad(loss, self.params)
        self.fprime = theano.function([self.x, self.z], grad, mode=mode)

        # test function
        self.predict = theano.function([self.x], outputs=y_pred, mode=mode)
        
        print('compiled.')
    
    ## info
    def report(self):
        for l in self.layers:
            if "monitor" in dir(l):
                info = '  '.join(['{:s}={:s}'.format(k,v) for k,v in l.monitor().items()])
                print("     [{:d}]\t{:s}".format(self.layers.index(l), info))
    
    ## handle parameter
    def get_theta(self):
        return np.concatenate([p.get_value().ravel() for p in self.params])
    
    def set_theta(self, theta):
        offset = 0
        for param in self.params:
            shape = param.shape.eval()
            len = np.prod(shape)
            param.set_value(theta[offset:offset+len].reshape(shape))
            offset += len
      
    def add_param(self, init, name=None):
        param = theano.shared(np.array(init, dtype=theano.config.floatX), name=name)
        self.params.append(param)
        return param        
        
    ## save/load model
    def save(self, filename):
        weights = self.get_theta()
        
        f = open(filename, 'wb')
        pickle.dump([self.structure, weights], f)
        f.close()
        
    @staticmethod
    def load(filename):
        f = open(filename, 'rb')
        [structure, weights] = pickle.load(f)
        f.close()
        
        net = RNN(*structure)        
        net.set_theta(weights)
        
        return net                
    
    ## batch methods
    def batch_predict(self, set_x):
        pred = [self.predict(x) for x in set_x]
        return pred
        
    def batch_fprime(self, set_x, set_y, set_w=None):
        set_size = float(len(set_x))
        if set_w == None:
            set_w = np.ones(set_size)
            
        gradmean = None
        for x, y, w in zip(set_x, set_y, set_w):
            grad = np.concatenate([np.array(g).flatten() for g in self.fprime(x, y)])
            if gradmean == None:
                gradmean = grad*0.
            gradmean += grad / set_size * w
        return gradmean
    
    ## optimizer methods    
    def opt_fprime(self, params, X, Y, W=None):
        #nX = [x + np.random.normal(0, 0.05, x.shape).astype(np.float32) for x in X]
        self.set_theta(params)
        return self.batch_fprime(X, Y, W)

    def opt_predict(self, params, X):
        self.set_theta(params)
        return self.batch_predict(X)

