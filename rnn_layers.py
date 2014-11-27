import numpy as np
import theano
import theano.tensor as T
import scipy
import utils_params as init

RectifiedLinear = lambda x: T.switch(x > 0, x, 0)
Identity = lambda x: x

## parameter free transformation layers        
class MergeLayer(object):
    def __init__(self):
        self.__dict__.update(locals())
    
    def __call__(self, forward, backward):
        return T.concatenate([forward, backward[::-1]], 1)

class xPoolLayer(object):
    def __init__(self, op=T.mean):
        self.__dict__.update(locals())
     
    def __call__(self, input):
        return self.op(input)
        
class xPoolSoftmaxLayer(object):
    def __init__(self):
        self.__dict__.update(locals())
     
    def __call__(self, input):
        return T.flatten(T.nnet.softmax(T.sum(input, 0)))  

class DropoutLayer(object):
    def __init__(self, p_dropout, rng):
        self.__dict__.update(locals())
    
    def __call__(self, input, dropout):
        if dropout:
            srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
            mask = srng.binomial(n=1, p=1-self.p_dropout, size=(input.shape[1],))
            output = input * T.cast(mask, theano.config.floatX)
        else:
            output = input * (1 - self.p_dropout)
        return output
 
## feed forward layers
class ForwardLayer(object):
    def __init__(self, n_in, n_out, irange=0.05, activation=T.tanh):        
        self.W_xh = init.uniform((n_in, n_out), irange)
        self.b_h = init.zeros((n_out,))
        
        self.params = [ self.W_xh, self.b_h ]
        self.__dict__.update(locals())
        self.infovals = theano.function([], [ T.min(self.W_xh), T.max(self.W_xh), T.mean(self.W_xh), T.std(self.W_xh) ])

    def __call__(self, input):
        return self.activation(T.dot(input, self.W_xh) + self.b_h)

    def monitor(self):
        vals = self.infovals()
        return { 
            'min': '{:.2f}'.format(float(vals[0])),
            'max': '{:.2f}'.format(float(vals[1])),
            'mean': '{:.2f}'.format(float(vals[2])),
            'std': '{:.2f}'.format(float(vals[3]))
        }

class PoolSoftmaxLayer(object):
    def __init__(self, n_in, n_out):     
        self.W_xh = init.normal((n_in, n_out), 0.01)
        self.b_h = init.zeros((n_out,))
        
        self.params = [ self.W_xh, self.b_h ]
        self.__dict__.update(locals())
     
    def __call__(self, input):
        h = T.dot(input, self.W_xh) + self.b_h
        y = T.flatten(T.nnet.softmax(T.sum(h, 0)))
        return y
        
## Vanilla        
class RecurrentLayer(object):
    def __init__(self, n_in, n_out, go_backwards=False):
        self.W_xh = init.normal((n_in, n_out), 0.001)
        self.W_hh = init.sample_weights((n_out, n_out))
        self.h_0 = init.zeros((n_out,))
        self.b_h = init.zeros((n_out,))
        
        self.params = [ self.W_xh, self.b_h, self.W_hh, self.h_0 ]
        self.__dict__.update(locals())

    def __call__(self, input):
        hinput = T.dot(input, self.W_xh) + self.b_h
            
        def step(x_t, h_tm1):
            pre = T.dot(h_tm1, self.W_hh) + x_t
            h_t = T.tanh(pre)
            return h_t
            
        output, _ = theano.scan(step, sequences=hinput, outputs_info=[self.h_0], go_backwards=self.go_backwards)
        return output

## Cho&Bengio 2014
class GatedRecurrentLayer(object):
    def __init__(self, n_in, n_out, go_backwards=False):
        self.W_xr = init.normal((n_in, n_out), 0.01)
        self.W_xz = init.normal((n_in, n_out), 0.01)
        self.W_xh = init.normal((n_in, n_out), 0.01)
        self.W_hr = init.sample_weights_orth((n_out, n_out))
        self.W_hz = init.sample_weights_orth((n_out, n_out))
        self.W_hh = init.sample_weights_orth((n_out, n_out)) 
        self.b_h = init.zeros((n_out, ))
        self.h_0 = init.zeros((n_out, ))
        
        self.params = [ self.W_xh, self.W_hr, self.W_hz, self.W_hh, self.b_h, self.h_0 ]
        self.__dict__.update(locals())
        
    def __call__(self, input):
        rinput = T.dot(input, self.W_xr)
        zinput = T.dot(input, self.W_xz)
        hinput = T.dot(input, self.W_xh)
        
        def step(xr_t, xz_t, xh_t, h_tm1):
            r_t = T.nnet.sigmoid(xr_t + T.dot(h_tm1, self.W_hr))
            z_t = T.nnet.sigmoid(xz_t + T.dot(h_tm1, self.W_hz))
            c_t = T.tanh(xh_t + T.dot(r_t * h_tm1, self.W_hh) + self.b_h)
            h_t = z_t * h_tm1 + (1 - z_t) * c_t
            return h_t
            
        output, _ = theano.scan(step, sequences=[rinput, zinput, hinput], outputs_info=[self.h_0], go_backwards=self.go_backwards)
        return output
        
## LSTM Graves 2013    
class LSTMLayer(object):
    def __init__(self, n_in, n_out, go_backwards=False):
        self.W_xh = init.normal((n_in, n_out*4), 0.1)
        self.b_h = init.zeros((n_out*4,))
        self.W_hh = init.uniform((n_out, n_out*4), 0.1)
        self.W_ci = init.uniform((n_out, ), 0.1)
        self.W_cf = init.uniform((n_out, ), 0.1)
        self.W_co = init.uniform((n_out, ), 0.1)
        self.c_0 = init.zeros((n_out, ))
        self.h_0 = init.zeros((n_out, ))
        
        self.params = [ self.W_xh, self.b_h, self.W_hh, self.W_ci, self.W_cf, self.W_co, self.c_0, self.h_0 ]
        self.__dict__.update(locals())

    def __call__(self, input):
        lstmin = T.dot(input, self.W_xh) + self.b_h
        
        def step(x_t, c_tm1, h_tm1):
            pre = x_t + T.dot(h_tm1, self.W_hh)
            i_t = T.nnet.sigmoid(pre[:self.n_out] + c_tm1 * self.W_ci)
            f_t = T.nnet.sigmoid(pre[self.n_out:2*self.n_out] + c_tm1 * self.W_cf)
            c_t = f_t * c_tm1 + i_t * T.tanh(pre[2*self.n_out:3*self.n_out])
            o_t = T.nnet.sigmoid(pre[3*self.n_out:4*self.n_out] + c_t * self.W_co)
            h_t = o_t * T.tanh(c_t)
            return c_t, h_t

        [_, output], _ = theano.scan(step, sequences=lstmin, outputs_info=[self.c_0, self.h_0], go_backwards=self.go_backwards)
        return output  
		
## LSTM used at Google, Zaremba&Sutskever 2014
class LSTMLayer2(object):
    def __init__(self, n_in, n_out, go_backwards=False):
        self.W_xh = init.uniform((n_in, n_out*4), 0.1)
        self.b_h = init.zeros((n_out*4,))
        self.W_hh = init.uniform((n_out, n_out*4), 0.1)
        self.c_0 = init.zeros((n_out, ))
        self.h_0 = init.zeros((n_out, ))
        
        self.params = [ self.W_xh, self.b_h, self.W_hh, self.c_0, self.h_0 ]
        self.__dict__.update(locals())

    def __call__(self, input):
        pinput = T.dot(input, self.W_xh) + self.b_h
        
        def step(xp_t, c_tm1, h_tm1):
            pre = xp_t + T.dot(h_tm1, self.W_hh)
            i_t = T.nnet.sigmoid(pre[0*self.n_out:1*self.n_out])
            f_t = T.nnet.sigmoid(pre[1*self.n_out:2*self.n_out])
            o_t = T.nnet.sigmoid(pre[2*self.n_out:3*self.n_out])
            g_t = T.tanh(pre[3*self.n_out:4*self.n_out])
            c_t = f_t * c_tm1 + i_t * g_t
            h_t = o_t * T.tanh(c_t)
            return c_t, h_t

        [_, output], _ = theano.scan(step, sequences=pinput, outputs_info=[self.c_0, self.h_0], go_backwards=self.go_backwards)
        return output

## J. Bayer 2014
class FastdropRecurrentLayer(object):
    def __init__(self, n_in, n_out, p_dropout, go_backwards=False):
        W_hh_init = np.random.normal(0, 1, (n_out, n_out)) * 0.1
        for i in range(n_out):
            W_hh_init[i][np.random.choice(n_out, n_out-16)] = 0
        pWhh = np.max(abs(scipy.linalg.eigvals(W_hh_init)))
        W_hh_init = W_hh_init / pWhh * 1.1
        
        self.W_xh = Tparam(np.random.normal(0, 1, (n_in, n_out)) * 0.001)
        self.W_hh = Tparam(W_hh_init)
        self.h_0 = Tparam(np.zeros((n_out,)))
        self.b_h = Tparam(np.zeros((n_out,)))
        self.params = [ self.W_xh, self.b_h, self.W_hh, self.h_0 ]
        self.__dict__.update(locals())

    def __call__(self, input):
        p = 1 - self.p_dropout
        alpha = 1
        mu2_x = T.square(T.mean(input, 0))
        si2_x = T.mean(T.square(input), 0) - mu2_x
        s2 = T.dot(alpha * p * (1-p) * mu2_x + p * si2_x, T.square(self.W_xh))
            
        def step(x_t, h_tm1):
            c_t = T.concatenate([x_t, h_tm1], 0)
            W_ch = T.concatenate([self.W_xh, self.W_hh])
            
            mu = p * T.dot(c_t, W_ch) + self.b_h
            pre = (2 * mu) / T.sqrt(1 + 0.125 * np.pi * (4 * s2))
            h_t = 2 * T.nnet.sigmoid(pre) - 1
            return h_t
            
        output, _ = theano.scan(step, sequences=input, outputs_info=[self.h_0], go_backwards=self.go_backwards)
        return output

## deprecated
class PoolSigmoidLayer(object):
    def __init__(self, n_in, n_out, irange=.0):
        n_out = 1
        self.W_xh = Tparam(np.random.uniform(-irange, irange, (n_in, n_out)))
        self.b_h = Tparam(np.zeros((n_out,)))
        self.params = [ self.W_xh, self.b_h ]
        self.__dict__.update(locals())
    
    def __call__(self, input):
        y_pre = T.dot(input, self.W_xh) + self.b_h
        y_pool = T.sum(y_pre)
        y_post = T.nnet.sigmoid(y_pool)
        y_out = T.as_tensor_variable([ y_post, 1-y_post ])
        return y_out
        
class SigmoidMeanLayer(object):
    def __init__(self, n_in, n_out):
        n_out = 1
        self.W_xh = Tparam(np.random.normal(0, 1, (n_in, n_out)) * 0.1)
        self.b_h = Tparam(np.zeros((n_out,)))
        self.params = [ self.W_xh, self.b_h ]
        self.__dict__.update(locals())
    
    def __call__(self, input):
        y_mean = T.mean(T.nnet.sigmoid(T.dot(input, self.W_xh) + self.b_h))
        return T.as_tensor_variable([ y_mean, 1-y_mean ])
        
        
def Tparam(init):
    param = theano.shared(np.array(init, dtype=theano.config.floatX))
    return param
    