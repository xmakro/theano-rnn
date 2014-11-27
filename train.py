import numpy as np
import climin
import sys
import utils
import time
from rnn import RNN
import dataset_bird as ds
#sys.stdout = open('logg.txt', 'w')

## load data
train_x, train_y = ds.load('train')
valid_x, valid_y = ds.load('valid')

## setup model
idim = len(train_x[0][0])
odim = max(train_y)+1
model = RNN(idim, 300, odim, 'lstm')

## setup optimizer
#train_w = utils.balance_prior(train_y)
params = model.get_theta()
#args, n_batches = utils.make_batches([train_x, train_y], None)
#opt = climin.Rprop(params, model.opt_fprime, args=args, init_step=0.0001)
args, n_batches = utils.make_batches([train_x, train_y], 30)
opt = climin.Adadelta(params, model.opt_fprime, offset=1e-6, args=args)
#args, n_batches = utils.make_batches([train_x, train_y], 30)
#opt = climin.rmsprop.RmsProp(params, model.opt_fprime, step_rate=0.01, args=args)

## perform optimization
epoch = 0
start = time.time()
for info in opt:
    if info['n_iter'] % n_batches == 0:
        epoch += 1
		# end
        if epoch == 100:
            break
		# print performance
        if epoch % 1 == 0:
            terr, tauc, tlos = utils.eval_perf(model.opt_predict(params, train_x), train_y)
            verr, vauc, vlos = utils.eval_perf(model.opt_predict(params, valid_x), valid_y)
            print('train    epoch={:03d}   loss={:.5f}   error={:.5f}   auc={:.5f}'.format(epoch, tlos, terr, tauc))
            print('     valid    epoch={:03d}   loss={:.5f}   error={:.5f}   auc={:.5f}'.format(epoch, vlos, verr, vauc))
		# print report
        if epoch % 10 == 1:
            model.report()
            print('     elapsed={:.5f}'.format(time.time()-start))

## make results
test_x, test_y = ds.load(t)
pred = np.array(model.opt_predict(params, test_x))
terr, tauc, _ = utils.eval_perf(pred, test_y)
print(terr, tauc)
