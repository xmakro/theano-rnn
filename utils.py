import climin
import climin.util
import itertools
import math
import numpy as np
from sklearn.metrics import roc_curve, auc

def make_sets(X, Y, ratio):
    np.random.seed(1)
    indices = np.random.permutation(len(X))
    cut = len(indices)*ratio
    train, valid = indices[:cut], indices[cut:]
    train_x = [X[i] for i in train]
    train_y = [Y[i] for i in train]
    valid_x = [X[i] for i in valid]
    valid_y = [Y[i] for i in valid]
    return train_x, train_y, valid_x, valid_y

def make_batches(sets, batch_size=None):
    if batch_size is None:
        args = itertools.repeat((sets, {}))
        batches_per_pass = 1
    else:
        args = climin.util.iter_minibatches(sets, batch_size, [0, 0])
        args = ((i, {}) for i in args)
        batches_per_pass = int(math.ceil(len(sets[0]) / batch_size))
    
    return args, batches_per_pass

def balance_prior(Y):
    c_counts = np.bincount(Y.astype(np.int32)).astype(np.float32)
    c_dist = c_counts / np.sum(c_counts)
    W = [(1. / float(len(c_dist))) / c_dist[y] for y in Y]
    return W
    
def eval_perf(predictions, Y):
    pred = np.array(predictions)
    errs = np.not_equal(np.argmax(pred, axis=1), Y)
    
    labels = np.unique(Y)
    aucs = []
    for i in labels:
        fpr, tpr, thresholds = roc_curve(Y, pred[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
    loss = [-np.log(pred[i][Y[i]] * 0.9999 + 0.00005) for i in range(len(Y))]
            
    return np.mean(errs), np.mean(aucs), np.mean(loss)
