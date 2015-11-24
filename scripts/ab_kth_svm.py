"""
    ab_kth_svm.py

    Jason Corso

    Train and test an SVM on the KTH data.
    
    The processed KTH data is available at 
    http://www.cse.buffalo.edu/~jcorso/r/actionbank
    It is split according to the KTH splits

    MAKE sure that ../code is in your PYTHONPATH, i.e., export PYTHONPATH=../code
    before running this script
"""

import argparse
import glob
import gzip
import numpy as np
import os
import os.path
import random as rnd
import scipy.io as sio
from multiprocessing import Pool
import seaborn
import matplotlib.pyplot as plt

from actionbank import *
import ab_svm

keys = ['boxing', 'clapping', 'handwaving', 'jogging', 'running', 'walking']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to perform testing on the KTH data set using the included SVM code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root",
                        help="path to the directory containing the action bank processed kth files structured"
                             " as in root/{train|test}/class/class000_banked.npy.gz for each class")

    args = parser.parse_args()

    vlen = 0

    Ds = []
    Ys = []
    for ki, k in enumerate(keys):
        files = glob.glob(os.path.join(args.root, 'train', '%s' % k, '*%s' % (banked_suffix)))

        if not vlen:
            fp = gzip.open(files[0], "rb")
            vlen = len(np.load(fp))
            fp.close()
            print "vector length is %d" % vlen

        Di = np.zeros((len(files), vlen), np.uint8)
        Yi = np.ones((len(files))) * ki

        for fi, f in enumerate(files):
            print f
            fp = gzip.open(f, "rb")
            Di[fi][:] = np.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    Dtrain = Ds[0]
    Ytrain = Ys[0]
    for i, Di in enumerate(Ds[1:]):
        Dtrain = np.vstack((Dtrain, Di))
        Ytrain = np.concatenate((Ytrain, Ys[i + 1]))

    Ds = []
    Ys = []
    for ki, k in enumerate(keys):
        files = glob.glob(os.path.join(args.root, 'test', '%s' % k, '*%s' % (banked_suffix)))

        Di = np.zeros((len(files), vlen), np.uint8)
        Yi = np.ones((len(files))) * ki

        for fi, f in enumerate(files):
            print f
            fp = gzip.open(f, "rb")
            Di[fi][:] = np.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    Dtest = Ds[0]
    Ytest = Ys[0]
    for i, Di in enumerate(Ds[1:]):
        Dtest = np.vstack((Dtest, Di))
        Ytest = np.concatenate((Ytest, Ys[i + 1]))

    print Dtrain.shape
    print Ytrain.shape
    print Dtest.shape
    print Ytest.shape

    res = ab_svm.SVMLinear(Dtrain, np.int32(Ytrain), Dtest)
    tp = np.sum(res == Ytest)
    print 'Accuracy is %.1f%%' % ((np.float64(tp) / Dtest.shape[0]) * 100)

    # c = np.zeros((len(Ytest), 2), np.uint8)
    # c[:, 0] = Ytest
    # c[:, 1] = res
    # print c
    # sio.savemat('kth_confusion.mat', {'M': c}, oned_as='column')
