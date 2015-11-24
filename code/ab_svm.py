"""ab_svm.py

   Jason Corso, Ananth Sadanand, David Johnson

   Code for using an svm classifier with the action bank representation.

   Include methods to
   (1) load the action bank vectors into a usable form
   (2) train a linear svm (using the shogun libraries)
   (3) do cross-validation

   LICENSE INFO
   See LICENSE file.  Generally, free for academic/research/non-commercial use.  
   For other use, contact us.  Derivative works need to be open-sourced (as this 
   is), or a different license needs to be negotiated.

"""

import glob
import gzip
import multiprocessing as multi
import numpy as np
import os
import os.path
import random as rnd
import scipy.io as sio

from shogun.Features import *
from shogun.Kernel import CustomKernel, GaussianKernel, LinearKernel, PolyKernel, Chi2Kernel
from shogun.Classifier import SVMOcas, LibLinear, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L1R_LR

from actionbank import *


def detectCPUs():
    """
    Detects the number of CPUs on a system.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        # Windows:
        if os.environ.has_key("NUMBER_OF_PROCESSORS"):
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
            if ncpus > 0:
                return ncpus
            return 1  # Default


def kfoldcv_svm_aux(i, k, Dk, Yk, threads=1, useLibLinear=False, useL1R=False):
    Di = Dk[0];
    Yi = Yk[0];
    for j in range(k):
        if i == j:
            continue
        Di = np.vstack((Di, Dk[j]))
        Yi = np.concatenate((Yi, Yk[j]))

    Dt = Dk[i]
    Yt = Yk[i]

    # now we train on Di,Yi, and test on Dt,Yt
    #   be careful about how you set the threads (because this is parallel already)
    # res=svm.SVMLinear(Di,np.int32(Yi),Dt,threads=2)
    res = SVMLinear(Di, np.int32(Yi), Dt, threads=threads, useLibLinear=useLibLinear, useL1R=useL1R)
    tp = np.sum(res == Yt)
    print 'Accuracy is %.1f%%' % ((np.float64(tp) / Dt.shape[0]) * 100)

    # examples of saving the results of the folds off to disk
    # np.savez('/tmp/%02d.npz' % (i),Yab=res,Ytrue=Yt)
    # sio.savemat('/tmp/%02d.mat' % (i),{'Yab':res,'Ytrue':np.int32(Yt)},oned_as='column')


def kfoldcv_svm(D, Y, k, cores=1, innerCores=1, useLibLinear=False, useL1R=False):
    """ Do k-fold cross-validation

        Folds are sampled by taking every kth item

        Does the k-fold CV with a fixed svm C constant set to 1.0.
    """

    # print D.shape, Y.shape, len(Y), D.dtype
    # print D.min(), D.max()

    Dk = [];
    Yk = [];

    for i in range(k):
        Dk.append(D[i::k, :])
        # Yk.append(np.squeeze(Y[i::k,:]))
        Yk.append(Y[i::k])
        # print i,Dk[i].shape, Yk[i].shape

    if cores == 1:
        for j in range(1, k):
            kfoldcv_svm_aux(j, k, Dk, Yk, innerCores, useLibLinear, useL1R)
    else:
        # for simplicity, we'll just throw away the first of the ten folds!
        pool = multi.Pool(processes=min(k - 1, cores))
        for j in range(1, k):
            pool.apply_async(kfoldcv_svm_aux, (j, k, Dk, Yk, innerCores, useLibLinear, useL1R))
        pool.close()
        pool.join()  # forces us to wait until all of the pooled jobs are finished


def load_simpleone(root):
    """ Code to load banked vectors at top-level directory root into a feature matrix and class-label vector.

        Classes are assumed to each exist in a single directory just under root.
        Example: root/jump, root/walk would have two classes "jump" and "walk" and in each
        root/X directory, there are a set of _banked.npy.gz files created by the actionbank.py
        script.

        For other more complex data set arrangements, you'd have to write some custom code, this is 
        just an example.

        A feature matrix D and label vector Y are returned.  Rows and D and Y correspond.

        You can use scipy.io to save these as .mat files if you want to export to matlab...
    """

    classdirs = os.listdir(root)
    vlen = 0  # length of each bank vector, we'll get it by loading one in...

    Ds = []
    Ys = []

    for ci, c in enumerate(classdirs):

        cd = os.path.join(root, c)
        files = glob.glob(os.path.join(cd, '*%s' % banked_suffix))
        print "%d files in %s" % (len(files), cd)

        if not vlen:
            fp = gzip.open(files[0], "rb")
            vlen = len(np.load(fp))
            fp.close()
            print "vector length is %d" % (vlen)

        Di = np.zeros((len(files), vlen), np.uint8)
        Yi = np.ones((len(files))) * ci

        for bi, b in enumerate(files):
            fp = gzip.open(b, "rb")
            Di[bi][:] = np.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    D = Ds[0]
    Y = Ys[0]
    for i, Di in enumerate(Ds[1:]):
        D = np.vstack((D, Di))
        Y = np.concatenate((Y, Ys[i + 1]))

    return D, Y


def wrapFeatures(data, sparse=False):
    """
    This class wraps the given set of features in the appropriate shogun feature
    object.
    data = n by d array of features
    sparse = if True, the features will be wrapped in a sparse feature object
   
    returns: your data, wrapped in the appropriate minimal shogun feature type
    """
    if data.dtype == np.float64:
        feats = LongRealFeatures(data.T)
        featsout = SparseLongRealFeatures()
    if data.dtype == np.float32:
        feats = RealFeatures(data.T)
        featsout = SparseRealFeatures()
    elif data.dtype == np.int64:
        feats = LongFeatures(data.T)
        featsout = SparseLongFeatures()
    elif data.dtype == np.int32:
        feats = IntFeatures(data.T)
        featsout = SparseIntFeatures()
    elif data.dtype == np.int16 or data.dtype == np.int8:
        feats = ShortFeatures(data.T)
        featsout = SparseShortFeatures()
    elif data.dtype == np.byte or data.dtype == np.uint8:
        feats = ByteFeatures(data.T)
        featsout = SparseByteFeatures()
    elif data.dtype == np.bool8:
        feats = BoolFeatures()
        featsout = SparseBoolFeatures()
    if sparse:
        featsout.obtain_from_simple(feats)
        return featsout
    else:
        return feats


def SVMLinear(traindata, trainlabs, testdata, C=1.0, eps=1e-5, threads=1, getw=False, useLibLinear=False, useL1R=False):
    """
    Does efficient linear SVM using the OCAS subgradient solver (as interfaced
    by shogun).  Handles multiclass problems using a one-versus-all approach.

    NOTE: the training and testing data should both be scaled such that each
    dimension ranges from 0 to 1
    traindata = n by d training data array
    trainlabs = n-length training data label vector (should be normalized
        so labels range from 0 to c-1, where c is the number of classes)
    testdata = m by d array of data to test
    C = SVM regularization constant
    eps = precision parameter used by OCAS
    threads = number of threads to use
    getw = whether or not to return the learned weight vector from the SVM (note:
        only works for 2-class problems)

    returns:
    m-length vector containing the predicted labels of the instances
         in testdata
    if problem is 2-class and getw == True, then a d-length weight vector is also returned
    """
    numc = trainlabs.max() + 1
    #
    # when using an L1 solver, we need the data transposed
    #
    # trainfeats = wrapFeatures(traindata, sparse=True)
    # testfeats = wrapFeatures(testdata, sparse=True)
    if not useL1R:
        ### traindata directly here for LR2_L2LOSS_SVC
        trainfeats = wrapFeatures(traindata, sparse=False)
    else:
        ### traindata.T here for L1R_LR
        trainfeats = wrapFeatures(traindata.T, sparse=False)
    testfeats = wrapFeatures(testdata, sparse=False)
    if numc > 2:
        preds = np.zeros(testdata.shape[0], dtype=np.int32)
        predprobs = np.zeros(testdata.shape[0])
        predprobs[:] = -np.inf
        for i in xrange(numc):
            # set up svm
            tlabs = np.int32(trainlabs == i)
            tlabs[tlabs == 0] = -1
            # print tlabs
            # print i, ' ', np.sum(tlabs==-1), ' ', np.sum(tlabs==1)
            labels = BinaryLabels(np.float64(tlabs))
            if useLibLinear:
                # Use LibLinear and set the solver type
                svm = LibLinear(C, trainfeats, labels)
                if useL1R:
                    # this is L1 regularization on logistic loss
                    svm.set_liblinear_solver_type(L1R_LR)
                else:
                    # most of my results were computed with this (ucf50)
                    svm.set_liblinear_solver_type(L2R_L2LOSS_SVC)
            else:
                # Or Use SVMOcas
                svm = SVMOcas(C, trainfeats, labels)
            svm.set_epsilon(eps)
            svm.parallel.set_num_threads(threads)
            svm.set_bias_enabled(True)
            # train
            svm.train()
            # test
            res = svm.apply(testfeats).get_labels()
            thisclass = res > predprobs
            preds[thisclass] = i
            predprobs[thisclass] = res[thisclass]
        return preds
    else:
        tlabs = trainlabs.copy()
        tlabs[tlabs == 0] = -1
        labels = Labels(np.float64(tlabs))
        svm = SVMOcas(C, trainfeats, labels)
        svm.set_epsilon(eps)
        svm.parallel.set_num_threads(threads)
        svm.set_bias_enabled(True)
        # train
        svm.train()
        # test
        res = svm.classify(testfeats).get_labels()
        res[res > 0] = 1
        res[res <= 0] = 0
        if getw == True:
            return res, svm.get_w()
        else:
            return res
