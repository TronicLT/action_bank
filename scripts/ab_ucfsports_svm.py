'''
    ab_ucfsports_svm.py

    Jason Corso

    Train and test an SVM on the UCF Sports data.
    
    The processed UCF Sports data is available at 
    http://www.cse.buffalo.edu/~jcorso/r/actionbank

    MAKE sure that ../code is in your PYTHONPATH, i.e., export PYTHONPATH=../code
    before running this script
'''

import argparse
import glob
import gzip
import numpy as np
import os
import os.path
import random as rnd
import scipy.io as sio
from multiprocessing import Pool


from actionbank import *
import ab_svm

keys = [ 'dive', 'golf', 'kick', 'lift', 'riding', 'run', 'skate', 'pswing', 'hswing', 'walk' ]


def crossvalidate(data, labels, folds, classifier, nclass=1, multiprocessing=True, savedir=None, **classargs):
    """
    A general wrapper method for doing cross-validation using an arbitrary classification scheme.
    data = n by d ndarray containing the data
    labels = n-length ndarray containing the labels
    folds = number of cross-validation folds
    classifier = a function handle - the function should be callable in the format:
        classifier(traindat, trainlabs, testdat, **classargs), and should return a vector containing the
        predicted classes for each member of testdat
    nclass = the number of classifications the classifier should return.  This method assumes
        classifications are returned in order of decreasing likelihood, and calculates accuracy
        both for the highest-ranked (i.e. first) class returned for each point, as well as maximum
        accuracy and relaxed accuracy (i.e. a point is considered correctly classified if ANY
        of the given classifications for it are correct) if nclass is greater than 1
    savedir = an optional argument specifying a directory where the chosen classifier should save
        intermediate information or results.  Each run of the method will be assigned one .mat file
        location in which it may save data.  If this argument is specified, then the classifier method
        given must accept a keyword argument called "saveloc" that determines the location of its
        output file
    **classargs = keyworded list of additional arguments to send to the classifier
    multiprocessing = if True, spawns multiple processes (up to folds or the number of cpus,
        whichever is less) to compute the folds in parallel
   
    returns: folds-length ndarray containing the classifier accuracy for each fold
    """
    labels = normalizelabels(labels)
    #generate folds
    n = data.shape[0]
    items = set(range(n))
    testdivisions = [None] * folds
    for i in range(folds - 1):
        print 'In fold ',str(i)
        testdivisions[i] = rnd.sample(items, int(np.floor(n / folds)))
        items.difference_update(testdivisions[i])
    testdivisions[folds - 1] = list(items)
    items = set(range(n))
    traindivisions = [None] * folds
    for i in range(folds):
        traindivisions[i] = list(items.difference(testdivisions[i]))
    #run folds
    if nclass == 1:
        preds = np.zeros(n, dtype=np.int32)
        accs = np.zeros(folds, dtype=np.float64)
    else:
        preds = np.zeros((n, nclass), dtype=np.int32)
        accs = np.zeros((folds, 3), dtype=np.float64)
    telabs = [None] * folds
    preds = [None] * folds
    #multiprocess over folds
    #if __name__ == '__main__':
    procs = min(detectCPUs(), folds)
    if multiprocessing:
        pool = Pool(processes=procs)
    #create directory if needed
    if(savedir != None and not(os.path.exists(savedir))):
        os.makedirs(savedir)
    for i in xrange(folds):
        #initialize variables
        trdat = data[traindivisions[i], :]
        trlabs = labels[traindivisions[i]]
        tedat = data[testdivisions[i], :]
        telabs[i] = labels[testdivisions[i]]
        args = (trdat, trlabs, tedat)
        if nclass > 1:
            classargs["nclass"] = nclass
        #assign file for saving data
        if savedir != None and multiprocessing == False:
            classargs["saveloc"] = os.path.join(savedir, "fold_" + str(i) + "_dat.mat")
            classargs["testlabs"] = telabs[i]
        if savedir != None and multiprocessing == True:
            classargs["returntosave"] = True
            classargs["testlabs"] = telabs[i]
        #submit task
        if multiprocessing:
            preds[i] = pool.apply_async(classifier, args, classargs)
        else:
            preds = classifier(trdat, trlabs, tedat, **classargs)
            if nclass == 1:
                accs[i] = np.float64((np.equal(preds, telabs[i])).sum()) / len(testdivisions[i])
            else:
                taccs = np.zeros((preds.shape[0], nclass), dtype=np.uint8)
                taccs2 = np.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (np.equal(preds[:, j], telabs[i]))
                    taccs2[j] = np.float64((taccs[:, j]).sum()) / len(testdivisions[i])
                #compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = np.float64((taccs.sum(axis=1) > 0).sum()) / len(testdivisions[i])
       
    #retrieve results and measure accuracy
    for i in xrange(folds):
        if multiprocessing and savedir == None:
            if nclass == 1:
                accs[i] = np.float64((np.equal(preds[i].get(), telabs[i])).sum()) / len(testdivisions[i])
            else:
                preds[i] = preds[i].get()
                taccs = np.zeros((preds[i].shape[0], nclass), dtype=np.uint8)
                taccs2 = np.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (np.equal(preds[i][:, j], telabs[i]))
                    taccs2[j] = np.float64((taccs[:, j]).sum()) / len(testdivisions[i])
                #compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = np.float64((taccs.sum(axis=1) > 0).sum()) / len(testdivisions[i])
               
        elif multiprocessing and savedir != None:
            temp = preds[i].get()
            savemat(os.path.join(savedir, "fold_" + str(i) + "_dat.mat"), temp[1])
            if nclass == 1:
                accs[i] = np.float64((np.equal(temp[0], telabs[i])).sum()) / len(testdivisions[i])
            else:
                preds[i] = temp[0]
                taccs = np.zeros((preds[i].shape[0], nclass), dtype=np.uint8)
                taccs2 = np.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (np.equal(preds[i][:, j], telabs[i]))
                    taccs2[j] = np.float64((taccs[:, j]).sum()) / len(testdivisions[i])
                #compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = np.float64((taccs.sum(axis=1) > 0).sum()) / len(testdivisions[i])
    if multiprocessing:
        pool.close()
    return accs


def normalizelabels(labs):
    """
    Normalizes a set of input labels to a 1-dimensional ndarray of int32s, with labels ranging
    from 0 to n, where n is the number of different labels
    labs = 1-dimensional array-like of labels (will be flattened if not 1-dimensional)
    """
    labs = np.int32(np.array(labs)).flatten()
    labset = np.unique(labs)
    newlabs = np.arange(labset.shape[0])
    temp = np.zeros_like(labs)
    for i in xrange(labset.shape[0]):
        temp[labs==labset[i]] = newlabs[i]
    return temp


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
        else: # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        # Windows:
        if os.environ.has_key("NUMBER_OF_PROCESSORS"):
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
            if ncpus > 0:
                return ncpus
            return 1 # Default
 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to perform leave one out cross-validation on the UCF Sports data set using the included SVM code.", 
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="path to the directory containing the action bank processed ucf sports files structured as in root/class00_banked.npy.gz for each class")

    args = parser.parse_args()

    vlen = 0

    Ds = []
    Ys = []
    for ki,k in enumerate(keys):
        files = glob.glob(os.path.join(args.root,'%s*%s'%(k,banked_suffix)))
        
        if not vlen:
            fp = gzip.open(files[0],"rb")
            vlen = len(np.load(fp))
            fp.close()
            print "vector length is %d"%vlen

        Di = np.zeros( (len(files),vlen), np.uint8 )
        Yi = np.ones ( (len(files)   )) * ki

        for fi,f in enumerate(files):
            print f
            fp = gzip.open(f,"rb")
            Di[fi][:] = np.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    D = Ds[0]
    Y = Ys[0]
    for i,Di in enumerate(Ds[1:]):
        D = np.vstack( (D,Di) )
        Y = np.concatenate( (Y,Ys[i+1]) )

    print D.shape
    print Y.shape

    acc = crossvalidate(D,Y,len(Y),ab_svm.SVMLinear)   

    print acc.mean()

    #res=ab_svm.SVMLinear(Dtrain,np.int32(Ytrain),Dtest)
    #tp=np.sum(res==Ytest)
    #print 'Accuracy is %.1f%%' % ((np.float64(tp)/Dtest.shape[0])*100)
    #
    #c = np.zeros( (len(Ytest),2), np.uint8 )
    #c[:,0] = Ytest
    #c[:,1] = res
    #sio.savemat('ucfsports_confusion.mat',{'M':c},oned_as='column')
















