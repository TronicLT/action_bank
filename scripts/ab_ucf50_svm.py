'''
    ab_ucf50_svm.py

    Jason Corso

    Train and test an SVM on the UCF 50 data. (10-fold cross-validation)
    Will produce the result statistic that we reported in the paper.
    
    The processed UCF 50 data is available at 
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to perform 10-fold cross-validation on the UCF 50 data set using the included SVM code.", 
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="path to the directory containing the action bank processed ucf 50 files structured as in root/class/class00_banked.npy.gz for each class")

    args = parser.parse_args()

    vlen = 0

    cdir = os.listdir(args.root)

    if (len(cdir) != 50):
        print "error: found %d classes, but there should be 50"%(len(cdir))

    Ds = []
    Ys = []
    for ci,cl in enumerate(cdir):
        print cl
        files = glob.glob(os.path.join(args.root,cl,'*%s'%(banked_suffix)))
        
        if not vlen:
            fp = gzip.open(files[0],"rb")
            vlen = len(np.load(fp))
            fp.close()
            print "vector length is %d"%vlen

        Di = np.zeros( (len(files),vlen), np.uint8 )
        Yi = np.ones ( (len(files)   )) * ci

        for fi,f in enumerate(files):
            #print f
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

    print "accuracy per fold is outputted"
    print "there is a lot of data here, this will take hours"
    ab_svm.kfoldcv_svm(D,Y,10,cores=1,innerCores=8,useLibLinear=True,useL1R=False)

    #res=ab_svm.SVMLinear(Dtrain,np.int32(Ytrain),Dtest)
    #tp=np.sum(res==Ytest)
    #print 'Accuracy is %.1f%%' % ((np.float64(tp)/Dtest.shape[0])*100)
    #
















