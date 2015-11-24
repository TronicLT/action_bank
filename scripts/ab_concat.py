'''
    ab_concat.py

    Jason Corso

    Script to combine multiple runs (presumably at different scales) of the 
    actionbank.py script over the same input video data, into one concatenated 
    output vector.

    MAKE sure that ../code is in your PYTHONPATH, i.e., export PYTHONPATH=../code
'''

import argparse
import glob
import gzip
import numpy as np
import os
import os.path
import scipy.io as sio


from actionbank import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Utility script to concatenate multiple action bank feature vectors into one long vector.  Will walk the whole directory tree and replicate it to the destination.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root1", help="path to the first input directory")
    parser.add_argument("root2", help="path to the second input directory (must have same exact structure and files, with different content as root1)")
    parser.add_argument("output", help="path to the output directory")

    args = parser.parse_args()


    for dirname, dirnames, filenames in os.walk(args.root1):
        new_dir = dirname.replace(args.root1,args.output)
        subp.call('mkdir '+new_dir,shell = True)


    for dirname, dirnames, filenames in os.walk(args.root1):
        files = glob.glob(os.path.join(dirname,'*%s'%banked_suffix))
        for f in files:
            i1 = path.join(dirname,f)
            i2 = i1.replace(args.root1,args.root2)
            o  = i1.replace(args.root1,args.output)

            fp = gzip.open(i1,"rb")
            v1 = np.load(fp)
            fp.close()

            fp = gzip.open(i2,"rb")
            v2 = np.load(fp)
            fp.close()

            vo = np.concatenate((v1,v2))
            fp = gzip.open(o,"wb")
            np.save(fp,vo)
            fp.close()

