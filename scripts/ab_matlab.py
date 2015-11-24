'''
    ab_matlab.py

    Jason Corso

    Convert the Python/Numpy formatted bank representation to a Matlab file format.

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

    parser = argparse.ArgumentParser(description="Utility script to convert the Python/Numpy bank representation into a Matlab .mat format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root", help="path to the input/output file/directory")

    args = parser.parse_args()

    for dirname, dirnames, filenames in os.walk(args.root):
        files = glob.glob(os.path.join(dirname,'*%s'%banked_suffix))
        for f in files:
            mat = f.replace(banked_suffix,banked_matsuffix)
            print mat

            fp = gzip.open(f,"rb")
            v = np.load(fp)
            fp.close()

            sio.savemat(mat,{'v':v},oned_as='column')


