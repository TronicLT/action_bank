'''
    actionbank.py

    Authors: Sreemanananth Sadanand, Jason J. Corso (sreemana@buffalo.edu,jcorso@buffalo.edu)

    Description: The main driver for action bank.

    INSTRUCTIONS are in the README file. 

   LICENSE INFO
   See LICENSE file.  Generally, free for academic/research/non-commercial use.  
   For other use, contact us.  Derivative works need to be open-sourced (as this 
   is), or a different license needs to be negotiated.


'''

import argparse
import gzip
import multiprocessing as multi
import numpy as np
import os
import os.path as path
import subprocess as subp
import sys
import time as t

import spotting


# some globals

featurized_suffix = '_featurized.npy.gz'
banked_suffix = '_banked.npy.gz'
banked_matsuffix = '_banked.mat'

verbose = None


class ActionBank(object):
    """ 
        Wrapper class storing the data/paths for an ActionBank
        that is based on the action spotting template classifier.
    """

    def __init__(self, bankpath):
        ''' Initialize the bank with the template paths. '''
        self.bankpath = bankpath
        self.templates = os.listdir(bankpath)
        self.size = len(self.templates)
        self.vdim = 73  # hard-coded for now (1^3+2^3+3^3)
        self.factor = 1

    def load_single(self, i):
        """ Load the ith template from the disk."""
        fp = gzip.open(path.join(self.bankpath, self.templates[i]), "rb")
        T = np.float32(np.load(fp))  # force a float32 format
        fp.close()

        # print "loading %s" % self.templates[i]

        # downsample if we need to 
        if self.factor != 1:
            T = spotting.call_resample_with_7D(T, self.factor)

        return T


def apply_bank_template(AB, query, template_index):
    """ Load the bank template (at template_index) and apply it to the query
        video (already featurized).
    """

    if verbose:
        ts = t.time()

    template = AB.load_single(template_index)

    temp_corr = spotting.match_bhatt(template, query)
    temp_corr *= 255
    temp_corr = np.uint8(temp_corr)

    pooled_values = []
    max_pool_3D(temp_corr, 2, 0, pooled_values)

    if verbose:
        te = t.time()
        print "bank template %d in %s seconds" % (template_index, str((te - ts)))

    return pooled_values


def bank_and_save(AB, f, out_prefix, cores=1):
    """
        Load the featurized video (from raw path 'f' that will be translated to featurized
        video path) and apply the bank to it asynchronously

        AB is an action bank instance (pointing to templates)

        If cores is not set or set to 0, a serial application of the bank is made. 
    """

    # first check if we actually need to do this process
    oname = out_prefix + banked_suffix
    if path.exists(oname):
        print "***skipping the bank on video %s (already cached)" % f,
        return

    print "***running the bank on video %s" % f,

    oname = out_prefix + featurized_suffix
    if not path.exists(oname):
        print "Expected the featurized video at %s, not there??? (skipping)" % oname
        return

    fp = gzip.open(oname, "rb")
    featurized = np.load(fp)
    fp.close()

    banked = np.zeros(AB.size * AB.vdim, dtype=np.uint8())

    if cores == 1:
        for k in range(AB.size):
            banked[k * AB.vdim:k * AB.vdim + AB.vdim] = apply_bank_template(AB, featurized, k)
    else:
        res_ref = [None] * AB.size
        pool = multi.Pool(processes=cores)
        for j in range(AB.size):
            res_ref[j] = pool.apply_async(apply_bank_template, (AB, featurized, j))
        pool.close()
        pool.join()  # forces us to wait until all of the pooled jobs are finished
        for k in range(AB.size):
            banked[k * AB.vdim:k * AB.vdim + AB.vdim] = np.array(res_ref[k].get())

    oname = out_prefix + banked_suffix
    fp = gzip.open(oname, "wb")
    np.save(fp, banked)
    fp.close()


def featurize_and_save(f, out_prefix, factor=1, postfactor=1, maxcols=None, lock=None):
    """
        Featurize the video at path 'f'.  But first, check if it exists on the dist
        at the output path already, if so, do not compute it again, just load it.

        lock is a semaphore (multiprocessing.Lock) in the case this is being called 
            from a pool of workers

        This function handles both the prefactor and the postfactor parameters.  
        Be sure to invoke actionbank.py with the same -f and -g parameters if 
        you call it multiple times in the same experiment.

        '_featurize.npz' is the format to save them in.
    """


    oname = out_prefix + featurized_suffix

    if not path.exists(oname):
        print oname, "computing"
        featurized = spotting.featurize_video(f, factor=factor, maxcols=maxcols, lock=lock)

        if postfactor != 1:
            featurized = spotting.call_resample_with_7D(featurized, postfactor)

        of = gzip.open(oname, "wb")
        np.save(of, featurized)
        of.close()
    else:
        print oname, "skipping; already cached"


def add_to_bank(bankpath, newvideos):
    """  Add video(s) as new templates to the bank at path bankpath. """

    if not path.isdir(newvideos):
        (h, t) = path.split(newvideos)
        print "adding %s\n" % (newvideos)
        F = spotting.featurize_video(newvideos);
        of = gzip.open(path.join(bankpath, t + ".npy.gz"), "wb")
        np.save(of, F)
        of.close()
    else:
        files = os.listdir(newvideos)
        for f in files:
            F = spotting.featurize_video(path.join(newvideos, f));
            (h, t) = path.split(f)
            print "adding %s\n" % (t)
            of = gzip.open(path.join(bankpath, t + ".npy.gz"), "wb")
            np.save(of, F)
            of.close()


def max_pool_3D(array_input, max_level, curr_level, output):
    """	Takes a 3D array as input and outputs a feature vector containing the max of each node of the octree,
    max_level takes the max levels of the octree and starts at '0'
    output is a linkedlist So if max-levels =3, then actually 4 levels of octree will be calculated i.e: 0,1,2,3..
    REMEMBER THIS!curr_level is just for programmatic use and should always be set to 0 when the function
    is being called
    """

    # print 'In level ' + str(curr_level)
    if curr_level > max_level:
        return
    else:
        max_val = array_input.max()
        # print str(max_val) +' ' +str(i)


        frames = array_input.shape[0]
        rows = array_input.shape[1]
        cols = array_input.shape[2]
        # np.concatenate((output,[max_val]))	
        # output[i]=max_val
        # i+=1
        output.append(max_val)
        max_pool_3D(array_input[0:frames / 2, 0:rows / 2, 0:cols / 2], max_level, curr_level + 1, output)
        max_pool_3D(array_input[0:frames / 2, 0:rows / 2, cols / 2 + 1:cols], max_level, curr_level + 1, output)
        max_pool_3D(array_input[0:frames / 2, rows / 2 + 1:rows, 0:cols / 2], max_level, curr_level + 1, output)
        max_pool_3D(array_input[0:frames / 2, rows / 2 + 1:rows, cols / 2 + 1:cols], max_level, curr_level + 1, output)

        max_pool_3D(array_input[frames / 2 + 1:frames, 0:rows / 2, 0:cols / 2], max_level, curr_level + 1, output)
        max_pool_3D(array_input[frames / 2 + 1:frames, 0:rows / 2, cols / 2 + 1:cols], max_level, curr_level + 1,
                    output)
        max_pool_3D(array_input[frames / 2 + 1:frames, rows / 2 + 1:rows, 0:cols / 2], max_level, curr_level + 1,
                    output)
        max_pool_3D(array_input[frames / 2 + 1:frames, rows / 2 + 1:rows, cols / 2 + 1:cols], max_level, curr_level + 1,
                    output)


def max_pool_2D(array_input, max_level, curr_level, output):
    """	Takes a 3D array as input and outputs a feature vector containing the max of each node of the octree,
    max_level takes the max levels of the octree and starts at '0'
    output is a linkedlist
    So if max-levels =3, then actually 4 levels of octree will be calculated i.e: 0,1,2,3.. REMEMBER THIS!
    curr_level is just for programmatic use and should always be set to 0 when the function is being called
    """

    # print 'In level ' + str(curr_level)
    if curr_level > max_level:
        return
    else:
        max_val = array_input.max()
        # print str(max_val) +' ' +str(i)



        rows = array_input.shape[0]
        cols = array_input.shape[1]

        output.append(max_val)
        max_pool_2D(array_input[0:rows / 2, 0:cols / 2], max_level, curr_level + 1, output)
        max_pool_2D(array_input[0:rows / 2, cols / 2 + 1:cols], max_level, curr_level + 1, output)
        max_pool_2D(array_input[rows / 2 + 1:rows, 0:cols / 2], max_level, curr_level + 1, output)
        max_pool_2D(array_input[rows / 2 + 1:rows, cols / 2 + 1:cols], max_level, curr_level + 1, output)


################################################################################################
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Main Action Bank routine to transform one or more videos into their respective action bank representations.\
    The system produces some intermediate files along the way and is somewhat computationally intensive.  Before executing some intermediate computation, it will always first check if the file that it would have produced is already present on the file system.  If it is not present, it will regenerate.  So, if you ever need to run from scratch, be sure to specify a new output directory.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-b", "--bank", default="../bank_templates/",
                        help="path to the directory of bank template entries")
    parser.add_argument("-e", "--bankfactor", type=int, default=1,
                        help="factor to reduce the computed bank template matrices down by after loading them.  The bank videos are computed at full-resolution and not downsampled (full res is 300-400 column videos).")
    parser.add_argument("-f", "--prefactor", type=int, default=1,
                        help="factor to reduce the video frames by, spatially; helps for dealing with larger videos (in x,y dimensions); reduced dimensions are treated as the standard input scale for these videos (i.e., reduced before featurizing and bank application)")
    parser.add_argument("-g", "--postfactor", type=int, default=1,
                        help="factor to further reduce the already featurized videos.  The postfactor is applied after featurization (and for space and speed concerns, the cached featurized videos are stored in this postfactor reduction form; so, if you use actionbank.py in the same experiment over multiple calls, be sure to use the same -f and -g parameters.)")
    parser.add_argument("-c", "--cores", type=int, default=2,
                        help="number of cores(threads) to use in parallel")
    parser.add_argument("-n", "--newbank", action="store_true",
                        help="SPECIAL mode: create a new bank or add videos into the bank.  The input is a path to a single video or a folder of videos that you want to be added to the bank path at \'--bank\', which will be created if needed.  Note that all downsizing arguments are ignored; the new video should be in exactly the dimensions that you want to use to add.")
    parser.add_argument("-s", "--single", action="store_true",
                        help="input is just a single video and not a directory tree")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="allow verbose output of commands")
    parser.add_argument("-w", "--maxcols", type=int,
                        help="A different way to downsample the videos, by specifying a maximum number of columns.")
    parser.add_argument("--onlyfeaturize", action="store_true",
                        help="do not compute the whole action bank on the videos; rather, just compute and store the action spotting oriented energy feature videos")
    parser.add_argument("--testsvm", action="store_true",
                        help="After running the bank, test through an svm with k-fold cv.  Assumes a two-layer directory structure was used; this is just an example.  The bank representation is the core output of this code.")
    parser.add_argument("input", help="path to the input file/directory")
    parser.add_argument("output", nargs='?', default="/tmp", help="path to the output file/directory")

    args = parser.parse_args()
    verbose = args.verbose

    # Notes:
    # Single video and whole directory tree processing are intermingled here.

    # Special Mode:
    if args.newbank:
        add_to_bank(args.bank, args.input)
        sys.exit()

    # Preparation
    # Replicate the directory tree in the output root if we are processing multiple files
    if not args.single:
        if args.verbose:
            print 'replicating directory tree for output'
        for dirname, dirnames, filenames in os.walk(args.input):
            new_dir = dirname.replace(args.input, args.output)
            subp.call('mkdir ' + new_dir, shell=True)

    # First thing we do is build the list of files to process
    files = []
    if args.single:
        files.append(args.input)
    else:
        if args.verbose:
            print 'getting list of all files to process'
        for dirname, dirnames, filenames in os.walk(args.input):
            for f in filenames:
                files.append(path.join(dirname, f))

    # Now, for each video, we go through the action bank process

    # Step 1: Compute the Action Spotting Featurized Videos

    manager = multi.Manager()
    lock = manager.Lock()
    pool = multi.Pool(processes=args.cores)
    for f in files:
        pool.apply_async(featurize_and_save,
                         (f, f.replace(args.input, args.output), args.prefactor, args.postfactor, args.maxcols, lock))
    pool.close()
    pool.join()

    if args.onlyfeaturize:
        sys.exit(0)

        # Step 2: Compute Action Bank Embedding of the Videos

    # Load the bank itself
    AB = ActionBank(args.bank)

    if (args.bankfactor != 1):
        AB.factor = args.bankfactor

    # Apply the bank
    #   do not do it asynchronously, as the individual bank elements are done that way
    for fi, f in enumerate(files):
        print "\b\b\b\b\b  %02d%%" % (100 * fi / len(files))
        bank_and_save(AB, f, f.replace(args.input, args.output), args.cores)

    if not args.testsvm:
        sys.exit(0)

    # Step 3:  Try a k-fold cross-validation classification with an SVM in the simple set-up data set case.
    import ab_svm

    (D, Y) = ab_svm.load_simpleone(args.output)
    ab_svm.kfoldcv_svm(D, Y, 10, cores=args.cores)

    # Nothing else to do here
