from __future__ import division

import numpy
import random as rnd
import scipy.io as sio
from multiprocessing import Pool

from actionbank import *
import ab_svm


def detect_cpus():
    """Detects the number of CPUs on a system.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names.keys():
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        # Windows:
        if "NUMBER_OF_PROCESSORS" in os.environ.keys():
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
            if ncpus > 0:
                return ncpus
            return 1  # Default


def cross_validate(data, labels, folds, classifier=ab_svm.SVMLinear, nclass=1, multiprocessing=True, save_path=None,
                   **classargs):
    """A general wrapper method for doing cross-validation using an arbitrary classification scheme.

    Parameters
    ----------
    data: An ndarray object
        An ndarray of shape (n, d) containing the data

    labels: An ndarray object
        An ndarray object of shape(n,) containing the labels

    folds: An int object
        The number of cross-validation folds

    classifier: A function object
        A function handle - the function should be callable in the format:
        classifier(traindat, trainlabs, testdat, **classargs), and should return a vector containing the
        predicted classes for each member of testdat

    nclass: An int object
        The number of classifications the classifier should return.  This method assumes
        classifications are returned in order of decreasing likelihood, and calculates accuracy
        both for the highest-ranked (i.e. first) class returned for each point, as well as maximum
        accuracy and relaxed accuracy (i.e. a point is considered correctly classified if ANY
        of the given classifications for it are correct) if nclass is greater than 1

    save_path: A string object
        an optional argument specifying a directory where the chosen classifier should save
        intermediate information or results.  Each run of the method will be assigned one .mat file
        location in which it may save data.  If this argument is specified, then the classifier method
        given must accept a keyword argument called "saveloc" that determines the location of its
        output file
    **classargs = keyworded list of additional arguments to send to the classifier
    multiprocessing: A
        if True, spawns multiple processes (up to folds or the number of cpus,
        whichever is less) to compute the folds in parallel

    Returns
    -------
    folds-length ndarray containing the classifier accuracy for each fold
    """
    labels = normalise_labels(labels)
    # generate folds
    n = data.shape[0]
    items = set(range(n))
    test_divisions = [None] * folds

    for i in range(folds - 1):
        print 'In fold ', str(i)
        test_divisions[i] = rnd.sample(items, int(numpy.floor(n / folds)))
        items.difference_update(test_divisions[i])

    test_divisions[folds - 1] = list(items)
    items = set(range(n))
    traindivisions = [None] * folds

    for i in range(folds):
        traindivisions[i] = list(items.difference(test_divisions[i]))

    # run folds
    if nclass == 1:
        preds = numpy.zeros(n, dtype=numpy.int32)
        accs = numpy.zeros(folds, dtype=numpy.float64)
    else:
        preds = numpy.zeros((n, nclass), dtype=numpy.int32)
        accs = numpy.zeros((folds, 3), dtype=numpy.float64)

    telabs = [None] * folds
    preds = [None] * folds

    # multiprocess over folds
    # if __name__ == '__main__':
    procs = min(detect_cpus(), folds)

    if multiprocessing:
        pool = Pool(processes=procs)

    # create directory if needed
    if (save_path is not None) and not (os.path.exists(save_path)):
        os.makedirs(save_path)

    for i in xrange(folds):
        print "Running through fold {0} of {1} !".format(i, folds)
        # initialize variables
        trdat = data[traindivisions[i], :]
        trlabs = labels[traindivisions[i]]
        tedat = data[test_divisions[i], :]
        telabs[i] = labels[test_divisions[i]]
        args = (trdat, trlabs, tedat)
        if nclass > 1:
            classargs["nclass"] = nclass
        # assign file for saving data
        if (save_path is not None) and not multiprocessing:
            classargs["saveloc"] = os.path.join(save_path, "fold_" + str(i) + "_dat.mat")
            classargs["testlabs"] = telabs[i]

        if (save_path is not None) and multiprocessing:
            classargs["returntosave"] = True
            classargs["testlabs"] = telabs[i]

        # submit task
        if multiprocessing:
            preds[i] = pool.apply_async(classifier, args, classargs)
        else:
            preds = classifier(trdat, trlabs, tedat, **classargs)
            if nclass == 1:
                accs[i] = numpy.asarray((numpy.equal(preds, telabs[i])).sum(), dtype='float64') / len(test_divisions[i])
            else:
                taccs = numpy.zeros((preds.shape[0], nclass), dtype=numpy.uint8)
                taccs2 = numpy.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (numpy.equal(preds[:, j], telabs[i]))
                    taccs2[j] = numpy.asarray((taccs[:, j]).sum(), dtype='float64') / len(test_divisions[i])

                # compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                print "Accuracy for fold {0}: {1}%".format(taccs2[0])
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = numpy.asarray((taccs.sum(axis=1) > 0).sum(), dtype='float64') / len(test_divisions[i])

    # retrieve results and measure accuracy
    for i in xrange(folds):

        if multiprocessing and save_path is None:
            if nclass == 1:
                accs[i] = numpy.asarray((numpy.equal(preds[i].get(), telabs[i])).sum(), dtype='float64') / len(
                    test_divisions[i])
            else:
                preds[i] = preds[i].get()
                taccs = numpy.zeros((preds[i].shape[0], nclass), dtype=numpy.uint8)
                taccs2 = numpy.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (numpy.equal(preds[i][:, j], telabs[i]))
                    taccs2[j] = numpy.asarray((taccs[:, j]).sum(), dtype='float64') / len(test_divisions[i])

                # compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                print "Accuracy for fold {0}: {1}%".format(taccs2[0])
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = numpy.asarray((taccs.sum(axis=1) > 0).sum(), dtype='float64') / len(test_divisions[i])

        elif multiprocessing and (save_path is not None):
            temp = preds[i].get()
            sio.savemat(os.path.join(save_path, "fold_" + str(i) + "_dat.mat"), temp[1])
            if nclass == 1:
                accs[i] = numpy.asarray((numpy.equal(temp[0], telabs[i])).sum(), dtype=numpy.float64) / len(
                    test_divisions[i])
            else:
                preds[i] = temp[0]
                taccs = numpy.zeros((preds[i].shape[0], nclass), dtype=numpy.uint8)
                taccs2 = numpy.zeros(nclass)
                for j in xrange(nclass):
                    taccs[:, j] = (numpy.equal(preds[i][:, j], telabs[i]))
                    taccs2[j] = numpy.asarray((taccs[:, j]).sum(), dtype='float64') / len(test_divisions[i])
                # compute accuracy of first prediction, maximum accuracy and relaxed accuracy
                accs[i, 0] = taccs2[0]
                accs[i, 1] = taccs2.max()
                accs[i, 2] = numpy.asarray((taccs.sum(axis=1) > 0).sum(), dtype=numpy.float64) / len(test_divisions[i])
    if multiprocessing:
        pool.close()

    return accs, preds, telabs


def normalise_labels(labs):
    """
    Normalizes a set of input labels to a 1-dimensional ndarray of int32s, with labels ranging
    from 0 to n, where n is the number of different labels

    Parameters
    ----------
    labs: 1-dimensional array-like of labels (will be flattened if not 1-dimensional)
    """
    labs = numpy.array(labs).astype('int32').flatten()
    labset = numpy.unique(labs)
    newlabs = numpy.arange(labset.shape[0])
    temp = numpy.zeros_like(labs)
    for i in xrange(labset.shape[0]):
        temp[labs == labset[i]] = newlabs[i]
    return temp
