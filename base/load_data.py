import os
import gzip
import glob
import numpy

from actionbank import banked_suffix


def read_cross_data(bank_path):
    """Reads action bank features for cross validation

    Parameters
    ----------
    bank_path: A string object
        The path to a folder containing folders with *_banked.npy.gz files.
        The folders are the classes for the dataset

    Returns
    -------
    output: A tuple of shape(3,)
        The tuple contains two ndarray object with data and labels and list of the class names
    """
    vlen = 0

    cdir = os.listdir(bank_path)
    Ds = []
    Ys = []

    for ci, cl in enumerate(cdir):
        print cl
        files = glob.glob(os.path.join(bank_path, cl, '*%s' % banked_suffix))

        if not vlen:
            fp = gzip.open(files[0], "rb")
            vlen = len(numpy.load(fp))
            fp.close()
            print "vector length is %d" % vlen

        Di = numpy.zeros((len(files), vlen), numpy.uint8)
        Yi = numpy.ones((len(files))) * ci

        for fi, f in enumerate(files):
            # print f
            fp = gzip.open(f, "rb")
            Di[fi][:] = numpy.load(fp)
            fp.close()

        Ds.append(Di)
        Ys.append(Yi)

    D = Ds[0]
    Y = Ys[0]
    for i, Di in enumerate(Ds[1:]):
        D = numpy.vstack((D, Di))
        Y = numpy.concatenate((Y, Ys[i + 1]))

    return D, Y, cdir
