# test.py
#
#  Action Bank Test routines...
#  For profiling, optimizing, and debugging.
#
#  Jason Corso
#


import gzip
import multiprocessing as multi
import numpy as np
import os
import os.path as path
import subprocess as subp
import sys
import time

import spotting 
from actionbank import *



def run_one_complete(vpath):
    ''' Run the video at vpath completely through the action bank '''

    ts = time.time()
    featurize_and_save(vpath,'/tmp/foo',factor=1,maxcols=100)
    AB = ActionBank("../bank_templates")
    AB.size=10
    bank_and_save(AB,vpath,'/tmp/foo')
    te = time.time()
    print str((te-ts)) + " second elapsed."

# you can profile the complete action bank by running the following in
#  the python shell
#
# import cProfile
# import pstats
# cProfile.run('run_one_complete('test.avi')', 'test_avi.prof')
# p = pstats.Stats('test_avi.prof')
# p.print_stats()
# or
# p.sort_stats('cumulative').print_stats(25)
# 
