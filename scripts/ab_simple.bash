#!/bin/bash

##
## Compute the action bank representation on a directory of videos
##
##  Reduce video res to 320 x YYY where YYY is computed from the original size 
##   to maintain aspect ratio.  Do action spotting correlation at half-res further.
##  
##  If you have a multicore machine, you should add -c XX for XX cores to use in parallel,
##   e.g., ARGS="-w 320 -g 2 -c 8" for an 8 core machine
##  actionbank.py defaults to assuming you have 2 cores in your machine (can you buy a 
##   machine these days with less?)
##
##  Change INPUT and OUTPUT based on where your videos are. Do not put trailing '/'.

INPUT=/tmp/foo
OUTPUT=/tmp/bar
ARGS="-w 320 -g 2 -c 2"

PYTHONPATH=$PYTHONPATH:../code/

python ../code/actionbank.py $ARGS $INPUT $OUTPUT

