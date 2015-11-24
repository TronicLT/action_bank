'''Useful functions for operating on videos, all functions should accept Video objects.

Early version of what is now at https://launchpad.net/ndvision

No memmapping of videos.
But, supports multi process loading

Author: Julian Ryde
 Jason Corso

   LICENSE INFO
   See LICENSE file.  Generally, free for academic/research/non-commercial use.  
   For other use, contact us.  Derivative works need to be open-sourced (as this 
   is), or a different license needs to be negotiated.
'''

# standard modules
import os
import tempfile

# numeric/scientific modules
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import pylab
import matplotlib
import subprocess as subp



def float_to_uint8(X):
    '''Quantises an array with values 0 to 1 inclusive to bins 0 to 255 
    inclusive'''
    inds_one = X == 1
    A = np.uint8(X*256) # floor operation
    A[inds_one] = 255  # handle edge case where X == 1 goes to 256
    return A

def asvideo(video_source, factor=1, maxcols=None, lock=None):
    '''Creates a Video object from a range of sources. These can be a video 
    file name or an nd array.
    
     The following arguments are only applied if the video_source is a file path on disk. 
     factor allows for downsizing the video by a constant factor.  Default is no downsizing.
     maxcols allows for downsizing the video such that the columns are a maximum size. If the video 
              columns number is already less than maxcols, then no downsizing occurs.  The aspect ratio of
              the video is maintained.
             maxcols is applied *after* the factor

     lock is an instance of multiprocessing.Manager.Lock that is needed in the case this function
     is being called from a multiprocessing thread.  There are some calls inside of it that
     are not threadsafe and crash (the system pipes to ffmpeg)
    '''

    # if video_source is ndarray like wrap the array and return video object
    if hasattr(video_source, 'shape') and hasattr(video_source, 'dtype'):
        if len(video_source.shape) == 3:
            video_source.shape = video_source.shape + (1,)

        vshape = video_source.shape
        vdtype = video_source.dtype
        vid = Video(frames=vshape[0], rows=vshape[1], columns=vshape[2], bands=vshape[3], dtype=vdtype, initialise=False)
        vid.V = video_source
        return vid

    if not os.path.exists(video_source):
        raise IOError(video_source + ' not found')

    sample_filename = tempfile.mkstemp()
    sample_filename = sample_filename[1]

    # get the lock if we need to
    if lock:
        lock.acquire()

    ffmpeg_options = ['ffmpeg', '-i', video_source,\
                                '-vframes', '1',\
                                '%s%%d.png'%sample_filename ]
    fpipe = subp.Popen(ffmpeg_options,stdout=subp.PIPE,stderr=subp.PIPE)
    fpipe.communicate()

    sample_filename = sample_filename + '1.png'
    sample_img = pylab.imread(sample_filename)
    (height, width, channels) = sample_img.shape
    if factor != 1:
        height = int(height/factor)
        width = int(width/factor)

    if (not maxcols is None) and width > maxcols:
        t = float(maxcols)/float(width)
        width = int(t*width)
        height = int(t*height)

    dirname = tempfile.mkdtemp()

    if not os.path.exists(dirname):
        os.makedirs(dirname);

    ffmpeg_options = ['ffmpeg', '-i', video_source ,\
                                '-s', '%dx%d'%(width,height) ,\
                                '-sws_flags', 'bicubic',\
                                '%s%s' %(dirname,'/frames%04d.png')]
    fpipe = subp.Popen(ffmpeg_options,stdout=subp.PIPE,stderr=subp.PIPE)
    fpipe.communicate()
    
    frame_names = os.listdir(dirname)
    frame_names.sort()
    frame_count = len(frame_names)

    vid = Video(frames=frame_count, rows=height, columns=width, bands=channels, dtype=np.uint8)
    for i, fname in enumerate(frame_names): 
        fullpath = os.path.join(dirname, fname)
        img_array = pylab.imread(fullpath)
        # comes in as floats (0 to 1 inclusive) from a png file
        img_array = float_to_uint8(img_array)
        vid.V[i, ...] = img_array
        # delete temporary files
        os.remove(fullpath)

    vid.source = video_source
    #vid.temp_fname = temp_fname
    os.rmdir(dirname)
    os.remove(sample_filename)

    if lock:
        lock.release()

    return vid

def play(vids, titles=None, rescale=False, loop=False):
    '''Plays a number of videos synchronously'''

    # Test whether we are dealing with a single video or an iterable of videos
    if isinstance(vids, Video):
        vids = [vids, ] # make it a list with a single item

    interpol = 'nearest'

    # Set up the figures
    pylab.ion()
    figs = []
    for i, vid in enumerate(vids):
        is_bool = (vid.V.dtype == bool)
        is_grey = (vid.bands == 1) and not is_bool

        pylab.figure(i)
        pylab.clf()
        if titles is not None:
            pylab.title(titles[i])
        pylab.subplots_adjust(left=0, right=1, bottom=0, top=1)
        first_im = np.squeeze(vid.V[0,...])
        if rescale:
          vmin=vid.V.min()
          vmax=vid.V.max()
        else:
          vmin=0
          vmax=255
        if is_grey:
            fig = pylab.imshow(first_im, matplotlib.cm.gray, interpolation=interpol, vmin=vmin, vmax=vmax) 
        elif is_bool:
            fig = pylab.imshow(first_im, interpolation=interpol, vmin=0, vmax=1)
        else:
            fig = pylab.imshow(first_im, interpolation=interpol,vmin=vmin, vmax=vmax)
        figs.append(fig)

    # TODO assert that all the arrays have the same number of frames
    
    pylab.ioff()
    while True:
        for i in range(len(vid)): # iterate through frames
            for j, vid in enumerate(vids):
                X = np.squeeze(vid.V[i,...])
                fig = figs[j]
                fig.set_data(X)
                pylab.figure(j)
                pylab.draw()
        if not loop:
            break


def _colour_map(im, vmax):
    '''Colour maps a 2D array that is not 3 channel uint8'''
    im = np.float32(im.squeeze())
    # TODO scaling to handle negatives
    im /= vmax
    return pylab.cm.jet(im, bytes=True)[..., :-1] # remove alpha channel

# TODO iterable over frames e.g. vid[0] is zeroth frame
class Video(object):
    '''
    vid.V[0,1,2,3] for row 1 col 2 in colour band 3 for frame 0.

    The image convention of row, column, band follows same convention as 
    imshow, imread.

    vid[0] returns the first frame of the video as an image object
    vid.V[0] returns the first frame of the video as an ndarray

    instantiate from both a file and an array

    TODO: Video should have a way to get its dtype without going to its store to get it.
           vid.dtype rather than vid.V.dtype  (for encapsulation)
    '''

    XD = 2  # X dimension
    YD = 1  # Y dimension
    BD = 3  # Band dimension
    TD = 0  # Temporal dimension

    def __init__(self, frames, rows, columns, bands, dtype=np.uint8, initialise=True):

        vshape = frames, rows, columns, bands

        if initialise:
            self.V = np.zeros(vshape)
        self.frames = frames
        self.rows = rows
        self.columns = columns
        self.bands = bands

    def __getitem__(self, i):
        return self.V[i, ...]

    def __len__(self):
        return self.V.shape[0]

    def __eq__(self, vid):
        # TODO check other attributes as well
        return np.alltrue(self.V == vid.V)

    def copy(self):
        '''Make a copy of the video.'''
        vid = Video(self.frames,self.rows,self.columns,self.bands,self.V.dtype, initialise=False)
        vid.V = self.V.copy()
        return vid 

    def display(self,rescale=None, loop=False):
        '''Plays the video.  Called display so that the method name can be the 
        same for images.'''
        if rescale is None:
            # default rescale to not rescale if it is np.uint8 otherwise do 
            # rescale
            rescale = self.V.dtype != np.uint8
        play(self, rescale=rescale, loop=loop)

    def difference_of_gaussians(self, sigma1, sigma2):
        fV = np.float32(self.V)
        G1 = ndimage.gaussian_filter(fV, sigma=sigma1, order=0)
        G2 = ndimage.gaussian_filter(fV, sigma=sigma2, order=0)
        return asvideo(G1 - G2)

    def filter_DoG(self, sigma):
        '''Compute the Difference of a Gaussian (DoG) on a video array.
           The Video array can be 3D or 4D, just set the sigma appropriately.
           So to get the spatial derivative of each frame sigma=(0, 5, 5, 0)
           and to get the time derivative of each pixel sigma=(5, 0, 0, 0). 
           Note the output video is grayscale and float32.
           sigma is the variance of the Gaussian filter.

           NOTE: does no scaling.
        '''
        # todo should it convert colour video automatically?
        #assert self.bands == 1, 'only applicable to grayscale video'
        #M = ndimage.gaussian_filter1d(np.float32(self.V), sigma=sigma, axis=axis, order=1)
        M = ndimage.gaussian_filter(np.float32(self.V), sigma=sigma, order=1)
        return asvideo(M)

    def rgb2gray(self):
        '''Convert the RGB video into a grayscale version.
           Note the output array is a 3D array
           fname is the filename to the memory map that can be created for the
            output video

           TODO: Check on the ordering of the bands in the video.
           The coefficients from rgb2gray should be
           R 0.2989
           G 0.5870
           B 0.1140

           TODO: support output video to have matching dtype as input video

           NOTE: Requires full storage of the video in memory
        '''
        G = np.asarray( (0.1140, 0.5870, 0.2989) )
        foo = np.tensordot(self.V,G,axes=(3,0))
        foo.shape = foo.shape +  (1,)
        return asvideo(np.uint8(foo))

    # TODO overlap between resize and save_array?
    def save(self,fname):
        '''Save the Video object as a video file, mpg
        '''
        # Using ffmpeg to save the video
        temp_dir = tempfile.mkdtemp()
        self.save_images(temp_dir)
        cmd = 'ffmpeg -i ' + temp_dir+'/%05d.png ' + '-r 30 ' + fname  
        subp.call(cmd, shell = True)
        cmd = 'rm -r ' + temp_dir
        subp.call(cmd, shell = True)

    def scale_DLogistic(self,steepness):
        '''  Scale the video with a double logistic of steepness.
          Double Logistic Definition (bottom):
               http://en.wikipedia.org/wiki/Logistic_function

          Use a double logistic to normalize the 1-band video array in Va and 
          then rescale output to the 0:1 range

          Some asserts that are required.
          1. Video is float32 already

          TODO: Inplace operation should be supported.  But, the sp.sign and 
          sp.exp don't seem to allow out= keywords even though their doc says 
          they do.
        '''

        assert(self.V.dtype == sp.float32)
        Va = self.V
        Va = sp.sign(Va) * (1.0-sp.exp( - (Va / steepness)*(Va / steepness) ))
        Va = (Va+1.0)/2.0

        return asvideo(Va)

    def thumbnails(self, rows=3, cols=3):
        '''Creates a montage image of the video with the number of rows by 
        columns specified.'''
        # TODO refactor merge with module function thumbnails which handles a 
        # list of arrays
        inds = np.linspace(0, self.frames - 1, rows*cols).astype(int)
        thumbs = self.V[inds, ...]

        montage = thumbs.reshape(rows, cols, self.rows, self.columns, self.bands)
        montage = montage.transpose((0, 2, 1, 3, 4))
        montage = montage.reshape(rows*self.rows, cols*self.columns, self.bands)
        return montage

    def save_images(self, dirpath, skip =1):
        ''' Saves all the frames of the video in the given directory path'''
        prefix = '%05d'
        for i in range(self.frames):
            if (i%skip != 0):
                continue
            filename = prefix % i
            filename = os.path.join(dirpath, filename) + '.png'
            sp.misc.imsave(filename, self.V[i, ...])

    def colour_space_dt(self):
        '''Determines the change for each pixel as the euclidean distance in 
        colourspace.'''
        # TODO should this be done pixel by pixel to conserve memory
        dV = np.diff(np.float32(self.V), axis=0)
        dV_dt = pylab.vector_lengths(dV, axis=3)
        return asvideo(np.squeeze(dV_dt))

    def colour_space_edge(self):
        '''Determine spatial gradient in colour space'''
        # TODO this should be done frame by frame to conserve memory
        fV = np.float32(self.V)
        dV_dy = pylab.vector_lengths(np.diff(fV, axis=1), axis=3)
        dV_dx = pylab.vector_lengths(np.diff(fV, axis=2), axis=3)
        # make the shapes compatible by discarding right and bottom edges
        dV_dy = dV_dy[:,:,:-1]
        dV_dx = dV_dx[:,:-1,:]
        return asvideo(np.hypot(dV_dx, dV_dy))

    def apply_to_frames(self, func, args):
        '''Applies the given func to each image frame in the video, returning 
        an appropriately shaped and dtyped video'''
        # TODO this can be parallised perhaps through 
        # http://www.scipy.org/Cookbook/Multithreading?action=AttachFile&do=view&target=handythread.py
        # the GIL is released by numpy when processing an array in C

        im = self.V[0].copy() # copy so that the source video remains untouched
        out_frame = func(im, args)
        shape = (len(self.V), ) + out_frame.shape
        out = np.empty(shape, dtype=out_frame.dtype)
        # TODO should avoid recalculating the out_frame frame?
        # TODO also func should take an out argument to avoid reassigning memory 
        # for the output.
        for i, frame in enumerate(self.V):
            #func(frame, out=out_frame)
            #out[i] = out_frame
            im = frame.copy() # copy so that the source video remains untouched
            out[i] = func(im, args)
        return asvideo(out)

    def colour_map(self, cmap=pylab.cm.jet):
        '''Colour maps a video  that is not 3 channel uint8'''
        vmax = self.V.max()
        return self.apply_to_frames(_colour_map, vmax)
