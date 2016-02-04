'''
    spotting.py
    Sreemanananth Sadanand and Jason Corso
    sreemana@buffalo.edu,jcorso@buffalo.edu
    Implementation of the Action Spotting method

    K.G. Derpanis, M. Sizintsev, K. Cannons, and R. P. Wildes. Efficient
    action spotting based on a spacetime oriented structure representation. In 
    Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 
    2010.

    Implemented and released as a part of the Action Bank codebase.  If you use this code, 
    please cite our Action Bank paper.
    S. Sadanand and J. J. Corso. Action Bank: A high-level representation of 
    activity in video. In Proceedings of IEEE Conference on Computer Vision and 
    Pattern Recognition, 2012.

    
   LICENSE INFO
   See LICENSE file.  Generally, free for academic/research/non-commercial use.  
   For other use, contact us.  Derivative works need to be open-sourced (as this 
   is), or a different license needs to be negotiated.
'''

import matplotlib
import numpy as np
import numpy.random as r
from numpy.fft import fftn,ifftn
from os import sys
import scipy.ndimage as ndimage
import scipy.signal as sg
import time as t

import scipy.weave as weave

import video

#matplotlib.use('MacOSX')



def imgInit3DG3(vid):
	
	# Filters formulas given in PHD dissertation page 244. Attached in mail. 
	img=np.float32(vid.V)

	#if img.max() > 1.0:
	#	img = img / 255.0;
	
	SAMPLING_RATE = 0.5;
	C=0.184
	i  = np.multiply(SAMPLING_RATE,range(-6,7,1))
	f1 = -4*C*(2*(i**3)-3*i)*np.exp(-1*i**2)
	f2 = i*np.exp(-1*i**2)
	f3 = -4*C*(2*(i**2)-1)*np.exp(-1*i**2)
	f4 = np.exp(-1*i**2)
	f5 = -8*C*i*np.exp(-1*i**2)
	
	filter_size=np.size(i)
	# Convolving image with filters mentioned in pg 245. Note the different filters along the different axes.
	# X-axis direction goes along the colums(this is how istare.video objects are stored. (Frames,rows,Colums)) and hence axis=2. Similarly axis=1 for y direction and axis=0 for z direction
	G3a_img = ndimage.convolve1d(img,    f1,axis=2,mode='reflect');   # x-direction
	G3a_img = ndimage.convolve1d(G3a_img,f4,axis=1,mode='reflect');   # y-direction
	G3a_img = ndimage.convolve1d(G3a_img,f4,axis=0,mode='reflect'); # z-direction
	#*****************************************************

	G3b_img = ndimage.convolve1d(img,    f3,axis=2,mode='reflect');   # x-direction
	G3b_img = ndimage.convolve1d(G3b_img,f2,axis=1,mode='reflect');   # y-direction
	G3b_img = ndimage.convolve1d(G3b_img,f4,axis=0,mode='reflect'); # z-direction
	#*****************************************************

	G3c_img = ndimage.convolve1d(img,    f2,axis=2,mode='reflect');   # x-direction
	G3c_img = ndimage.convolve1d(G3c_img,f3,axis=1,mode='reflect');   # y-direction
	G3c_img = ndimage.convolve1d(G3c_img,f4,axis=0,mode='reflect'); # z-direction
	
	#*****************************************************
	
	G3d_img = ndimage.convolve1d(img,    f4,axis=2,mode='reflect');   # x-direction
	G3d_img = ndimage.convolve1d(G3d_img,f1,axis=1,mode='reflect');   # y-direction
	G3d_img = ndimage.convolve1d(G3d_img,f4,axis=0,mode='reflect'); # z-direction

	#*****************************************************

	G3e_img = ndimage.convolve1d(img,    f3,axis=2,mode='reflect');   # x-direction
	G3e_img = ndimage.convolve1d(G3e_img,f4,axis=1,mode='reflect');   # y-direction
	G3e_img = ndimage.convolve1d(G3e_img,f2,axis=0,mode='reflect'); # z-direction
		
	#*****************************************************

	G3f_img = ndimage.convolve1d(img,    f5,axis=2,mode='reflect');   # x-direction
	G3f_img = ndimage.convolve1d(G3f_img,f2,axis=1,mode='reflect');   # y-direction
	G3f_img = ndimage.convolve1d(G3f_img,f2,axis=0,mode='reflect'); # z-direction

	#*****************************************************

	G3g_img = ndimage.convolve1d(img,    f4,axis=2,mode='reflect');   # x-direction
	G3g_img = ndimage.convolve1d(G3g_img,f3,axis=1,mode='reflect');   # y-direction
	G3g_img = ndimage.convolve1d(G3g_img,f2,axis=0,mode='reflect'); # z-direction
	
	#*****************************************************

	G3h_img = ndimage.convolve1d(img,    f2,axis=2,mode='reflect');   # x-direction
	G3h_img = ndimage.convolve1d(G3h_img,f4,axis=1,mode='reflect');   # y-direction
	G3h_img = ndimage.convolve1d(G3h_img,f3,axis=0,mode='reflect'); # z-direction

	#*****************************************************

	G3i_img = ndimage.convolve1d(img,    f4,axis=2,mode='reflect');   # x-direction
	G3i_img = ndimage.convolve1d(G3i_img,f2,axis=1,mode='reflect');   # y-direction
	G3i_img = ndimage.convolve1d(G3i_img,f3,axis=0,mode='reflect'); # z-direction
	
	#*****************************************************

	G3j_img = ndimage.convolve1d(img,    f4,axis=2,mode='reflect');   # x-direction
	G3j_img = ndimage.convolve1d(G3j_img,f4,axis=1,mode='reflect');   # y-direction
	G3j_img = ndimage.convolve1d(G3j_img,f1,axis=0,mode='reflect'); # z-direction

	#*****************************************************
	
	return (G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)

def imgSteer3DG3(direction,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img):

	a=direction[0]
	b=direction[1]
	c=direction[2]

	# Linear Combination of the G3 basis filters. See Pg 243, Table B.8 of Phd dissertation.
	img_G3_steer= G3a_img*a**3     \
                + G3b_img*3*a**2*b \
                + G3c_img*3*a*b**2 \
                + G3d_img*b**3     \
                + G3e_img*3*a**2*c \
                + G3f_img*6*a*b*c  \
                + G3g_img*3*b**2*c \
                + G3h_img*3*a*c**2 \
                + G3i_img*3*b*c**2 \
                + G3j_img*c**3
	
	return img_G3_steer


def calc_total_energy(n_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img):
    # This is where the 4 directions in eq3 are calculated.
    direction0= get_directions(n_hat,e_axis,0)
    direction1= get_directions(n_hat,e_axis,1)
    direction2= get_directions(n_hat,e_axis,2)
    direction3= get_directions(n_hat,e_axis,3)

    # Given the 4 directions, the energy along each of the 4 directions are found sepreately and then added. This gives the total energy along one spatio-temporal direction.
    #print 'All directions done.. calculating energy along 1st direction'
    energy1= calc_directional_energy(direction0,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    #print'Now along second direction'
    energy2= calc_directional_energy(direction1,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)

    #print 'Now along third direction'
    energy3= calc_directional_energy(direction2,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)

    #print 'Now along fourth direction'
    energy4= calc_directional_energy(direction3,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)


    total_energy= energy1+energy2+energy3+energy4
    #print 'Total energy calculated'

    return total_energy

def calc_directional_energy(direction,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img):
	G3_steered= imgSteer3DG3(direction,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
	unnormalised_energy= G3_steered**2
	return unnormalised_energy

def get_directions(n_hat,e_axis,i):
	n_cross_e=np.cross(n_hat,e_axis)

	theta_na= n_cross_e/mag_vect(n_cross_e)
	theta_nb= np.cross(n_hat,theta_na)

	theta_i= np.cos((np.pi*i)/(4))*theta_na + np.sin((np.pi*i)/4)*theta_nb # Gettin theta Eq3

	orthogonal_direction= np.cross(n_hat,theta_i) # Angle in spatial domain
	orthogonal_magnitude= mag_vect(orthogonal_direction) # Its magnitude
		
	#alpha=	orthogonal_direction[2]/ orthogonal_magnitude	
	#beta =	orthogonal_direction[1]/ orthogonal_magnitude
	#gamma=	orthogonal_direction[0]/ orthogonal_magnitude
	
	mag_theta=mag_vect(theta_i)
	
	alpha=theta_i[0]/mag_theta
	beta=theta_i[1]/mag_theta
	gamma=theta_i[2]/mag_theta

	return ([alpha,beta,gamma])

def mag_vect(a):
	mag=np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
	return mag


def calc_spatio_temporal_energies(vid):
    ''' 
    This function returns a 7 Feature per pixel video corresponding to 7 energies oriented towards the left, right, up, down, flicker, static and 'lack of structure' 	  spatio-temporal energies.

    JJC:  Returned as a list of seven grayscale-videos
    '''
    ts=t.time()
    #print 'Generating G3 basis Filters.. Function definition in G3H3_helpers.py'
    (G3a_img\
    ,G3b_img\
    ,G3c_img\
    ,G3d_img\
    ,G3e_img\
    ,G3f_img\
    ,G3g_img\
    ,G3h_img\
    ,G3i_img\
    ,G3j_img) = imgInit3DG3(vid)

    #'Unit normals for each spatio-temporal direction. Used in eq 3 of paper'
    root2 = 1.41421356
    leftn_hat    =  ([-1/root2, 0,      1/root2])
    rightn_hat   =	([1/root2,  0,      1/root2])
    downn_hat    =	([0,        1/root2,1/root2])
    upn_hat      =	([0,       -1/root2,1/root2])
    flickern_hat =	([0,        0,      1      ])
    staticn_hat  =	([1/root2,  1/root2,0      ])


    e_axis = ([0,1,0])
    sigmag=1.0

    #print('Calculating Left Oriented Energy')	
    energy_left= calc_total_energy(leftn_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_left=ndimage.gaussian_filter(energy_left,sigma=sigmag)

    #*******************************
    #print('Calculating Right Oriented Energy')	
    energy_right= calc_total_energy(rightn_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_right=ndimage.gaussian_filter(energy_right,sigma=sigmag)

    #*******************************
    #print('Calculating Up Oriented Energy')	
    energy_up= calc_total_energy(upn_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_up=ndimage.gaussian_filter(energy_up,sigma=sigmag)

    #*******************************
    #print('Calculating Down Oriented Energy')	
    energy_down= calc_total_energy(downn_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_down=ndimage.gaussian_filter(energy_down,sigma=sigmag)

    #*******************************
    #print('Calculating Static Oriented Energy')
    energy_static= calc_total_energy(staticn_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_static=ndimage.gaussian_filter(energy_static,sigma=sigmag)

    #*******************************
    #print('Calculating Flicker Oriented Energy')
    energy_flicker= calc_total_energy(flickern_hat,e_axis,G3a_img,G3b_img,G3c_img,G3d_img,G3e_img,G3f_img,G3g_img,G3h_img,G3i_img,G3j_img)
    energy_flicker=ndimage.gaussian_filter(energy_flicker,sigma=sigmag)
    #*******************************
    #print 'Normalising Energies'

    c=np.max([np.mean(energy_left),np.mean(energy_right),np.mean(energy_up),np.mean(energy_down),np.mean(energy_static),np.mean(energy_flicker)])*1/100 
    #print ("normalize with c %d" %c)

    # norm_energy is the sum of the consort planar energies. c is the epsillon value in eq5
    norm_energy = 		  energy_left    \
                        + energy_right   \
                + energy_up      \
                + energy_down    \
                + energy_static  \
                + energy_flicker \
                + c

    # Normalisation with consort planar energy
    vid_left_out     = video.asvideo( energy_left    / ( norm_energy ))
    vid_right_out    = video.asvideo( energy_right   / ( norm_energy ))
    vid_up_out       = video.asvideo( energy_up      / ( norm_energy ))
    vid_down_out     = video.asvideo( energy_down    / ( norm_energy ))
    vid_static_out   = video.asvideo( energy_flicker / ( norm_energy ))
    vid_flicker_out  = video.asvideo( energy_static  / ( norm_energy ))
    vid_structure_out= video.asvideo( c              / ( norm_energy ))

    #print 'Done'
    te=t.time()
    print str((te-ts)) + ' Seconds to execution (calculating energies)'
    return   vid_left_out      \
            ,vid_right_out     \
            ,vid_up_out        \
            ,vid_down_out      \
            ,vid_static_out    \
            ,vid_flicker_out   \
            ,vid_structure_out


def resample_with_gaussian_blur(input_array,sigma_for_gaussian,resampling_factor):
	sz=input_array.shape	
	
	gauss_temp=ndimage.gaussian_filter(input_array,sigma=sigma_for_gaussian)
	
	resam_temp=sg.resample(gauss_temp,axis=1,num=sz[1]/resampling_factor)
	
	resam_temp=sg.resample(resam_temp,axis=2,num=sz[2]/resampling_factor)
	
	
	return (resam_temp)	

def resample_without_gaussian_blur(input_array,resampling_factor):
	sz=input_array.shape	
	
	resam_temp=sg.resample(input_array,axis=1,num=sz[1]/resampling_factor)
	resam_temp=sg.resample(resam_temp,axis=2,num=sz[2]/resampling_factor)
	return (resam_temp)	

def linstretch(A):
    min_res=A.min()	
    max_res=A.max()
    return (A-min_res)/(max_res-min_res)

def call_resample_with_7D(input_array,factor):
	sz=input_array.shape
	temp_output=np.zeros((sz[0],sz[1]/factor,sz[2]/factor,7),dtype=np.float32)
	for i in range(7): 	
		temp_output[:,:,:,i]=resample_with_gaussian_blur(input_array[:,:,:,i],1.25,factor)	
	return linstretch(temp_output) 

def featurize_video(vid_in,factor=1,maxcols=None,lock=None):
    '''
      Takes a video, converts it into its 5 dim of "pure" oriented energy.
      This is a slight deviation from the original Action Spotting method, which uses 
      all 7 dimensions.  We found the extra two dimensions (static and lack of structure) 
      to decrease performance and sharpen the other 5 motion energies when used to remove
      "background".

      Input: vid_in is either a numpy video array or a path to a video file
        lock is a multiprocessing.Lock that is needed if this is being called
              from multiple threads.
    '''	
    # Converting video to video object (if needed)
    svid_obj=None
    if type(vid_in) is video.Video:
        svid_obj = vid_in
    else:
        svid_obj=video.asvideo(vid_in,factor,maxcols=maxcols,lock=lock)

    if svid_obj.V.shape[3] > 1:
        svid_obj=svid_obj.rgb2gray()

    # Calculating and storing the 7D feature videos for the search video
    left_search,right_search,up_search,down_search,static_search,flicker_search,los_search=calc_spatio_temporal_energies(svid_obj)

    # Compressing all search feature videos to a single 7D array.
    search_final=compress_to_7D(left_search,right_search,up_search,down_search,static_search,flicker_search,los_search,7)

    #do not force a downsampling.  
    #res_search_final=call_resample_with_7D(search_final)

    # Taking away static and structure features and normalising again
    fin = normalize(takeaway(linstretch(search_final)))

    return fin

	
###########################
##### Correlation Routines
###########################


def match_bhatt(T,A):
    ''' Implements the Bhattacharyya Coefficient Matching via FFT 
            Forces a full correlation first and then extracts the center 
         portion of the convolution

        Our bhatt correlation, that assumes the static and lack of 
        structure channels (4 and 6) have already been subtracted out 
        See takeaway below.
    '''

    szT   = T.shape
    szA   = A.shape
    #szOut = [szA[0],szA[1],szA[2]]
    szOut = [szA[0]+szT[0],szA[1]+szT[1],szA[2]+szT[2]]

    Tsqrt = T**0.5
    T[np.isnan(T)] = 0
    T[np.isinf(T)] = 0
    Asqrt = A**0.5

    M = np.zeros(szOut,dtype=np.float32)

    for i in [0,1,2,3,5]:
        rotTsqrt = np.squeeze(Tsqrt[::-1,::-1,::-1,i])
        Tf = fftn(rotTsqrt,szOut)
        Af = fftn(np.squeeze(Asqrt[:,:,:,i]),szOut)
        M = M + Tf*Af

    #M = ifftn(M).real / np.prod([szT[0],szT[1],szT[2]])
    # normalize by the number of nonzero locations in the template rather than 
    #  total number of location in the template
    temp = np.sum( (T.sum(axis=3)>0.00001).flatten() )
    #print (np.prod([szT[0],szT[1],szT[2]]),temp)
    M = ifftn(M).real / temp

    return M[szT[0]/2:szA[0]+szT[0]/2, \
             szT[1]/2:szA[1]+szT[1]/2, \
             szT[2]/2:szA[2]+szT[2]/2]

def match_bhatt_weighted(T,A):
    ''' Implements the Bhattacharyya Coefficient Matching via FFT 
            Forces a full correlation first and then extracts the center 
         portion of the convolution

        Raw Spotting bhatt correlation (uses weighting on the static and 
        lack of structure channels).
    '''

    szT   = T.shape
    szA   = A.shape
    #szOut = [szA[0],szA[1],szA[2]]
    szOut = [szA[0]+szT[0],szA[1]+szT[1],szA[2]+szT[2]]

    W     = 1 - T[:,:,:,6] - T[:,:,:,4]

    # apply the weight matrix to the template after the sqrt op.
    T     = T**0.5
    Tsqrt = T*W.reshape([szT[0],szT[1],szT[2],1])
    Asqrt = A**0.5

    M = np.zeros(szOut,dtype=np.float32)

    for i in range(7):
        rotTsqrt = np.squeeze(Tsqrt[::-1,::-1,::-1,i])
        Tf = fftn(rotTsqrt,szOut)
        Af = fftn(np.squeeze(Asqrt[:,:,:,i]),szOut)
        M = M + Tf*Af

    #M = ifftn(M).real / np.prod([szT[0],szT[1],szT[2]])
    # normalize by the number of nonzero locations in the template rather than 
    #  total number of location in the template
    temp = np.sum( (T.sum(axis=3)>0.00001).flatten() )
    #print (np.prod([szT[0],szT[1],szT[2]]),temp)
    M = ifftn(M).real / temp

    return M[szT[0]/2:szA[0]+szT[0]/2, \
             szT[1]/2:szA[1]+szT[1]/2, \
             szT[2]/2:szA[2]+szT[2]/2]


def match_ncc(T,A):
	''' Implements normalized cross-correlation of the template to the search video A

        Will do weighting of the template inside here...
	'''
	szT   = T.shape
	szA   = A.shape

	# leave this in here if you want to weight the template
	W = 1 - T[:,:,:,6] - T[:,:,:,4]
	T = T*W.reshape([szT[0],szT[1],szT[2],1])

	split(video.asvideo(T)).display()

	M = np.zeros([szA[0],szA[1],szA[2]],dtype=np.float32)

	for i in range(7):
		if i==4 or i==6:
			continue

		t = np.squeeze(T[:,:,:,i])

		# need to zero-mean the template per the normxcorr3d function below
		t = t - t.mean()
		M = M + normxcorr3d(t,np.squeeze(A[:,:,:,i]))

	M = M / 5

	return M

def normxcorr3d(T,A):
	# Matlab code for this normalized cross-correlation provided 
	#  here: http://www.cs.ubc.ca/~deaton/tut/normxcorr3.html
	# This is the python version of the same.
	# Implements: http://scribblethink.org/Work/nvisionInterface/nip.html

	szT = np.array(T.shape)
	szA = np.array(A.shape)
	if (szT.any()>szA.any()):
		print 'Template must be smaller than the Search video'
		sys.exit(0)
	pSzT = np.prod(szT)

	intImgA=integralImage(A,szT)
	intImgA2=integralImage(A*A,szT)

	szOut = intImgA[:,:,:].shape

	rotT = T[::-1,::-1,::-1]
	fftRotT = fftn(rotT,s=szOut)
	fftA = fftn(A,s=szOut)
	corrTA = ifftn(fftA*fftRotT).real

	# Numerator calculation
	num = (corrTA - intImgA*np.sum(T.flatten())/pSzT)/(pSzT-1)
	# Denominator calculaton
	denomA = np.sqrt((intImgA2 - (intImgA**2)/pSzT)/(pSzT-1))
	denomT = np.std(T.flatten())
	denom=denomT*denomA

	C=num/denom
	nanpos=np.isnan(C)
	C[nanpos]=0

	return C[szT[0]/2:szA[0]+szT[0]/2, \
             szT[1]/2:szA[1]+szT[1]/2, \
             szT[2]/2:szA[2]+szT[2]/2]

def integralImage(A,szT):

	'''A=np.zeros([2,2,3])
	A[:,:,0]=np.array(([1,2],[3,4]))
	A[:,:,1]=np.array(([5,6],[7,8]))
	A[:,:,2]=np.array(([9,10],[11,12]))
	szT=np.array([2,2,3])
	'''

	szA = np.array(A.shape) #A is just a 3d matrix here. 1 Feature video
	B=np.zeros(szA+2*szT-1,dtype=np.float32)
	B[szT[0]:szT[0]+szA[0],szT[1]:szT[1]+szA[1],szT[2]:szT[2]+szA[2]]=A
	s=np.cumsum(B,0)
	c=s[szT[0]:,:,:]-s[:-szT[0],:,:]
	s=np.cumsum(c,1)
	c=s[:,szT[1]:,:]-s[:,:-szT[1],:]
	s=np.cumsum(c,2)
	integralImageA=s[:,:,szT[2]:]-s[:,:,:-szT[2]]

	return integralImageA
		

	
###########################
##### Visualization and Transformation -->7D<-- spotting representation
###########################


#def compress_to_7D(left,right,up,down,static,flicker,los,last_no_features_consider):
def compress_to_7D(*args):
	'''
	This function takes those 7 feature istare.video objects and an argument mentioning the first 'n' arguments to be considered for the compression to a single [:,:,:,n] dim video
	'''
	ret_array=np.zeros([args[0].V.shape[0],args[0].V.shape[1],args[0].V.shape[2],args[-1]],dtype=np.float32)

	for i in range(0,args[-1]):
		ret_array[:,:,:,i]=args[i].V.squeeze()
		
	return ret_array	
	
def normalize(V):
	'''
     Takes arguments of ndarray and normalizes along the 4th dim.
	'''

	Z = V / (V.sum(axis=3))[:,:,:,np.newaxis]
	Z[np.isnan(Z)] = 0
	Z[np.isinf(Z)] = 0

	return Z

def pretty(*args):
    '''
    Takes the argument videos, assumes they are all the same size, and drops them into
    one monster video, row-wise.
    '''

    n  = len(args)
    if type(args[0]) is video.Video:
        sz = np.asarray(args[0].V.shape)
    else:  # assumed it is a numpy.ndarray
        sz = np.asarray(args[0].shape)
    w  = sz[2]
    sz[2] *= n

    A = np.zeros(sz,dtype=np.float32)

    if type(args[0]) is video.Video:
        for i in np.arange(n):
            A[:,:,i*w:(i+1)*w,:] = args[i].V
    else:  #assumed it is a numpy.ndarray
        for i in np.arange(n):
            A[:,:,i*w:(i+1)*w,:] = args[i]

    return video.asvideo(A)

def split(V):
	''' split a N-band image into a 1-band image side-by-side, like pretty
	'''

	sz = np.asarray(V.shape)

	n      = sz[3]
	sz[3]  = 1
	w      = sz[2]
	sz[2] *= n

	A = np.zeros(sz,dtype=np.float32)

	for i in np.arange(n):
		A[:,:,i*w:(i+1)*w,0] = V[:,:,:,i]

	return video.asvideo(A)

def ret_7D_video_objs(V):
	return [(video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]),video.asvideo(V[:,:,:,0]))]

def takeaway(V):
    ''' subtracts all energy from channels static and los
       clamps at 0 at the bottom

      V is an ndarray with 7-bands
    '''

    A = np.zeros(V.shape,dtype=np.float32)

    for i in range(7):
        a = V[:,:,:,i] - V[:,:,:,4] - V[:,:,:,6]
        a[a<0] = 0
        A[:,:,:,i] = a

    return A

