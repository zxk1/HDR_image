import numpy as np
import sys
import filter_util as util
from scipy import signal
import c_filter

def global_tone_mapping(HDRIMG, WB = 'True'):
    """ Perform Global tone mapping on HDRIMG
            Note:
                1.Please remember clip the range of intensity to [0, 1] and convert type of LDRIMG to "uint8" with range [0, 255] for display. You can use the following code snippet.
                  >> LDRIMG = np.round(LDRIMG*255).astype("uint8")
                2.Make sure the LDRIMG's range is in 0-255(uint8). If the value is larger than upperbound, modify it to upperbound. If the value is smaller than lowerbound, modify it to lowerbound.
                3.Beware on the log results of 0(small number). Replace the small numbers(less then DBL_MIN) with DBL_MIN. The following code shows how to get DBL_MIN in python.
                  >> import sys
                  >> sys.float_info.min # get DBL_MIN
            Args:
                HDRIMG (np.ndarray): The input image to process
                WB (bool): The flag to indicate whether to perform white balance
            Returns:
                LDRIMG (np.ndarray): The processed corresponding low dynamic range image of HDRIMG
            Todo:
                - implement global tone mapping (with white balance) here
    """
    if WB == 'True':
        #HDRIMG = white_balance(HDRIMG,x_range=(457,481),y_range=(400,412))
        #HDRIMG = white_balance(HDRIMG,x_range=(152,162),y_range=(134,155))
        #HDRIMG = white_balance(HDRIMG,x_range=(1858,2334),y_range=(3778,3848))
        HDRIMG = white_balance(HDRIMG,x_range=(520,1068),y_range=(1170,1625))

    LDRIMG = np.empty_like(HDRIMG)
    X = np.empty((HDRIMG.shape[0], HDRIMG.shape[1]))
    LOG_X = np.empty_like(X)
    LOG_X_0 = np.empty(1)
    LOG_X_hat = np.empty_like(X)
    s = 0.99
    gamma = 2.2
    DBL_MIN = sys.float_info.min
    for ch in range(HDRIMG.shape[2]):
        # Gamma compression
        X = HDRIMG[:,:,ch]
        X_0 = np.max(X)
        np.log2(X_0 + DBL_MIN, LOG_X_0)
        np.log2(X + DBL_MIN, LOG_X)
        LOG_X_hat = s * (LOG_X - LOG_X_0) + LOG_X_0
        # Restore log(X_hat) to X_hat, and store them to LDRIMG
        np.power(2.0, LOG_X_hat, LDRIMG[:,:,ch])
        # Gamma Correction
        np.power(LDRIMG[:,:,ch], (1.0/gamma), LDRIMG[:,:,ch])
    # Fix out of range pixels
    LDRIMG[LDRIMG < 0.0] = 0.0
    LDRIMG[LDRIMG > 1.0] = 1.0
    LDRIMG = np.round(LDRIMG*255).astype("uint8")
    return LDRIMG

def local_tone_mapping(HDRIMG, Filter, window_size, sigma_s, sigma_r):
    """ Perform Local tone mapping on HDRIMG
            Note:
                1.Please remember clip the range of intensity to [0, 1] and convert type of LDRIMG to "uint8" with range [0, 255] for display. You can use the following code snippet.
                  >> LDRIMG = np.round(LDRIMG*255).astype("uint8")
                2.Make sure the LDRIMG's range is in 0-255(uint8). If the value is larger than upperbound, modify it to upperbound. If the value is smaller than lowerbound, modify it to lowerbound.
            Args:
                HDRIMG (np.ndarray): The input image to process
                Filter (function): 'Filter' is a function that is used for filter operation to get base layer. It can be gaussian or bilateral. 
                                   It's input is log of the intensity and filter's parameters. And the output is the base layer.
                window size(diameter) (int): default 35
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LDRIMG (np.ndarray): The processed corresponding low dynamic range image of HDRIMG
            Todo:
                - implement local tone mapping here
    """
    scale = 5
    gamma = 2.2
    DBL_MIN = sys.float_info.min
    LDRIMG = np.empty_like(HDRIMG)
    HDRIMG[HDRIMG == 0.0] = sys.float_info.min
    X = np.empty((HDRIMG.shape[0], HDRIMG.shape[1]),dtype = np.float64)
    I = np.empty_like(X,dtype = np.float64)
    Color_ratio = np.empty_like(X,dtype = np.float64)
    L = np.empty_like(X,dtype = np.float64)
    LB = np.empty_like(X,dtype = np.float64)
    LD = np.empty_like(X,dtype = np.float64)
    LB_prime = np.empty_like(X,dtype = np.float64)
    I_prime = np.empty_like(X,dtype = np.float64)
    
    # Get Color intensity
    I = np.average(HDRIMG, axis=2)
    # Take log of intensity
    np.log2(I+DBL_MIN, L)
    
    # Apply filter to get base layer
    if Filter == gaussian :
        # Call gaussian filter
        LB = gaussian(L, window_size, sigma_s, sigma_r)
    elif Filter == bilateral :
        # Call bilateral filter
        LB = c_filter.c_bilateral(L, window_size, sigma_s, sigma_r)
    else :
        sys.exit("Undefined Filter")
    # Get detail layer
    np.subtract(L, LB, LD)
    # Find the range of base layer
    L_min = np.amin(LB)
    L_max = np.amax(LB)
    # Adjust contrast on base layer
    LB_prime = (np.subtract(LB, L_max)) * np.float64(scale / (L_max - L_min)) 
    # Reconstruct intensity I'
    I_prime = np.power(2, np.add(LB_prime, LD))
    for ch in range(HDRIMG.shape[2]):
        X = HDRIMG[:,:,ch]
        # Get color ratio
        np.divide(X, I, Color_ratio)
        # Reconstruct R,G, and B 
        np.multiply(Color_ratio, I_prime, LDRIMG[:,:,ch])
        # Apply gamma correction
        np.power(LDRIMG[:,:,ch], (1.0/gamma), LDRIMG[:,:,ch])
    # Fix out of range pixels and convert data type
    LDRIMG[LDRIMG < 0.0] = 0.0
    LDRIMG[LDRIMG > 1.0] = 1.0
    LDRIMG = np.round(LDRIMG*255).astype("uint8")
    
    return LDRIMG

def gaussian(L,window_size,sigma_s,sigma_r):
    """ Perform gaussian filter 
            Notes:
                Please use "symmetric padding" for image padding
            Args:
                L (np.ndarray): Log of the intensity 
                window size(diameter) (int): default 35
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LB (np.ndarray): The base layer
            Todo:
                - implement gaussian filter for local tone mapping
    """
    # Declare variables
    LB = np.empty_like(L, dtype = np.float64)
    kernel = np.empty((window_size, window_size),dtype=np.float64)
    # Generate Gaussian kernel
    kernel = util.gen_gaussian_kernel(window_size, sigma_s)
    # Convolution, use Cython implementation to speed up
    #LB = util.conv2d(L, kernel)
    LB = c_filter.c_conv2d(L, kernel)

    return LB

def bilateral(L,window_size,sigma_s,sigma_r):
    """ Perform bilateral filter 
            Notes:
                Please use "symmetric padding" for image padding   
            Args:
                L (np.ndarray): Log of the intensity 
                window size(diameter) (int): default 35
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LB (np.ndarray): The base layer
            Todo:
                - implement bilateral filter for local tone mapping
    """

    return LB

def white_balance(IMG,x_range,y_range):
    """ Perform white balance 
            Args:
                IMG (np.ndarray): The input image to process
                x_range (tuple): The rectangular range in x direction
                y_range (tuple): The rectangular range in y direction
            Returns:
                IMG (np.ndarray): The processed corresponding white balance image of HDRIMG
            Todo:
                - implement white balance here
    """
    sample = np.empty((x_range[1]-x_range[0] + 1,y_range[1]-y_range[0]+1),dtype = np.float64)
    layer = np.empty((IMG.shape[0],IMG.shape[1]))
    color_avg = np.zeros(3)

    # Sampling from given region 
    for ch in range(IMG.shape[2]):
        layer = IMG[:,:,ch] 
        sample = layer[x_range[0]:x_range[1]+1, y_range[0]:y_range[1]+1]
        color_avg[ch] = np.mean(sample, axis=(0,1))
    green_ratio = color_avg[0] / color_avg[1]
    blue_ratio = color_avg[0] / color_avg[2]
    # Adjust G,B channel
    np.multiply(IMG[:,:,1], green_ratio, IMG[:,:,1])
    np.multiply(IMG[:,:,2], blue_ratio, IMG[:,:,2])

    return IMG
