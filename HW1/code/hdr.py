import numpy as np
import sys

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
    #if WB == 'True':
    #  HDRIMG = white_balance(HDRIMG,x_range=(457,481),y_range=(400,412))
    LDRIMG = np.empty_like(HDRIMG)
    X = np.empty((HDRIMG.shape[0], HDRIMG.shape[1]))
    LOG_X = np.empty_like(X)
    LOG_X_0 = np.empty(1)
    LOG_X_hat = np.empty_like(X)
    s = 0.9
    gamma = 2.2
    
    for ch in range(HDRIMG.shape[2]):
        # Gamma compression
        X = HDRIMG[:,:,ch]
        X_0 = np.max(X)
        np.log2(X_0,LOG_X_0)
        np.log2(X, LOG_X)
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
    LDRIMG = np.empty_like(HDRIMG)
    X = np.empty((HDRIMG.shape[0], HDRIMG.shape[1]))
    Color_ratio = np.empty_like(X)
    L = np.empty_like(X)
    for ch in range(HDRIMG.shape[2]):
        X = HDRIMG[:,:,ch]
        I = np.average(X)
        np.divide(X, I, Color_ratio)
        np.log2(I, L)

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
                IMG_wb (np.ndarray): The processed corresponding white balance image of HDRIMG
            Todo:
                - implement white balance here
    """



    return IMG_wb



