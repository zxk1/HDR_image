import numpy as np
import math
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
    DBL_MIN = sys.float_info.min
    LDRIMG = np.empty_like(HDRIMG)
    s = 0.9
    #for i in range(HDRIMG.shape[0]):
    #    for j in range(HDRIMG.shape[1]):
    for i in range(5):
        for j in range(5):
            X_0 = max(HDRIMG[i][j])
            LOG_X_0 = math.log(X_0, 2)
            pixel = HDRIMG[i][j]
            #print(pixel)
            # Fix log value
            LOG_min = math.pow(2.0, DBL_MIN)
            LOG_pixel = np.empty(3)
            for k in range(3):
                if(pixel[k] < LOG_min):
                    LOG_pixel[k] = DBL_MIN
                else:
                    LOG_pixel[k] = s * (math.log(pixel[k], 2) - LOG_X_0) + LOG_X_0
            #print(LOG_pixel)
            # Restore R_hat, G_hat, B_hat value
            np.power([2.0, 2.0, 2.0], LOG_pixel, LDRIMG[i][j])
            #R_hat = math.pow(2.0, LOG_R)
            #G_hat = math.pow(2.0, LOG_G)
            #B_hat = math.pow(2.0, LOG_B)
            #LDRIMG[i][j] = [R_hat, G_hat, B_hat]
            
            # Do gamma correction by taking X' = X_hat ^ (1/2.2)
            exp = np.empty(3)
            exp.fill(1.0/2.2)
            np.power(LDRIMG[i][j], exp, LDRIMG[i][j])
            #LDRIMG[i][j][0] = math.pow(R_hat, 1.0/2.2)
            #LDRIMG[i][j][1] = math.pow(G_hat, 1.0/2.2)
            #LDRIMG[i][j][2] = math.pow(B_hat, 1.0/2.2)             
            # Fix out of range pixels
            for k in range(3):
                if (LDRIMG[i][j][k] > 1.0):
                    LDRIMG[i][j][k] = 1.0
                if (LDRIMG[i][j][k] < 0.0):
                    LDRIMG[i][j][k] = 0.0                                   
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



