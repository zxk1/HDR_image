import numpy as np
import math

def gen_gaussian_kernel(kernel_len, sigma = 100):
    """ Return a 2D Gaussian filter kernel
            Args:
                kernel_len (int): the size of kernel
                sigma (int): 
            Return:
                kernel (np.ndarray): Gaussian Kernel for convolution
    """
    # Type Checking
    assert isinstance(kernel_len, int)
    assert isinstance(sigma, int)
    # Allocate kernel
    kernel = np.empty((kernel_len,kernel_len))

    ax = np.arange(-kernel_len // 2 + 1., kernel_len // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)

def gen_bilateral_kernel(kernel_len, sigma):
    """ Return a 2D Bilateral filter kernel
            Args:
                kernel_len (int): the size of kernel
                sigma (int): 
            Return:
                kernel (np.ndarray): Bilateral Kernel for convolution
    """
    # Type Checking
    assert isinstance(kernel_len, int)
    assert isinstance(sigma, float)
    # Allocate kernel
    kernel = np.empty((kernel_len,kernel_len))

def conv2d(image, kernel):
    """ 2D Convolution
            Args:
                image: 2D image(before padding)
                kernel: convolution
            Return:
                result (np.ndarray): Convolution result
    """
    # Assert the inputs are 2D array
    assert(len(image.shape) == 2)
    assert(len(image.shape) == 2)
    padding_px = (kernel.shape[0] - 1) / 2
    image_padded = np.empty((image.shape[0] + padding_px, image.shape[1] + padding_px))
    result = np.empty_like(image) 
    
    image_padded = np.pad(image, padding_px, 'symmetric')
    
    # Real computation
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.vdot(kernel, image_padded[i:i+kernel.shape[0],j:j+kernel.shape[1]])
    return result