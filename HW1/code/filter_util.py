import numpy as np

def gen_gaussian_kernel(kernel_len, sigma = 100):
    """ Return a 2D Gaussian kernel array
            Args:
                kernel_len (int): the size of kernel
                sigma (int): 
            Return:
                Kernel (np.ndarray):
    """
    # Type Checking
    assert isinstance(kernel_len, int)
    assert isinstance(sigma, int)

    ax = np.arange(-kernel_len // 2 + 1., kernel_len // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)