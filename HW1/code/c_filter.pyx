from __future__ import division

import cython
from cython.parallel import prange
from cython.parallel import parallel
import numpy

cimport numpy

cdef extern from "math.h" nogil:
    double expf(double)
    double fabs(double)


cdef inline double sqr(double x) nogil:
    return x * x


@cython.boundscheck(False)
@cython.wraparound(False)
def c_bilateral(double[:, :] src, int d, s_space, s_color):
    cdef int x, y, i, j, y_r, x_r
    cdef int r = d // 2
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef double f, p
    cdef double[:, :] pad_src = numpy.zeros((h + (2 * r), w + (2 * r)), dtype=numpy.float64)
    cdef double wt, sum = 0
    cdef double inv_ss = -0.5 / (s_space * s_space)
    cdef double inv_sc = -0.5 / (s_color * s_color)
    cdef double[:, :] ws = numpy.zeros((d, d), dtype=numpy.float64)
    cdef double[:, :] dst = numpy.zeros((h, w), dtype=numpy.float64)

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            ws[i+r, j+r] = (sqr(i) + sqr(j)) * inv_ss

    pad_src = numpy.pad(src, ((r, r), (r, r)), 'symmetric')

    with nogil, parallel():
        for y in prange(r, h + r, schedule='guided'):
            y_r = y - r
            for x in range(r, w + r):
                sum = 0
                x_r = x - r
                p = pad_src[y, x]
                for i in range(d):
                    for j in range(d):
                        f = pad_src[y_r + i, x_r + j]
                        wt = expf(ws[i, j] + sqr(p - f) * inv_sc)
                        dst[y_r, x_r] += f * wt
                        sum += wt
                dst[y_r, x_r] /= sum

    return numpy.asarray(dst)

def c_conv2d(double[:, :] image, double[:, :] kernel):
    cdef int i,j,padding_px, rh, rw
    cdef int image_h = image.shape[0]
    cdef int image_w = image.shape[1]
    cdef int kernel_size = kernel.shape[0]
    padding_px = (kernel.shape[0] - 1) // 2
    cdef double[:, :] image_padded = numpy.empty((image.shape[0] + padding_px, image.shape[1] + padding_px),dtype=numpy.float64)
    cdef double[:, :] result = numpy.empty_like(image)

    image_padded = numpy.pad(image, padding_px, 'symmetric')
    with nogil, parallel():
        for i in prange(image_h, schedule='guided'):
            for j in range(image_w):
                for rh in range(i,i+kernel_size):
                    for rw in range(j,j+kernel_size):
                        result[i, j] += image_padded[rh,rw] * kernel[rh-i,rw-j]
    return numpy.asarray(result)
