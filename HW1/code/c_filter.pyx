from __future__ import division

import cython
from cython.parallel import prange
from cython.parallel import parallel
import numpy

cimport numpy

cdef extern from "math.h" nogil:
    float expf(float)
    float fabs(float)


cdef inline float sqr(float x) nogil:
    return x * x


@cython.boundscheck(False)
@cython.wraparound(False)
def c_bilateral(float[:, :] src, int d, s_space, s_color):
    cdef int x, y, i, j, y_r, x_r
    cdef int r = d // 2
    cdef int h = src.shape[0]
    cdef int w = src.shape[1]
    cdef float f, p
    cdef float[:, :] pad_src = numpy.zeros((h + (2 * r), w + (2 * r)), dtype=numpy.float32)
    cdef float wt, sum = 0
    cdef float inv_ss = -0.5 / (s_space * s_space)
    cdef float inv_sc = -0.5 / (s_color * s_color)
    cdef float[:, :] ws = numpy.zeros((d, d), dtype=numpy.float32)
    cdef float[:, :] dst = numpy.zeros((h, w), dtype=numpy.float32)

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

def c_conv2d(float[:, :] image, float[:, :] kernel)
    cdef int i,j,padding_px

    padding_px = (kernel.shape[0] - 1) / 2
    cdef float image_padded = np.empty((image.shape[0] + padding_px, image.shape[1] + padding_px))
    cdef float result = np.empty_like(image)

    image_padded = np.pad(image, padding_px, 'symmetric')
    with nogil, parallel():
        for i in prange(result.shape[0], schedule='guided'):
            for j in range(result.shape[1]):
                result[i, j] = np.vdot(kernel, image_padded[i:i+kernel.shape[0],j:j+kernel.shape[1]])
    return result