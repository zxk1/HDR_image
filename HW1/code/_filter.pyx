from __future__ import division

import cython
from cython.parallel import prange
from cython.parallel import parallel
import numpy

cimport numpy


ctypedef unsigned char uint8_t


cdef extern from "math.h" nogil:
    double expf(double)
    double fabs(double)


cdef inline double sqr(double x) nogil:
    return x * x


@cython.boundscheck(False)
@cython.wraparound(False)
def _l_bilateral_solver(double[:, :] src, int d, s_space, s_color):
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

    pad_src = numpy.pad(src, ((r, r), (r, r)), 'edge')

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