



cimport cython
import numpy
cimport numpy
numpy.import_array()  # required in order to use C-API

from libc.math cimport fabs, sqrt#, exp, pow, cos, sin, asin

from numpy.math cimport NAN, isfinite

ctypedef numpy.float64_t DTYPE_F64_t
ctypedef numpy.float32_t DTYPE_F32_t
ctypedef numpy.intp_t ITYPE_t
ctypedef numpy.npy_bool DTYPE_B_t





def calculate_smoothness(x_grads,
                         y_grads,
                         mask,
                         n_rows,
                         n_cols,
                         zero_tol=1e-13,
                         out=None):


    '''
    Description:
    ============
    Function to compare pixel gradients to calculate a "smoothness score" for a 2D image.
    Uses the cosine distance between the current pixel's gradient and its left, right
    bottom, and top neighbor gradients, and sums them for every pixel.

    Note that this code intentionally is skipping the 1-pixel boundary around the edges
    of every input.

    For reference, on a simple 1000x1000 input example this cython runs in 0.014 seconds
    versus a 99.15 second run in pure python (factor 10,000 difference)

    Cosine distance defined as 1.0 - (u.v)/(|u||v|).


    Parameters:
    ============
    x_grads - 32 bit float array shape (n_rows, n_cols)
        values of the gradient in the X direction at every pixel location

    y_grads - 32 bit float array shape (n_rows, n_cols)
        values of the gradient in the Y direction at every pixel location

    mask - bool type (numpy.uint8 usually) array of shape (n_rows, n_cols)
        mask[row,col] = 1 means that pixel is masked and will be skipped in calculations
        use an array of np.zeros((n_rows,n_cols), dtype=np.uint8)) to mask nothing
        Note this code will still skip a 1-pixel outer boundary.

    n_rows - integer, number of columns in x_grads, y_grads, and mask

    n_cols - integer, number of columns in x_grads, y_grads, and mask

    zero_tol - float, compared against for zero gradients distance division

    out - 32 bit float array of shape (n_rows, n_cols)
        memory to write out pixelwise distance values


    Returns:
    ========

    out_sum, 32 bit float containing sum of all gradient distances


    '''

    if out is None:

        return calculate_smoothness_no_out(x_grads,
                                          y_grads,
                                          mask,
                                          n_rows,
                                          n_cols,
                                          zero_tol)

    else:

        return calculate_smoothness_with_out(x_grads,
                                          y_grads,
                                          mask,
                                          n_rows,
                                          n_cols,
                                          zero_tol,
                                          out)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef DTYPE_F64_t calculate_smoothness_no_out(DTYPE_F32_t[:,:] x_grads,
                                      DTYPE_F32_t[:,:] y_grads,
                                      DTYPE_B_t[:,:] mask,
                                      ITYPE_t n_rows,
                                      ITYPE_t n_cols,
                                      DTYPE_F32_t zero_tol):




    cdef DTYPE_F32_t out_sum = 0.0

    cdef DTYPE_F32_t curr_grad_x

    cdef DTYPE_F32_t curr_grad_y

    cdef DTYPE_F32_t compare_grad_x

    cdef DTYPE_F32_t compare_grad_y


    cdef DTYPE_F32_t curr_dist
    cdef DTYPE_F32_t curr_grad_mag
    cdef DTYPE_F32_t compare_grad_mag

    cdef DTYPE_B_t left_masked
    cdef DTYPE_B_t right_masked
    cdef DTYPE_B_t top_masked
    cdef DTYPE_B_t bot_masked

    cdef ITYPE_t row_idx

    cdef ITYPE_t col_idx

    for row_idx in range(1, n_rows-1):

        for col_idx in range(1, n_cols-1):

            if mask[row_idx, col_idx]:

                continue

            left_masked = mask[row_idx, col_idx-1]
            right_masked = mask[row_idx, col_idx+1]
            top_masked = mask[row_idx-1, col_idx]
            bot_masked = mask[row_idx+1, col_idx]



            curr_grad_x = x_grads[row_idx, col_idx]
            curr_grad_y = y_grads[row_idx, col_idx]

            curr_grad_mag = sqrt(curr_grad_x*curr_grad_x + curr_grad_y*curr_grad_y)

            if not left_masked:

                compare_grad_x = x_grads[row_idx, col_idx-1]
                compare_grad_y = y_grads[row_idx, col_idx-1]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist


            if not right_masked:

                compare_grad_x = x_grads[row_idx, col_idx+1]
                compare_grad_y = y_grads[row_idx, col_idx+1]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

            if not top_masked:

                compare_grad_x = x_grads[row_idx-1, col_idx]
                compare_grad_y = y_grads[row_idx-1, col_idx]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist


            if not bot_masked:

                compare_grad_x = x_grads[row_idx+1, col_idx]
                compare_grad_y = y_grads[row_idx+1, col_idx]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

    return out_sum




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef DTYPE_F64_t calculate_smoothness_with_out(DTYPE_F32_t[:,:] x_grads,
                                      DTYPE_F32_t[:,:] y_grads,
                                      DTYPE_B_t[:,:] mask,
                                      ITYPE_t n_rows,
                                      ITYPE_t n_cols,
                                      DTYPE_F32_t zero_tol,
                                      DTYPE_F32_t[:,:] out_arr):

    '''
    Description:
    ============
    Function to compare pixel gradients to calculate a "smoothness score" for a 2D image.
    Uses the cosine distance between the current pixel's gradient and its left, right
    bottom, and top neighbor gradients, and sums them for every pixel.

    Note that this code intentionally is skipping the 1-pixel boundary around the edges
    of every input.

    For reference, on a simple 1000x1000 input example this cython runs in 0.014 seconds
    versus a 99.15 second run in pure python (factor 10,000 difference)

    Cosine distance defined as 1.0 - (u.v)/(|u||v|).


    Parameters:
    ============
    x_grads - 32 bit float array shape (n_rows, n_cols)
        values of the gradient in the X direction at every pixel location

    y_grads - 32 bit float array shape (n_rows, n_cols)
        values of the gradient in the Y direction at every pixel location

    mask - bool type (numpy.uint8 usually) array of shape (n_rows, n_cols)
        mask[row,col] = 1 means that pixel is masked and will be skipped in calculations
        use an array of np.zeros((n_rows,n_cols), dtype=np.uint8)) to mask nothing
        Note this code will still skip a 1-pixel outer boundary.

    n_rows - integer, number of columns in x_grads, y_grads, and mask

    n_cols - integer, number of columns in x_grads, y_grads, and mask

    zero_tol - float, compared against for zero gradients distance division


    Returns:
    ========

    out_sum, 32 bit float containing sum of all gradient distances


    '''


    cdef DTYPE_F32_t out_sum = 0.0

    cdef DTYPE_F32_t curr_grad_x

    cdef DTYPE_F32_t curr_grad_y

    cdef DTYPE_F32_t compare_grad_x

    cdef DTYPE_F32_t compare_grad_y


    cdef DTYPE_F32_t curr_dist
    cdef DTYPE_F32_t curr_grad_mag
    cdef DTYPE_F32_t compare_grad_mag

    cdef DTYPE_B_t left_masked
    cdef DTYPE_B_t right_masked
    cdef DTYPE_B_t top_masked
    cdef DTYPE_B_t bot_masked

    cdef ITYPE_t row_idx

    cdef ITYPE_t col_idx

    for row_idx in range(1, n_rows-1):

        for col_idx in range(1, n_cols-1):

            if mask[row_idx, col_idx]:

                continue

            left_masked = mask[row_idx, col_idx-1]
            right_masked = mask[row_idx, col_idx+1]
            top_masked = mask[row_idx-1, col_idx]
            bot_masked = mask[row_idx+1, col_idx]



            curr_grad_x = x_grads[row_idx, col_idx]
            curr_grad_y = y_grads[row_idx, col_idx]

            curr_grad_mag = sqrt(curr_grad_x*curr_grad_x + curr_grad_y*curr_grad_y)

            if not left_masked:

                compare_grad_x = x_grads[row_idx, col_idx-1]
                compare_grad_y = y_grads[row_idx, col_idx-1]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

                    out_arr[row_idx, col_idx] = curr_dist


            if not right_masked:

                compare_grad_x = x_grads[row_idx, col_idx+1]
                compare_grad_y = y_grads[row_idx, col_idx+1]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

                    out_arr[row_idx, col_idx] = curr_dist

            if not top_masked:

                compare_grad_x = x_grads[row_idx-1, col_idx]
                compare_grad_y = y_grads[row_idx-1, col_idx]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

                    out_arr[row_idx, col_idx] = curr_dist


            if not bot_masked:

                compare_grad_x = x_grads[row_idx+1, col_idx]
                compare_grad_y = y_grads[row_idx+1, col_idx]

                compare_grad_mag = sqrt(compare_grad_x*compare_grad_x + compare_grad_y*compare_grad_y)

                if curr_grad_mag*compare_grad_mag < zero_tol:
                    pass
                else:

                    curr_dist = 1.0 - (curr_grad_x*compare_grad_x + curr_grad_y*compare_grad_y)/(curr_grad_mag*compare_grad_mag)

                    out_sum += curr_dist

                    out_arr[row_idx, col_idx] = curr_dist

    return out_sum
