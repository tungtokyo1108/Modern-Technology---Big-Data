import numpy as np 
cimport numpy as np 

ctypedef np.npy_float32 DTYPE_t 
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp SIZE_t
ctypedef np.npy_int32 INT32_t
ctypedef np.npy_uint32 UINT32_t 

cdef struct Node:

    SIZE_t left_child
    SIZE_t right_child
    SIZE_t feature
    DOUBLE_t threshold
    DOUBLE_t impurity
    SIZE_t n_node_samples
    DOUBLE_t weighted_n_node_samples