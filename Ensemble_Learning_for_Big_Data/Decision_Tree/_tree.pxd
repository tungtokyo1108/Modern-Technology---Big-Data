import numpy as np 
cimport numpy as np 

cimport _splitter as sp

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

cdef class Tree:

    cdef public SIZE_t n_features
    cdef SIZE_t* n_classes
    cdef public SIZE_t n_outputs
    cdef public SIZE_t max_n_classes

    cdef public SIZE_t max_depth
    cdef public SIZE_t node_count #
    cdef public SIZE_t capacity 
    cdef Node* nodes 
    cdef double* value 
    cdef SIZE_t value_stride

    #cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
    #                      SIZE_t feature, double threshold, double impurity,
    #                      SIZE_t n_node_samples, double weighted_n_samples) nogil except -1
    #cdef int _resize(self, SIZE_t capacity) nogil except -1
    #cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1 

    #cdef np.ndarray _get_value_ndarray(self)
    #cdef np.ndarray _get_node_ndarray(self)

    #cdef np.ndarray predict(self, object X)

    #cdef np.ndarray apply(self, object X)
    #cdef np.ndarray _apply_dense(self, object X)
    #cdef np.ndarray _apply_sparse_csr(self, object X)
