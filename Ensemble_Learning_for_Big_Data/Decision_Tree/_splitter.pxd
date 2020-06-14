
import numpy as np 
cimport numpy as np 

cimport _criterion as cr
from _criterion cimport Criterion
#cimport _tree as tr
cimport _utils as ut 
from _utils cimport DTYPE_t
from _utils cimport DOUBLE_t
from _utils cimport SIZE_t
from _utils cimport INT32_t 
from _utils cimport UINT32_t

cdef struct SplitRecord:

    SIZE_t feature
    SIZE_t pos          # Split samples array at the given position

    double threshold
    double improvement
    double impurity_left 
    double impurity_right 

cdef class Splitter: 

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples 

    cdef public Criterion criterion 
    cdef public SIZE_t max_features
    cdef public SIZE_t min_samples_leaf
    cdef public double min_weight_leaf

    cdef object random_state
    cdef UINT32_t rand_r_state

    cdef SIZE_t* samples 
    cdef SIZE_t n_samples 
    cdef double weighted_n_samples 
    cdef SIZE_t* features 
    cdef SIZE_t* constant_features
    cdef SIZE_t n_features
    cdef DTYPE_t* feature_values 

    cdef SIZE_t start
    cdef SIZE_t end 

    cdef const DOUBLE_t[:, ::1]y 
    cdef DOUBLE_t* sample_weight

    cdef int init(self, 
                  object X, 
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, 
                  np.ndarray X_idx_sorted=*) except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end, 
                        double* weight_n_samples) nogil except -1

    cdef int node_split(self,
                        double impurity,
                        SplitRecord* split, 
                        SIZE_t* n_constant_features) nogil except -1 

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil