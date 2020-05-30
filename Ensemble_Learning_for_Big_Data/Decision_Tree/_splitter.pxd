
import numpy as np 
cimport numpy as np 

cimport _criterion as cr
cimport _tree as tr

cdef struct SplitRecord:

    tr.SIZE_t feature
    tr.SIZE_t pos          # Split samples array at the given position

    double threshold
    double improvement
    double impurity_left 
    double impurity_right 

cdef class Splitter: 

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples 

    cdef public cr.Criterion criterion 
    cdef public tr.SIZE_t max_features
    cdef public tr.SIZE_t min_samples_leaf
    cdef public double min_weight_leaf

    cdef object random_state
    cdef tr.UINT32_t rand_r_state

    cdef tr.SIZE_t* samples 
    cdef tr.SIZE_t n_samples 
    cdef double weighted_n_samples 
    cdef tr.SIZE_t* features 
    cdef tr.SIZE_t* constant_features
    cdef tr.SIZE_t n_features
    cdef tr.DTYPE_t* feature_values 

    cdef tr.SIZE_t start
    cdef tr.SIZE_t end 

    cdef const tr.DOUBLE_t[:, ::1]y 
    cdef tr.DOUBLE_t* sample_weight

    cdef int init(self, 
                  object X, 
                  const tr.DOUBLE_t[:, ::1] y,
                  tr.DOUBLE_t* sample_weight, 
                  np.ndarray X_idx_sorted=*) except -1

    cdef int node_reset(self, tr.SIZE_t start, tr.SIZE_t end, 
                        double* weight_n_samples) nogil except -1

    cdef int node_split(self,
                        double impurity,
                        SplitRecord* split, 
                        tr.SIZE_t* n_constant_features) nogil except -1 

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil