import numpy as np 
cimport numpy as np 

#cimport _tree as tr
cimport _utils as ut 
from _utils cimport DTYPE_t
from _utils cimport DOUBLE_t
from _utils cimport SIZE_t
from _utils cimport INT32_t 
from _utils cimport UINT32_t

cdef class Criterion:

    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight
    cdef SIZE_t* samples
    cdef SIZE_t start
    cdef SIZE_t pos
    cdef SIZE_t end 

    cdef SIZE_t n_outputs 
    cdef SIZE_t n_samples 
    cdef SIZE_t n_node_samples
    cdef double weighted_n_samples
    cdef double weighted_n_node_samples
    cdef double weighted_n_left
    cdef double weighted_n_right 

    cdef double* sum_total 
    cdef double* sum_left 
    cdef double* sum_right

    # Methods 
    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1 
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1 
    cdef double node_impurity(self) nogil 
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil 
    cdef double impurity_improvement(self, double impurity) nogil 
    cdef double proxy_impurity_improvement(self) nogil 
    
cdef class ClassificationCriterion(Criterion):
    
    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride 
    
cdef class RegressionCriterion(Criterion):
    
    cdef double sq_sum_total