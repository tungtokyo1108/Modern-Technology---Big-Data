import numpy as np 
cimport numpy as np 

#cimport _tree as tr
cimport _utils as ut 

cdef class Criterion:

    cdef const ut.DOUBLE_t[:, ::1] y
    cdef ut.DOUBLE_t* sample_weight
    cdef ut.SIZE_t* samples
    cdef ut.SIZE_t start
    cdef ut.SIZE_t pos
    cdef ut.SIZE_t end 

    cdef ut.SIZE_t n_outputs 
    cdef ut.SIZE_t n_samples 
    cdef ut.SIZE_t n_node_samples
    cdef double weighted_n_samples
    cdef double weighted_n_node_samples
    cdef double weighted_n_left
    cdef double weighted_n_right 

    cdef double* sum_total 
    cdef double* sum_left 
    cdef double* sum_right

    # Methods 
    cdef int init(self, const ut.DOUBLE_t[:, ::1] y, ut.DOUBLE_t* sample_weight,
                  double weighted_n_samples, ut.SIZE_t* samples, ut.SIZE_t start,
                  ut.SIZE_t end) nogil except -1 
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, ut.SIZE_t new_pos) nogil except -1 
    cdef double node_impurity(self) nogil 
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil 
    cdef double impurity_improvement(self, double impurity) nogil 
    cdef double proxy_impurity_improvement(self) nogil 
    
cdef class ClassificationCriterion(Criterion):
    
    cdef ut.SIZE_t* n_classes
    cdef ut.SIZE_t sum_stride 
    
cdef class RegressionCriterion(Criterion):
    
    cdef double sq_sum_total