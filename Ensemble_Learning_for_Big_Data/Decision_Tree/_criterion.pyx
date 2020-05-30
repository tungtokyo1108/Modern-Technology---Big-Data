from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs 

import numpy as np 
cimport numpy as np 
np.import_array()

cimport _utils as ut 

cdef class Criterion:

    def __dealloc__(self):

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return()

    def __setstate__(self, d):
        pass 

    cdef int init(self, const ut.DOUBLE_t[:, ::1] y, ut.DOUBLE_t* sample_weight,
                  double weighted_n_samples, ut.SIZE_t* samples, ut.SIZE_t start,
                  ut.SIZE_t end) nogil except -1:
        pass
    
    cdef int reset(self) nogil except -1:
        pass 

    cdef int reverse_reset(self) nogil except -1:
        pass 

    cdef int update(self, ut.SIZE_t new_pos) nogil except -1:
        pass 

    cdef double node_impurity(self) nogil:
        pass 

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass 

    cdef void node_value(self, double* dest) nogil:
        pass 

    cdef double proxy_impurity_improvement(self) nogil:

        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left) 

    cdef double impurity_improvement(self, double impurity) nogil:

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) * 
                (impurity - (self.weighted_n_right / 
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left / 
                             self.weighted_n_node_samples * impurity_left)))

cdef class ClassificationCriterion(Criterion):

    def __cinit__(self, ut.SIZE_t n_outputs,
                  np.ndarray[ut.SIZE_t, ndim=1] n_classes):
        
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0 
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL

        ut.safe_realloc(&self.n_classes, n_outputs)

        cdef ut.SIZE_t k = 0
        cdef ut.SIZE_t sum_stride = 0

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]
            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]

        self.sum_stride = sum_stride

        cdef ut.SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))

        if (self.sum_total == NULL or 
            self.sum_left == NULL or 
            self.sum_right == NULL):
            raise MemoryError()
    
    def __dealloc__(self):
        free(self.n_classes)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs,
                 ut.sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                 self.__getstate__())

    cdef int init(self, const ut.DOUBLE_t[:, ::1] y, ut.DOUBLE_t* sample_weight,
                  double weighted_n_samples, ut.SIZE_t* samples, ut.SIZE_t start,
                  ut.SIZE_t end) nogil except -1:
        
        self.y = y 
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total

        cdef ut.SIZE_t i 
        cdef ut.SIZE_t p 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c 
        cdef ut.DOUBLE_t w = 1.0
        cdef ut.SIZE_t offset = 0

        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                c = <ut.SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w 
            
            self.weighted_n_node_samples += w 

        self.reset()

        return 0 
    
    cdef int reset(self) nogil except -1:

        self.pos = self.start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef ut.SIZE_t k 

        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride

        return 0 

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef ut.SIZE_t k

        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))

            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int update(self, ut.SIZE_t new_pos) nogil except -1:
        
        cdef ut.SIZE_t pos = self.pos
        cdef ut.SIZE_t end = self.end 

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef ut.SIZE_t* samples = self.samples 
        cdef ut.DOUBLE_t* sample_weight = self.sample_weight

        cdef ut.SIZE_t i 
        cdef ut.SIZE_t p 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c
        cdef ut.SIZE_t label_index 
        cdef ut.DOUBLE_t w = 1.0

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <ut.SIZE_t> self.y[i, k]
                    sum_left[label_index] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <ut.SIZE_t> self.y[i, k]
                    sum_left[label_index] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]

            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left, 
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:

        cdef double* sum_total = self.sum_total
        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef ut.SIZE_t k 

        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride

cdef class Entropy(ClassificationCriterion):

    cdef double node_impurity(self) nogil:

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c 

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * ut.log(count_k)
            
            sum_total += self.sum_stride

        return entropy / self.n_outputs 

    cdef void children_impurity(self, double* impurity_left, 
                                double* impurity_right) nogil:

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c 

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * ut.log(count_k)
                
                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * ut.log(count_k)

            sum_left += self.sum_stride
            sum_right += self.sum_stride

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs

cdef class Gini(ClassificationCriterion):

    cdef double node_impurity(self) nogil:

        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count 
        cdef double count_k 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c 

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k
            
            gini += 1.0 - sq_count / (self.weighted_n_node_samples * 
                                      self.weighted_n_node_samples)

            sum_total += self.sum_stride
        
        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left, 
                                double* impurity_right) nogil:
        
        cdef ut.SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left 
        cdef double sq_count_right
        cdef double count_k 
        cdef ut.SIZE_t k 
        cdef ut.SIZE_t c 

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k

                count_k = sum_right[c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left/(self.weighted_n_left * 
                                              self.weighted_n_left)
            gini_right += 1.0 - sq_count_right/(self.weighted_n_right * 
                                                self.weighted_n_right)

            sum_left += self.sum_stride
            sum_right += self.sum_stride
        
        impurity_left[0] = gini_left/self.n_outputs
        impurity_right[0] = gini_right/self.n_outputs

