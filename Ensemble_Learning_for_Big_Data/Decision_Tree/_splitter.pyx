
cimport _criterion as cr 

from libc.stdlib cimport free
from libc.stdlib cimport qsort 
from libc.string cimport memcpy 
from libc.string cimport memset 

import numpy as np 
cimport numpy as np 
np.import_array()

from scipy.sparse import csc_matrix 

cimport _utils as ut 

cdef double INFINITY = np.inf 

cdef tr.DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef tr.DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, tr.SIZE_t start_pos) nogil:

    self.impurity_left = INFINITY
    self.impurity_right = INFINITY 
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter: 

    def __cinit__(self, cr.Criterion criterion, tr.SIZE_t max_features, 
                  tr.SIZE_t min_samples_leaf, double min_weight_leaf, 
                  object random_state):
        
        self.criterion = criterion
        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL 
        self.sample_weight = NULL 

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
    
    def __dealloc__(self):

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):

        return {}

    def __setstate__(self, d): 
        pass

    cdef int init(self, 
                  object X, 
                  const tr.DOUBLE_t[:, ::1] y,
                  tr.DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        
        self.rand_r_state = self.random_state.randint(0, ut.RAND_R_MAX)
        cdef tr.SIZE_t n_samples = X.shape[0]

        cdef tr.SIZE_t* samples = ut.safe_realloc(&self.samples, n_samples)

        cdef tr.SIZE_t i, j 
        cdef double weight_n_samples = 0.0
        j = 0

        for i in range(n_samples):

            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i 
                j += 1

            if sample_weight != NULL: 
                weight_n_samples += sample_weight[i]
            else:
                weight_n_samples += 1.0 

        self.n_samples = j
        self.weight_n_samples = weight_n_samples

        cdef tr.SIZE_t n_features = X.shape[1]
        cdef tr.SIZE_t* features = ut.safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i 

        self.n_features = n_features
        
        ut.safe_realloc(&self.feature_values, n_samples)
        ut.safe_realloc(&self.constant_features, n_features)

        self.y = y 
        self.sample_weight = sample_weight

        return 0