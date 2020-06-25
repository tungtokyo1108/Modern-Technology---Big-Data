
cimport _criterion as cr 
from _criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort 
from libc.string cimport memcpy 
from libc.string cimport memset 

import numpy as np 
cimport numpy as np 
np.import_array()

from scipy.sparse import csc_matrix 

cimport _utils as ut 
from _utils cimport log
from _utils cimport safe_realloc
from _utils cimport sizet_ptr_to_ndarray
from _utils cimport rand_int
from _utils cimport rand_uniform
from _utils cimport RAND_R_MAX

cdef double INFINITY = np.inf 

cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:

    self.impurity_left = INFINITY
    self.impurity_right = INFINITY 
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter: 

    def __cinit__(self, Criterion criterion, SIZE_t max_features, 
                  SIZE_t min_samples_leaf, double min_weight_leaf, 
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
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  np.ndarray X_idx_sorted=None) except -1:
        
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j 
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):

            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i 
                j += 1

            if sample_weight != NULL: 
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0 

        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i 

        self.n_features = n_features
        
        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = y 
        self.sample_weight = sample_weight

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end, 
                        double* weighted_n_node_samples) nogil except -1:

        self.start = start
        self.end = end

        self.criterion.init(self.y, 
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples, 
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0 

    cdef int node_split(self,
                        double impurity,
                        SplitRecord* split, 
                        SIZE_t* n_constant_features) nogil except -1:
        pass

    cdef void node_value(self, double* dest) nogil:

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:

        return self.criterion.node_impurity()

cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
        return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)

cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]

cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
   
    cdef DTYPE_t a = Xf[0], b = Xf[n/2], c = Xf[n-1]
    if a < b:
        if b < c:
            return b 
        elif a < c:
            return c 
        else: 
            return a 
    elif b < c:
        if a < c:
            return a 
        else:
            return c 
    else: 
        return b 

cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
    
    cdef SIZE_t child, maxind, root 

    root = start 
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child 
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1 

        if maxind == root:
            break
        else: 
            swap(Xf, samples, root, maxind)
            root = maxind

cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1

cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef class BaseDenseSplitter(Splitter):
    
    cdef const DTYPE_t[:, :] X 

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask 

    def __cinit__(self, Criterion criterion, SIZE_t max_features, 
                  SIZE_t min_samples_leaf, double min_weight_leaf, object random_state):
        
        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0 
        self.sample_mask = NULL 

    cdef int init(self, 
                  object X, 
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight, 
                  np.ndarray X_idx_sorted = None) except -1: 

        Splitter.init(self, X, y, sample_weight)

        self.X = X
        return 0 

cdef class BestSplitter(BaseDenseSplitter):

    def __reduce__(self):
        
        return (BaseDenseSplitter, (self.criterion,
                                    self.max_features,
                                    self.min_samples_leaf,
                                    self.min_weight_leaf,
                                    self.random_state), self.__getstate__())


    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p 
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i 
        cdef SIZE_t j 

        cdef SIZE_t n_visited_features = 0
        cdef SIZE_t n_found_constants = 0
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        """
        Sample up to max_features without replacement using a Fisher-Yates based algorithm
        Using the local variables f_i and f_j to compute a permutation of the features array 
        """

        while(f_i > n_total_constants and 
                (n_visited_features < max_features or 
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            
            n_visited_features += 1

            # Draw a feature at random 
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:

                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1
            
            else:

                f_j += n_found_constants 
                current.feature = features[f_j]

                # Sort samples along that feature 
                for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]

                sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and 
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1
                        
                        p += 1

                        if p < end:
                            current.pos = p 

                            # Reject if min_samples_leaf is not guaranteed 
                            if (((current.pos - start) < min_samples_leaf) or 
                                ((end - current.pos) < min_samples_leaf)):
                                continue
                            
                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not statisfied 
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or 
                                 (self.criterion.weighted_n_right < min_weight_leaf)):
                                 continue
                            
                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                                if ((current.threshold == Xf[p]) or 
                                    (current.threshold == INFINITY) or 
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p - 1]
                                
                                best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end 
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1
                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
            
            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        memcpy(constant_features + n_known_constants, 
                features + n_known_constants,
                sizeof(SIZE_t) * n_found_constants)
        
        split[0] = best 
        n_constant_features[0] = n_total_constants
        
        return 0

cdef class RandomSplitter(BaseDenseSplitter):

    def __reduce__(self):

        return (RandomSplitter, (self.criterion,
                                 self.max_features, 
                                 self.min_samples_leaf,
                                 self.min_weight_leaf,
                                 self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split, 
                        SIZE_t* n_constant_features) nogil except -1:
        
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t partition_end
        cdef SIZE_t feature_stride
        cdef SIZE_t n_found_constants = 0
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value

        _init_split(&best, end)

        while (f_i > n_total_constants and 
                (n_visited_features < max_features or 
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, 
                                random_state)
            
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1

            else:
                f_j += n_found_constants
                current.feature = features[f_j]

                # Find min, max 
                min_feature_value = self.X[samples[start], current.feature]
                max_feature_value = min_feature_value
                Xf[start] = min_feature_value

                for p in range(start + 1, end):
                    current_feature_value = self.X[samples[p], current.feature]
                    Xf[p] = current_feature_value

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], current.feature
                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    current.threshold = rand_uniform(min_feature_value,
                                                        max_feature_value,
                                                        random_state)
                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    p, partition_end = start, end 
                    while p < partition_end:
                        if Xf[p] <= current.threshold:
                            p += 1
                        else:
                            partition_end -= 1 
                            Xf[p], Xf[partition_end] = Xf[partition_end], Xf[p]
                            samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    
                    current.pos = partition_end

                    if (((current.pos - start) < min_samples_leaf) or 
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        best = current
        
        if best.pos < end:
            if current.feature != best.feature:
                p, partition_end = start, end 

                while p < partition_end:
                    if self.X[samples[p], best.feature] <= best.threshold:
                        p += 1
                    else:
                        partition_end -= 1
                        samples[p], samples[partition_end] = samples[partition_end], samples[p]

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left, &best.impurity_right)

        memcpy(features, constant_features, sizeof(ut.SIZE_t) * n_known_constants)
        memcpy(constant_features + n_known_constants, 
                features + n_known_constants,
                sizeof(ut.SIZE_t) * n_found_constants)

        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0

