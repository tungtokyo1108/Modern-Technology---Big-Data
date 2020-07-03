from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np 
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

#from ._utils cimport Stack
#from ._utils cimport StackRecord
#from ._utils cimport PriorityHeap
#from ._utils cimport PriorityHeapRecord
#from ._utils cimport safe_realloc
#from ._utils cimport sizet_ptr_to_ndarray
cimport _utils as ut

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims, 
                                np.npy_intp* strides, 
                                void* data, int flags, object obj)

from numpy import float32 as DTYPE 
from numpy import float64 as DOUBLE 

cdef double INFINITY = np.inf 
cdef double EPSILON = np.finfo('double').eps 

cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0 
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1 
TREE_UNDEFINED = -2 
cdef SIZE_t _TREE_LEAF = TREE_LEAF 
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10 

NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
                'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offset': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

cdef class DepthFirstTreeBuilder(TreeBuilder):

    def __cinit__(self, sp.Splitter splitter, SIZE_t min_samples_split, 
                  SIZE_t min_samples_leaf, double min_weight_leaf, 
                  SIZE_t max_depth, double min_impurity_decrease, 
                  double min_impurity_split):
        
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight = None, 
                np.ndarray X_idx_sorted = None):
        
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data
        
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047 

        tree._resize(init_capacity)

        cdef sp.Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start 
        cdef SIZE_t end 
        cdef SIZE_t depth 
        cdef SIZE_t parent
        cdef bint is_left 
        cdef SIZE_t n_node_samples = splitter.n_samples 
        cdef double weighted_n_samples = splitter.weighted_n_samples 
        cdef double weighted_n_node_samples
        cdef sp.SplitRecord split 
        cdef SIZE_t node_id 

        cdef double impurity = INFINITY 
        cdef SIZE_t n_constant_features 
        cdef bint is_leaf 
        cdef bint first = 1 
        cdef SIZE_t max_depth_seen = -1 
        cdef int rc = 0

        cdef ut.Stack stack = ut.Stack(INITIAL_STACK_SIZE)
        cdef ut.StackRecord stack_record

        with nogil:

            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                with gil:
                    raise MemoryError()

            while not stack.is_empty():

                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start 
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or 
                           n_node_samples < min_samples_split or 
                           n_node_samples < 2 * min_samples_leaf or 
                           weighted_n_node_samples < 2 * min_weight_leaf)
                
                if first: 

                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or 
                           (impurity <= min_impurity_split))

                if not is_leaf: 
                    
                    splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = (is_leaf or split.pos >= end or 
                               (split.improvement + EPSILON < min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature, 
                                         split.threshold, impurity, n_node_samples, weighted_n_node_samples)
                
                if node_id == SIZE_MAX:
                    rc = -1 
                    break

                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:

                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    rc = stack.push(start, split.pos, depth + 1, node_id, 1, 
                                    split.impurity_left, n_constant_features)

                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
    
        if rc == -1:
            raise MemoryError()



cdef class Tree:

    property n_classes:
        def __get__(self):
            return ut.sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes, int n_outputs):

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        ut.safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k 
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):

        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       ut.sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):

        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]

        node_ndarray = d["nodes"]
        value_ndarray = d["values"]

        value_shape = (node_ndarray.shape[0], self.n_outputs, 
                        self.max_n_classes)

        self.capacity = node_ndarray.shape[0]

        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data, 
                        self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                        self.capacity * self.value_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity = SIZE_MAX) nogil except -1:

        if capacity == self.capacity and self.nodes != NULL:
            return 0 

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3 
            else:
                capacity = 2 * self.capacity

        ut.safe_realloc(&self.nodes, capacity)
        ut.safe_realloc(&self.value, capacity * self.value_stride)

        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride * sizeof(double))

        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity 

        return 0 

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                            SIZE_t feature, double threshold, double impurity,
                            SIZE_t n_node_samples, 
                            double weighted_n_node_samples) nogil except -1:
        
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX
        
        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id
        
        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            node.feature = feature
            node.threshold = threshold
        
        self.node_count += 1 

        return node_id


    cpdef np.ndarray predict(self, object X):

        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                                mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        
        return out

    cpdef np.ndarray apply(self, object X):

        return self._apply_dense(X)

    cdef inline np.ndarray _apply_dense(self, object X):

        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*>out.data

        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                while node.left_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)

        return out

    cdef np.ndarray _get_value_ndarray(self):
        
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr 
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr 

    cdef np.ndarray _get_node_ndarray(self):

        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape, 
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self

        return arr 

    cpdef object decision_path(self, object X):

        return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):

        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype = np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data
        
        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples*(1+self.max_depth), dtype = np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                while node.left_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i+1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]
                
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices), dtype=np.intp)

        out = csr_matrix((data, indices, indptr), shape=(n_samples, self.node_count))
    
    cpdef compute_feature_importances(self, normalize=True):

        cdef Node* left 
        cdef Node* right 
        cdef Node* nodes = self.nodes 
        cdef Node* node = nodes 
        cdef Node* end_node = node + self.node_count 

        cdef double normalizer = 0

        cdef np.ndarray[np.float64_t, ndim=1] importances 
        importances = np.zeros((self.n_features, ))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity - 
                        left.weighted_n_node_samples * left.impurity - 
                        right.weighted_n_node_samples * right.impurity
                    )
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                importances /= normalizer 

        return importances

    def compute_partial_dependence(self, DTYPE_t[:, ::1] X,
                                    int[::1] target_features, double[::1] out):

        cdef double[::1] weight_stack = np.zeros(self.node_count, dtype=np.float64)
        cdef SIZE_t[::1] node_idx_stack = np.zeros(self.node_count, dtype=np.intp)
        cdef SIZE_t sample_idx
        cdef SIZE_t feature_idx
        cdef int stack_size 
        cdef double left_sample_frac
        cdef double current_weight 
        cdef double total_weight
        cdef Node* current_node
        cdef SIZE_t current_node_idx
        cdef bint is_target_feature
        cdef SIZE_t _TREE_LEAF = TREE_LEAF

        for sample_idx in range(X.shape[0]):
            stack_size = 1
            node_idx_stack[0] = 0
            weight_stack[0] = 1
            total_weight = 0

            while stack_size > 0:
                stack_size -= 1
                current_node_idx = node_idx_stack[stack_size]
                current_node = &self.nodes[current_node_idx]

                if current_node.left_child == _TREE_LEAF:
                    out[sample_idx] += (weight_stack[stack_size] * 
                                        self.value[current_node_idx])
                    total_weight += weight_stack[stack_size]
                else:

                    is_target_feature = False
                    for feature_idx in range(target_features.shape[0]):
                        if target_features[feature_idx] == current_node.feature:
                            is_target_feature = True
                            break
                    
                    if is_target_feature:
                        if X[sample_idx, feature_idx] <= current_node.threshold:
                            node_idx_stack[stack_size] = current_node.left_child
                        else:
                            node_idx_stack[stack_size] = current_node.right_child
                    else:
                        node_idx_stack[stack_size] = current_node.left_child
                        left_sample_frac = (
                            self.nodes[current_node.left_child].weighted_n_node_samples / 
                            current_node.weighted_n_node_samples)
                        current_weight = weight_stack[stack_size]
                        weight_stack[stack_size] = current_weight * left_sample_frac

                        node_idx_stack[stack_size] = current_node.right_child
                        weight_stack[stack_size] = (
                            current_weight * (1 - left_sample_frac)
                        )    
                        stack_size += 1
                 
cdef class _CCPPruneController:
    """Base class used by build_pruned_tree_ccp and ccp_pruning_path
    to control pruning.
    """
    cdef bint stop_pruning(self, DOUBLE_t effective_alpha) nogil:
        """Return 1 to stop pruning and 0 to continue pruning"""
        return 0

    cdef void save_metrics(self, DOUBLE_t effective_alpha,
                           DOUBLE_t subtree_impurities) nogil:
        """Save metrics when pruning"""
        pass

    cdef void after_pruning(self, unsigned char[:] in_subtree) nogil:
        """Called after pruning"""
        pass

cdef class _AlphaPruner(_CCPPruneController):
    """
    Use alpha to control when to stop pruning
    """

    cdef DOUBLE_t ccp_alpha
    cdef SIZE_t capacity

    def __cinit__(self, DOUBLE_t ccp_alpha):
        self.ccp_alpha = ccp_alpha
        self.capacity = 0

    cdef bint stop_pruning(self, DOUBLE_t effective_alpha) nogil:
        
        return self.ccp_alpha < effective_alpha

    cdef void after_pruning(self, unsigned char[:] in_subtree) nogil:

        """
        Updates the number of leaves in subtree
        """

        for i in range(in_subtree.shape[0]):
            if in_subtree[i]:
                self.capacity += 1

cdef class _PathFinder(_CCPPruneController):

    cdef DOUBLE_t[:] ccp_alphas
    cdef DOUBLE_t[:] impurities 
    cdef UINT32_t count 

    def __cinit__(self, int node_count):

        self.ccp_alphas = np.zeros(shape=(node_count), dtype = np.float64)
        self.impurities = np.zeros(shape=(node_count), dtype = np.float64)
        self.count = 0

    cdef void save_metrics(self, DOUBLE_t effective_alpha, 
                            DOUBLE_t subtree_impurities) nogil:

        self.ccp_alphas[self.count] = effective_alpha
        self.impurities[self.count] = subtree_impurities
        self.count += 1

cdef _cost_complexity_prune(unsigned char[:] leaves_in_subtree, 
                            Tree orig_tree, 
                            _CCPPruneController controller):
    """
    Perform cost complexity pruning 

    This function takes an already grown tree, "orig_tree" and outputs a 
    boolean mask 'leaves_in_subtree' to are the leaves in the pruned tree 
    """ 

    cdef:
        SIZE_t i
        SIZE_t n_nodes = orig_tree.node_count
        # prior probability using weighted samples
        DOUBLE_t[:] weighted_n_node_samples = orig_tree.weighted_n_node_samples
        DOUBLE_t total_sum_weights = weighted_n_node_samples[0]
        DOUBLE_t[:] impurity = orig_tree.impurity
        # weighted impurity of each node
        DOUBLE_t[:] r_node = np.empty(shape=n_nodes, dtype=np.float64)

        SIZE_t[:] child_l = orig_tree.children_left
        SIZE_t[:] child_r = orig_tree.children_right
        SIZE_t[:] parent = np.zeros(shape=n_nodes, dtype=np.intp)

        # Only uses the start and parent variables
        ut.Stack stack = ut.Stack(INITIAL_STACK_SIZE)
        ut.StackRecord stack_record
        int rc = 0
        SIZE_t node_idx

        SIZE_t[:] n_leaves = np.zeros(shape=n_nodes, dtype=np.intp)
        DOUBLE_t[:] r_branch = np.zeros(shape=n_nodes, dtype=np.float64)
        DOUBLE_t current_r
        SIZE_t leaf_idx
        SIZE_t parent_idx

        # candidate nodes that can be pruned
        unsigned char[:] candidate_nodes = np.zeros(shape=n_nodes,
                                                    dtype=np.uint8)
        # nodes in subtree
        unsigned char[:] in_subtree = np.ones(shape=n_nodes, dtype=np.uint8)
        DOUBLE_t[:] g_node = np.zeros(shape=n_nodes, dtype=np.float64)
        SIZE_t pruned_branch_node_idx
        DOUBLE_t subtree_alpha
        DOUBLE_t effective_alpha
        SIZE_t child_l_idx
        SIZE_t child_r_idx
        SIZE_t n_pruned_leaves
        DOUBLE_t r_diff
        DOUBLE_t max_float64 = np.finfo(np.float64).max

    with nogil:

        for i in range(r_node.shape[0]):
            r_node[i] = (
                weighted_n_node_samples[i] * impurity[i] / total_sum_weights
            )
        
        rc = stack.push(0, 0, 0, -1, 0, 0, 0)
        if rc == -1:
            with gil:
                raise MemoryError("pruning tree")

        while not stack.is_empty():
            stack.pop(&stack_record)
            node_idx = stack_record.start
            parent[node_idx] = stack_record.parent
            if child_l[node_idx] == _TREE_LEAF:
                leaves_in_subtree[node_idx] = 1
            else:
                rc = stack.push(child_l[node_idx], 0, 0, node_idx, 0, 0, 0)
                rc = stack.push(child_r[node_idx], 0, 0, node_idx, 0, 0, 0)

        # Computes number of leaves in all branches and the overall impurity of the branch 
        # The overall impurity is the sum of r_node in its leaves.

        for leaf_idx in range(leaves_in_subtree.shape[0]):
            if not leaves_in_subtree[leaf_idx]:
                continue
            r_branch[leaf_idx] = r_node[leaf_idx]

            current_r = r_node[leaf_idx]
            while leaf_idx != 0:
                parent_idx = parent[leaf_idx]
                r_branch[parent_idx] += current_r
                n_leaves[parent_idx] += 1
                leaf_idx = parent_idx
        
        for i in range(leaves_in_subtree.shape[0]):
            candidate_nodes[i] = not leaves_in_subtree[i]
        
        controller.save_metrics(0.0, r_branch[0])

        while candidate_nodes[0]:

            effective_alpha = max_float64
            for i in range(n_nodes):
                if not candidate_nodes[i]:
                    continue
                subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
                if subtree_alpha < effective_alpha:
                    effective_alpha = subtree_alpha
                    pruned_branch_node_idx = i 

            if controller.stop_pruning(effective_alpha):
                break
            
            rc = stack.push(pruned_branch_node_idx, 0, 0, 0, 0, 0, 0)
            
            # descendants of branch are not in subtree
            while not stack.is_empty():
                stack.pop(&stack_record)
                node_idx = stack_record.start

                if not in_subtree[node_idx]:
                    continue
                candidate_nodes[node_idx] = 0
                leaves_in_subtree[node_idx] = 0
                in_subtree[node_idx] = 0

                if child_l[node_idx] != _TREE_LEAF:
                    rc = stack.push(child_l[node_idx], 0, 0, 0, 0, 0, 0)
                    rc = stack.push(child_r[node_idx], 0, 0, 0, 0, 0, 0)

            leaves_in_subtree[pruned_branch_node_idx] = 1
            in_subtree[pruned_branch_node_idx] = 1

            n_pruned_leaves = n_leaves[pruned_branch_node_idx] - 1
            n_leaves[pruned_branch_node_idx] = 0

            r_diff = r_node[pruned_branch_node_idx] - r_branch[pruned_branch_node_idx]
            r_branch[pruned_branch_node_idx] = r_node[pruned_branch_node_idx]

            node_idx = parent[pruned_branch_node_idx]
            while node_idx != -1:
                n_leaves[node_idx] -= n_pruned_leaves
                r_branch[node_idx] += r_diff

            controller.save_metrics(effective_alpha, r_branch[0])
        
        controller.after_pruning(in_subtree)

def _build_pruned_tree_ccp(Tree tree, Tree orig_tree, DOUBLE_t ccp_alpha):

    cdef SIZE_t n_nodes = orig_tree.node_count
    cdef unsigned char[:] leaves_in_subtree = np.zeros(shape=n_nodes, dtype=np.uint8)

    pruning_controller = _AlphaPruner(ccp_alpha=ccp_alpha)
    _cost_complexity_prune(leaves_in_subtree, orig_tree, pruning_controller)
    _build_pruned_tree(tree, orig_tree, leaves_in_subtree, 
                        pruning_controller.capacity)

def ccp_pruning_path(Tree orig_tree):

    cdef unsigned char[:] leaves_in_subtree = np.zeros(
        shape = orig_tree.node_count, dtype = np.uint8)
    
    path_finder = _PathFinder(orig_tree.node_count)
    _cost_complexity_prune(leaves_in_subtree, orig_tree, path_finder)

    cdef UINT32_t total_items = path_finder.count
    cdef np.ndarray ccp_alphas = np.empty(shape = total_items, dtype = np.float64)
    cdef np.ndarray impurities = np.empty(shape = total_items, dtype = np.float64)
    cdef UINT32_t count = 0

    while count < total_items:
        ccp_alphas[count] = path_finder.ccp_alphas[count]
        impurities[count] = path_finder.impurities[count]
        count += 1

    return {'ccp_alphas': ccp_alphas, 'impurities': impurities}

cdef _build_pruned_tree(Tree tree, Tree orig_tree, 
                        const unsigned char[:] leaves_in_subtree, SIZE_t capacity):
    
    tree._resize(capacity)

    cdef:
        SIZE_t orig_node_id
        SIZE_t new_node_id
        SIZE_t depth
        SIZE_t parent
        bint is_left
        bint is_leaf

        SIZE_t value_stride = orig_tree.value_stride
        SIZE_t max_depth_seen = -1
        int rc = 0
        Node* node 
        double* orig_value_ptr
        double* new_value_ptr

        ut.Stack stack = ut.Stack(INITIAL_STACK_SIZE)
        ut.StackRecord stack_record
    
    with nogil:

        rc = stack.push(0, 0, 0, _TREE_UNDEFINED, 0, 0.0, 0)

        while not stack.is_empty():
            stack.pop(&stack_record)
            
            orig_node_id = stack_record.start
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left

            is_leaf = leaves_in_subtree[orig_node_id]
            node = &orig_tree.nodes[orig_node_id]

            new_node_id = tree._add_node(
                parent, is_left, is_leaf, node.feature, node.threshold, 
                node.impurity, node.n_node_samples,
                node.weighted_n_node_samples
            )

            if new_node_id == SIZE_MAX:
                rc = -1
                break
            
            orig_value_ptr = orig_tree.value + value_stride * orig_node_id
            new_value_ptr = tree.value + value_stride * new_node_id
            memcpy(new_value_ptr, orig_value_ptr, sizeof(double) * value_stride)

            if not is_leaf:
                rc = stack.push(
                    node.right_child, 0, depth + 1, new_node_id, 0, 0.0, 0
                )
                if rc == -1:
                    break

                rc = stack.push(node.left_child, 0, depth + 1, new_node_id, 1, 0.0, 0)
                if rc == -1:
                    break
            
            if depth > max_depth_seen:
                max_depth_seen = depth

        if rc >= 0:
            tree.max_depth = max_depth_seen
