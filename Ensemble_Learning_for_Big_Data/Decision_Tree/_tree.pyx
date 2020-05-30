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
