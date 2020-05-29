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

