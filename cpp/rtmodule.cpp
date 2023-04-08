#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "raytracing.hh"

extern "C"
{
    static PyObject *
    rtc_octa(PyObject *self, PyObject *args)
    {
        
    }

    static PyMethodDef RTCMethods[] = {
        {"octa",  rtc_octa, METH_VARARGS,"Do OCTA raytracing (CPU)"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef rtcmodule = {
        PyModuleDef_HEAD_INIT,
        "rtc",   /* name of module */
        "C++ implementation of the short-characteristics RT", /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        RTCMethods
    };

    PyMODINIT_FUNC
    PyInit_RTC(void)
    {   
        PyObject* module = PyModule_Create(&rtcmodule);
        import_array();
        return module;
    }
}