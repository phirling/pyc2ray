#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "raytracing.cuh"

// ===========================================================================
// OCTA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the OCTA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

extern "C"
{   
    // ========================================================================
    // Raytrace all sources and compute photoionization rates
    // ========================================================================
    static PyObject *
    octa_do_all_sources(PyObject *self, PyObject *args)
    {
        PyArrayObject * srcpos;
        PyArrayObject * srcflux;
        double R;
        PyArrayObject * coldensh_out;
        double sig;
        double dr;
        PyArrayObject * ndens;
        PyArrayObject * xh_av;
        PyArrayObject * phi_ion;
        int NumSrc;
        int m1;
        double minlogtau;
        double dlogtau;
        int NumTau;

        if (!PyArg_ParseTuple(args,"OOdOddOOOiiddi",
        &srcpos,
        &srcflux,
        &R,
        &coldensh_out,
        &sig,
        &dr,
        &ndens,
        &xh_av,
        &phi_ion,
        &NumSrc,
        &m1,
        &minlogtau,
        &dlogtau,
        &NumTau))
            return NULL;
        
        // Error checking
        if (!PyArray_Check(srcpos) || !PyArray_ISINTEGER(srcpos))
        {
            PyErr_SetString(PyExc_TypeError,"Srcpos must be Array of type int");
            return NULL;
        }
        if (!PyArray_Check(coldensh_out) || PyArray_TYPE(coldensh_out) != NPY_DOUBLE)
        {
            PyErr_SetString(PyExc_TypeError,"coldensh_out must be Array of type double");
            return NULL;
        }

        // Get Array data
        int * srcpos_data = (int*)PyArray_DATA(srcpos);
        double * srcflux_data = (double*)PyArray_DATA(srcflux);
        double * coldensh_out_data = (double*)PyArray_DATA(coldensh_out);
        double * ndens_data = (double*)PyArray_DATA(ndens);
        double * phi_ion_data = (double*)PyArray_DATA(phi_ion);
        double * xh_av_data = (double*)PyArray_DATA(xh_av);

        do_all_sources_octa_gpu(srcpos_data,srcflux_data,R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1,minlogtau,dlogtau,NumTau);

        return Py_None;
    }

    // ========================================================================
    // Allocate GPU memory for grid data
    // ========================================================================
    static PyObject *
    octa_device_init(PyObject *self, PyObject *args)
    {
        int N;
        if (!PyArg_ParseTuple(args,"i",&N))
            return NULL;
        device_init(N);
        return Py_None;
    }

    // ========================================================================
    // Deallocate GPU memory
    // ========================================================================
    static PyObject *
    octa_device_close(PyObject *self, PyObject *args)
    {
        device_close();
        return Py_None;
    }

    // ========================================================================
    // Copy density grid to GPU
    // ========================================================================
    static PyObject *
    octa_density_to_device(PyObject *self, PyObject *args)
    {
        int N;
        PyArrayObject * ndens;
        if (!PyArg_ParseTuple(args,"Oi",&ndens,&N))
            return NULL;

        double * ndens_data = (double*)PyArray_DATA(ndens);
        density_to_device(ndens_data,N);

        return Py_None;
    }

    // ========================================================================
    // Copy radiation table to GPU
    // ========================================================================
    static PyObject *
    octa_photo_table_to_device(PyObject *self, PyObject *args)
    {
        int NumTau;
        PyArrayObject * table;
        if (!PyArg_ParseTuple(args,"Oi",&table,&NumTau))
            return NULL;

        double * table_data = (double*)PyArray_DATA(table);
        photo_table_to_device(table_data,NumTau);

        return Py_None;
    }

    // ========================================================================
    // Define module functions and initialization function
    // ========================================================================
    static PyMethodDef octaMethods[] = {
        {"do_all_sources",  octa_do_all_sources, METH_VARARGS,"Do OCTA raytracing (GPU)"},
        {"device_init",  octa_device_init, METH_VARARGS,"Free GPU memory"},
        {"device_close",  octa_device_close, METH_VARARGS,"Free GPU memory"},
        {"density_to_device",  octa_density_to_device, METH_VARARGS,"Copy density field to GPU"},
        {"photo_table_to_device",  octa_photo_table_to_device, METH_VARARGS,"Copy radiation table to GPU"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef octamodule = {
        PyModuleDef_HEAD_INIT,
        "libocta",   /* name of module */
        "CUDA C++ implementation of the short-characteristics RT", /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        octaMethods
    };

    PyMODINIT_FUNC
    PyInit_libocta(void)
    {   
        PyObject* module = PyModule_Create(&octamodule);
        import_array();
        return module;
    }
}