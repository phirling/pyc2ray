#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "raytracing.hh"
#include "raytracing_gpu.cuh"

extern "C"
{
    static PyObject *
    rtc_octa(PyObject *self, PyObject *args)
    {
        PyArrayObject * srcpos;
        PyArrayObject * srcflux;
        int ns;
        double R;
        PyArrayObject * coldensh_out;
        double sig;
        double dr;
        PyArrayObject * ndens;
        PyArrayObject * xh_av;
        PyArrayObject * phi_ion;
        int NumSrc;
        int m1;

        if (!PyArg_ParseTuple(args,"OOidOddOOOii",
        &srcpos,
        &srcflux,
        &ns,
        &R,
        &coldensh_out,
        &sig,
        &dr,
        &ndens,
        &xh_av,
        &phi_ion,
        &NumSrc,
        &m1))
            return NULL;
        
        // Error checking
        if (!PyArray_Check(srcpos) || (PyArray_TYPE(srcpos) != NPY_INT32))
        {   
            printf("%i",PyArray_TYPE(coldensh_out));
            printf("%i",NPY_INT32);
            PyErr_SetString(PyExc_TypeError,"Srcpos must be Array of type int32");
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

        do_source_octa(srcpos_data,srcflux_data,ns,R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1);

        return PyFloat_FromDouble(1.0);
    }

    static PyObject *
    rtc_octa_multiple(PyObject *self, PyObject *args)
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

        if (!PyArg_ParseTuple(args,"OOdOddOOOii",
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
        &m1))
            return NULL;
        
        // Error checking
        if (!PyArray_Check(srcpos) || (PyArray_TYPE(srcpos) != NPY_INT32))
        {   
            printf("%i",PyArray_TYPE(coldensh_out));
            printf("%i",NPY_INT32);
            PyErr_SetString(PyExc_TypeError,"Srcpos must be Array of type int32");
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

        all_sources_octa(srcpos_data,srcflux_data,R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1);

        return PyFloat_FromDouble(1.0);
    }

    static PyObject *
    rtc_octa_gpu(PyObject *self, PyObject *args)
    {
        PyArrayObject * srcpos;
        PyArrayObject * srcflux;
        int ns;
        double R;
        PyArrayObject * coldensh_out;
        double sig;
        double dr;
        PyArrayObject * ndens;
        PyArrayObject * xh_av;
        PyArrayObject * phi_ion;
        int NumSrc;
        int m1;

        if (!PyArg_ParseTuple(args,"OOidOddOOOii",
        &srcpos,
        &srcflux,
        &ns,
        &R,
        &coldensh_out,
        &sig,
        &dr,
        &ndens,
        &xh_av,
        &phi_ion,
        &NumSrc,
        &m1))
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

        do_source_octa_gpu(srcpos_data,srcflux_data,ns,R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1);

        return PyFloat_FromDouble(1.0);
    }

    static PyObject *
    rtc_device_init(PyObject *self, PyObject *args)
    {
        int N;
        if (!PyArg_ParseTuple(args,"i",&N))
            return NULL;
        device_init(N);
        return PyFloat_FromDouble(1.0);
    }

    static PyObject *
    rtc_device_close(PyObject *self, PyObject *args)
    {
        device_close();
        return PyFloat_FromDouble(1.0);
    }

    static PyMethodDef RTCMethods[] = {
        {"octa",  rtc_octa, METH_VARARGS,"Do OCTA raytracing (CPU)"},
        {"octa_multiple",  rtc_octa_multiple, METH_VARARGS,"Do OCTA raytracing for multiple sources (CPU)"},
        {"octa_gpu",  rtc_octa_gpu, METH_VARARGS,"Do OCTA raytracing (GPU)"},
        {"device_init",  rtc_device_init, METH_VARARGS,"Free GPU memory"},
        {"device_close",  rtc_device_close, METH_VARARGS,"Free GPU memory"},
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