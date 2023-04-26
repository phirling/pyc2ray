#!/bin/bash
nvcc -Xcompiler -fPIC -O2 -shared -o RTC.so -I/usr/include/python3.10 -I/home/phirling/miniconda3/envs/astro/lib/python3.10/site-packages/numpy/core/include rtmodule.cpp raytracing.cpp raytracing_gpu.cu -v
cp RTC.so ../testing_area
