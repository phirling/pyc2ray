project('pyc2ray','c',
    version : '0.1',
    default_options : [
        'warning_level=2'])

add_languages('fortran')
add_global_arguments('-cpp', language : 'fortran') # For lowercase-uppercase stuff

# https://numpy.org/doc/stable/f2py/buildtools/meson.html
py_mod = import('python')
py3 = py_mod.find_installation('python3')
py3_dep = py3.dependency()
message(py3.path())
message(py3.get_install_dir())

# Get Numpy and F2PY include dirs
incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

# F2PY type mapping file for Fortran -> C
f2cmapfile = files('src/c2ray/f2c.f2py_f2cmap')

# Fortran source files for C2Ray
c2ray_sources = [
    'src/c2ray/photorates.f90',
    'src/c2ray/raytracing.f90',
    'src/c2ray/chemistry.f90',
]

# Construct C-wrapper for Fortran extension module using F2PY
c2ray_wrapper_source = custom_target('libc2raymodule.c',
    input : c2ray_sources,
    output : ['libc2raymodule.c','libc2ray-f2pywrappers2.f90'],
    command : [py3,'-m','numpy.f2py','@INPUT@','-m','libc2ray','--f2cmap',f2cmapfile])

# Include directories for extension module
inc_np = include_directories(incdir_numpy,incdir_f2py)

# Build C2Ray extension module with F2PY
py3.extension_module('libc2ray',
    c2ray_wrapper_source,
    c2ray_sources,
    incdir_f2py+'/fortranobject.c',
    include_directories: inc_np,
    dependencies : py3_dep,
    install : true,
    subdir : 'pyc2ray/lib',
    fortran_args : '-DUSE_SUBBOX')