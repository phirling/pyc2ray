project('pyc2ray','c',
    version : '0.1',
    default_options : ['warning_level=2'])

add_languages('fortran')

add_global_arguments('-cpp', language : 'fortran')

py_mod = import('python')
py3 = py_mod.find_installation('python3')
py3_dep = py3.dependency()
message(py3.path())
message(py3.get_install_dir())

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

#--f2cmap f2c.f2py_f2cmap

srcdir_c2ray = 'src/c2ray/'

c2ray_source = custom_target('libc2raymodule.c',
    input : [srcdir_c2ray+'photorates.f90',srcdir_c2ray+'raytracing.f90',srcdir_c2ray+'chemistry.f90'],
    output : ['libc2raymodule.c','libc2ray-f2pywrappers.f90'],
    command : [py3,'-m','numpy.f2py','@INPUT@','-m','libc2ray','--f2cmap f2c.f2py_f2cmap'])

inc_np = include_directories(incdir_numpy,incdir_f2py)

py3.extension_module('libc2ray',
    srcdir_c2ray+'photorates.f90',
    srcdir_c2ray+'raytracing.f90',
    srcdir_c2ray+'chemistry.f90',
    incdir_f2py+'/fortranobject.c',
    include_directories: inc_np,
    dependencies : py3_dep,
    install : true,
    fortran_args : '-DUSE_SUBBOX')