# Centralized place to load Fortran and C++/CUDA extensions
_c2ray_lib = None
_c2ray_lib_loaded = False
_octa_lib = None
_octa_lib_loaded = False

def load_c2ray():
    global _c2ray_lib
    global _c2ray_lib_loaded

    if _c2ray_lib_loaded:
        return _c2ray_lib
    else:
        try:
            from .lib import libc2ray
            _c2ray_lib_loaded = True
            _c2ray_lib = libc2ray
        except ImportError:
            print("Could not load c2ray library (image not found)")
            _c2ray_lib_loaded = False

        return _c2ray_lib
    

def load_octa():
    global _octa_lib
    global _octa_lib_loaded

    if not _octa_lib_loaded:
        return _octa_lib
    else:
        try:
            from .lib import libocta
            _octa_lib_loaded = True
            _octa_lib = libocta
        except ImportError:
            print("Could not load OCTA library")
            _octa_lib_loaded = False
        
        return _octa_lib