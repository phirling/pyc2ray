# Centralized place to load Fortran and C++/CUDA extensions

_c2ray_lib = None
_c2ray_lib_loaded = None
_octa_lib = None
_octa_lib_loaded = None

# Load the f2py-compiled C2Ray subroutines
def load_c2ray():
    global _c2ray_lib
    global _c2ray_lib_loaded

    # Try once to load the library
    if _c2ray_lib_loaded is None:
        try:
            from .lib import libc2ray
            _c2ray_lib_loaded = True
            _c2ray_lib = libc2ray
            return _c2ray_lib
        except ImportError:
            # If C2Ray is not found, stop execution (the package cannot be used without)
            _c2ray_lib_loaded = False
            raise RuntimeError("Could not load c2ray library (image not found)")
    elif _c2ray_lib_loaded is True:
        return _c2ray_lib
    else:
        return None
    
# Load the OCTA raytracing library
def load_octa():
    global _octa_lib
    global _octa_lib_loaded

    # Try once to load the library
    if _octa_lib_loaded is None:
        try:
            from .lib import libocta
            _octa_lib_loaded = True
            _octa_lib = libocta
            return _octa_lib
        except ImportError:
            # If octa is not found, the package can still be used but we inform the user that GPU raytracing is not available
            print("Info: could not load OCTA library (image not found)")
            _octa_lib_loaded = False
    elif _octa_lib_loaded is True:
        return _octa_lib
    else:
        return None