import numpy as np

def read_sources(file,n,case):
    """ Read in a source file formatted in the style of C2Ray
    
    Parameters
    ----------
    file : string
        Name of the file to read
    n : int
        Numer of sources to read from the file
    case : string
        Code for which to format the sources. Can be either "pyc2ray" or "pyc2ray_octa"
    
    """
    
    with open(file,"r") as f:
        # Exclude first row and last column which are just conventional for c2ray
        inp = np.loadtxt(f,skiprows=1)[:,:-1]
        
        max_n = inp.shape[0]
        
        if (n > max_n):
            raise ValueError(f"Number of sources given ({n:n}) is larger than that of the file ({max_n:n})")
        else:
            inp = inp[:n]
            src_x = inp[:,0]
            src_y = inp[:,1]
            src_z = inp[:,2]
            src_flux = inp[:,3]
            src_num = inp.shape[0]
            
            if (case == "pyc2ray"):
                srcpos = np.empty((3,src_num),dtype='int64')
                srcpos[0] = src_x
                srcpos[1] = src_y
                srcpos[2] = src_z
                return srcpos, src_flux, src_num
            elif (case == "pyc2ray_octa"):
                srcpos = np.empty((3,src_num),dtype='int32')
                srcpos[0] = src_x - 1
                srcpos[1] = src_y - 1
                srcpos[2] = src_z - 1
                srcpos = np.ravel(srcpos,order='F')
                return srcpos, src_flux, src_num
            else:
                raise ValueError("Unknown case: " + case)