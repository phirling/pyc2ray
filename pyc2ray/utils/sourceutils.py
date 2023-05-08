import numpy as np

def read_sources(file,n,mode):
    """ Read in a source file formatted for C2Ray
    
    Reads in a source list file formatted for the Fortran version of C2Ray and returns its contents
    as python objects suitably shaped for the wrapped versions.

    Parameters
    ----------
    file : string
        Name of the file to read
    n : int
        Numer of sources to read from the file
    case : string
        Raytracing code for which to format the sources. Can be either "pyc2ray" or "pyc2ray_octa". This is because
        the data has to be structured differently for these two modes.
    """
    
    with open(file,"r") as f:
        # Exclude first row and last column which are just conventional for c2ray
        #inp = np.loadtxt(f,skiprows=1)[:,:-1]
        inp = np.loadtxt(f,skiprows=1,usecols=(0,1,2,3),ndmin=2) # < -- ndmin = 2 in case of single source in the file
        
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
            
            if (mode == "pyc2ray"):
                srcpos = np.empty((3,src_num),dtype='int64')
                srcpos[0] = src_x
                srcpos[1] = src_y
                srcpos[2] = src_z
                return srcpos, src_flux, src_num
            elif (mode == "pyc2ray_octa"):
                srcpos = np.empty((3,src_num),dtype='int32')
                srcpos[0] = src_x - 1
                srcpos[1] = src_y - 1
                srcpos[2] = src_z - 1
                srcpos = np.ravel(srcpos,order='F')
                return srcpos, src_flux.astype('float64'), src_num
            else:
                raise ValueError("Unknown mode: " + mode)
            
def generate_test_sourcefile(filename,N,numsrc,strength,seed=100):
    """Generate a test source file for C2Ray

    Generate sources of equal strength at random grid positions and write to file
    formatted for C2Ray.

    Parameters
    ----------
    filename : string
        Name of the file to write (will overwrite an existing file!)
    N : int
        Grid size
    numsrc : int
        Number of sources to generate
    strength : float
        Strength of the sources in number of ionizing photons per second
    seed : int (optional)
        Seed to use with numpy.random. Default: 100
    """

    # Random Generator with given seed
    rng = np.random.RandomState(seed)
    srcpos = 1+rng.randint(0,N,size=3*numsrc)
    srcpos = srcpos.reshape((numsrc,3),order='C')
    srcflux = strength * np.ones((numsrc,1))
    zerocol = np.zeros((numsrc,1)) # By convention for c2ray

    output = np.hstack((srcpos,srcflux,zerocol))

    with open(filename,'w') as f:
        f.write(f"{numsrc:n}\n")

    with open(filename,'a') as f:
        np.savetxt(f,output,("%i %i %i %.0e %.1f"))