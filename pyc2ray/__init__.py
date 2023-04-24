from .evolve import *
from .sourceutils import *
from .parameters import *

def printlog(s,filename,quiet=False):
    """Write log and print to screen

    Parameters
    ----------
    s : str
        String to write to log
    filename : str
        Name of the logfile to append text to
    quiet : bool
        Don't write to stdout. Default is False
    """

    with open(filename,"a") as f:
        f.write(s + "\n")
    if not quiet: print(s)
