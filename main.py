import numpy as np
from clocks import ClockSet
from IOManager import IOManager
from grid import Grid, Material

paramfile = "parameters.yml"

# The I/O manager reads from a yml file and global parameters
# can be read from it using the .param(group,key) method.
# Important parameters can be copied out
io = IOManager(paramfile)
mesh = io.param('Grid','mesh')
boxsize = io.param('Grid','boxsize')

# Setup Clocks
clock = ClockSet(io.timefile)

# Initialize Output
io.setup_output()

# Initialize Grid
grid = Grid(mesh,boxsize)
clock.write_walltimestamp("Time after grid init")

# TODO: rad_ini

# TODO: material_ini
mat = Material(grid)
clock.write_walltimestamp("Time after material init")

# This replaces nbody_ini which essentially just reads in a list of redshifts
NumZred, zred_array = io.redshift_ini()