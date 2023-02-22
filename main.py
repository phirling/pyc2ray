import numpy as np
from clocks import ClockSet
from IOManager import IOManager
from grid import Grid, Material
import astropy as ap

paramfile = "parameters.yml"

"""
The I/O manager reads the parameter file and saves its values
internally. They are accessed using its param(group,key) method,
where 'group' and 'key' label the relevant entries in the yml file.

It is then used for any operation that involves reading and writing
from external files and handles filenames and so on internally.
In particular, it is used to read the DM densities from an input
N-Body simulation, which are then passed to the material class.

The clock set starts wall and cpu timers and can be used to write
to a time log file.

The grid contains information about cell sizes, coordinates, etc.

The material represents a set of quantities living on the grid,
namely density and temperature (one value per cell)
"""

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
# This module essentially only calls its own functions once to fill
# a set of attributes (mostly tables i.e. arrays) which are
# then used by other components of the program. Tedius but easy to translate

# Initialize Material
mat = Material(grid,io.param('Material','isothermal'))
mat.material_ini(io.param('Material','initial_temperature'))
clock.write_walltimestamp("Time after material init")

# Initialize redshifts (nbody_ini in original C2Ray)
io.redshift_ini()
clock.write_walltimestamp("Time after redshift/nbody init")

# TODO: source_ini

# time_ini is replaced by simply reading the two relevant parameters from yml file

# TODO: evolve_ini

# TODO: cosmology_ini
from astropy.cosmology import Planck18
cosmology = Planck18()

# Testing
import matplotlib.pyplot as plt
ndens = io.read_density(z=1.23)
mat.density.set_density(ndens)
# plt.imshow(mat.density.ndens.sum(axis=2))
# plt.show()