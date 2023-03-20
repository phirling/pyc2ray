import numpy as np
import astropy as ap

# TODO:
# Deal with internal/physical units. Depending on Nbody code used, 
# internal units can be either with h factors or not, etc
class Grid:
    """ Class that represents a simulation grid.
    This unifies the "grid" and "material" modules from the Fortran version
    (which in turn contain the density, temperature and ionfractions modules)

    Parameters
    ----------
    mesh: int or array
        Number of cells in each dimension. If scalar, same number of
        cells in each dimension will be used.
    boxsize: float or array
        Physical length of each dimension (= meshsize * cell size)
        If scalar, same number of cells in each dimension will be used.

    Attributes
    ----------
    mesh: 1d-array of size 3
        Number of cells in each dimension
    boxsize: 1d-array of size 3
        Physical length of each dimension (= meshsize * cell size)
    dr: 1d-array of size 3
        Physical dimensions of a cell in each dimension
    vol: float
        Volume of a cell
    igrid: float
        Physical coordinates of the centers of the cells in the i direction (i=x,y,z)
    
    """
    # in grid.F90:
    # dr gives the physical cell size (in internal units, in c2ray meters)
    # x,y,z give the physical coordinates of the centers of the cells
    def __init__(self,mesh,boxsize) -> None:

        # Mesh and grid sizes
        self.mesh = to_triple(mesh,dtype=np.int_)
        self.boxsize = to_triple(boxsize)

        # Grid cell size (in each dim)
        self.dr = self.boxsize / self.mesh
        
        # Cell volume
        self.vol = np.prod(self.dr)

        # Coordinates of the cell centers
        self.xgrid = (np.arange(self.mesh[0]) + 0.5)*self.dr[0]
        self.ygrid = (np.arange(self.mesh[1]) + 0.5)*self.dr[1]
        self.zgrid = (np.arange(self.mesh[2]) + 0.5)*self.dr[2]

        # For convenience: mesh shape as a tuple
        self.shape = (self.mesh[0],self.mesh[1],self.mesh[2])

    """Check if an array of values is compatible with the dimensions of the grid
    """
    def is_compatible(self,field):
        if field.shape == self.shape:
            return True
        else:
            print(f"Provided field doesn't match grid size ({field.shape} =/= {self.shape})")
            return False
        
class Material:
    """Object that represents physical properties living on a grid.
    These properties are scalar fields of density, temperature and
    ionization fraction.

    Parameters
    ----------
    grid: Grid
        The grid object the material is defined on
    isothermal: bool
        Whether or not the run using this material is isothermal
    
    Attributes
    ----------
    grid: Grid
        The grid object the material is defined on
    density: DensityField
        Represents the density in each grid cell
    temperature: TemperatureField
        Represents the temperature in each grid cell

    """
    def __init__(self,grid,isothermal) -> None:
        self.grid = grid
        self.density = DensityField(self.grid)
        if isothermal:
            self.temperature = IsothermalTemperatureField(self.grid)
        else:
            self.temperature = VariableTemperatureField(self.grid)

    def material_ini(self,temp):
        self.temperature.temperature_init(temp)
        # The density field is initialized at construction
    
class DensityField:
    """Class representing a density distributed on a grid

    Parameters
    ----------
    grid: Grid
        The grid object the field is defined on

    Attributes
    ----------
    grid: Grid
        The grid object the field is defined on
    ndens: 3D-array of floats
        The density in each grid cell
    avg_dens: float
        The average density on the grid
    """
    def __init__(self,grid) -> None:
        self.grid = grid
        self.ndens = np.ones(grid.shape)
        self.avg_dens = 1.0
    
    def set_density(self,dens):
        if self.grid.is_compatible(dens):
            self.ndens = dens
        else:
            raise ValueError("(Density field)")
    
    def set_constant_average_density(self,n):
        self.ndens = n*np.ones(self.grid.shape)

class TemperatureField:
    """Base class representing a temperature distributed on a grid

    """
    def __init__(self,grid) -> None:
        self.grid = grid

    def temperature_init(self,temp):
        pass
    
    def value(self,i,j,k):
        pass

class IsothermalTemperatureField(TemperatureField):
    """Represents an isothermal, homogeneous temperature distribution
    
    """
    def __init__(self, grid) -> None:
        super().__init__(grid)
        self.isothermal = True
        self.temper = None
    
    def temperature_init(self,temp):
        self.temper = temp

    def value(self, i, j, k):
        return self.temper
    
class VariableTemperatureField(TemperatureField):
    """Represents a spatially and temporally variable temperature distribution
    
    """
    def __init__(self, grid) -> None:
        super().__init__(grid)
        self.isothermal = False
        self.temper = np.zeros(self.grid.shape)

    def temperature_init(self, temp):
        if self.grid.is_compatible(temp):
            self.temper = temp
        else:
            raise ValueError("(Temperature field)")
    
class GridField:
    """Abstract class representing any quantity (field) distributed on a grid

    Is subclassed to form the relevant physical fields.

    Parameters
    ----------
    grid: Grid
        The grid object the field is defined on

    Attributes
    ----------
    grid: Grid
        The grid object the field is defined on
    values: 3D-array of floats
        The scalar value of the field in each grid cell
    avg: float
        The average value of the field on the grid

    """
    def __init__(self,grid) -> None:
            self.grid = grid
            self.values = np.ones(grid.shape)
            self.avg = 1.0
    def set_values(self,val):
        if val.shape != self.grid.shape:
            raise ValueError(f"Provided density field doesn't match"
                              "grid size ({val.shape} =/= {self.grid.shape})")
        else:
            self.values = val








"""Convenience Method to convert scalar input to a cube"""
def to_triple(input,dtype = None):
    if hasattr(input,"__len__"):
        if len(input) == 3:
            return np.array(input,dtype=dtype) # (int(input[0]),int(input[1]),int(input[2]))
        else:
            raise ValueError("Mesh and grid must be either scalars or array-like of dimension 3")
    else:
        # _inp = int(input)
        # return (_inp,_inp,_inp)
        return input*np.ones(3,dtype=dtype)
