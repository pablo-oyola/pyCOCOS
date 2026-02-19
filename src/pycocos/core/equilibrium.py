"""
Library to handle the magnetic equilibrium and compute magnetic coordinates
related to tokamaks.
"""

import numpy as np
import xarray as xr
import os
from typing import Union, Optional, Tuple, Dict, Any
from findiff import FinDiff
from skimage import measure
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.constants import mu_0

# Importing the internal utils.
from ..coordinates.registry import get_jacobian_function
from ..coordinates.compute_coordinates import compute_magnetic_coordinates
from .magnetic_coordinates import magnetic_coordinates as MagneticCoordinates


def _require_matplotlib_pyplot():
    """
    Import matplotlib pyplot lazily for plotting helpers.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for equilibrium plotting helpers. "
            "Install pyCOCOS with plotting extras: pip install 'pyCOCOS[plot]'."
        ) from exc
    return plt

# ----------------------------------------------------------------------------
# AUXILIAR FUNCTIONS TO COMPUTE MAGNETIC FIELD PROPERTIES.
# ----------------------------------------------------------------------------
def getFSprop(R: float, z: float):
    """
    For a given flux surface described by the corresponding (R, z) the
    following properties are computed:
        - Outermost radius (Raus)
        - Innermost radius (Rin)
        - Upper z-value (Zup)
        - Lower z-value (Zdown)
        - Geometrical radius (Rgeo)
        - Minor radius (ageo)
        - Upper triangularity (delta_u)
        - Lower triangularity (delta_d)
        - Average triangularity (delta)
        - Elongation (kappa)
    """

    R = np.atleast_1d(R)
    z = np.atleast_1d(z)

    # Checking the consistency of the inputs.
    if R.size != z.size:
        raise ValueError('R and z must have same size (%d != %d)'%(R.size,
                                                                   z.size))

    # Computing the properties.
    output = { 'Raus': R.max(),
               'Rin':  R.min(),
               'Zup':  z.max(),
               'Zdown': z.min()
             }

    output['Rgeo'] = (output['Raus'] + output['Rin'])/2.0
    output['ageo'] = (output['Raus'] - output['Rin'])/2.0

    # To compute the upper/lower triangularity we need to get major radius at
    # which we have the Zup and the Zdown, respectively.
    Rup   = R[z.argmax()]
    Rdown = R[z.argmin()]

    output['delta_u'] = (output['Rgeo'] - Rup)/output['ageo']
    output['delta_d'] = (output['Rgeo'] - Rdown)/output['ageo']
    output['delta']   = (output['delta_u'] + output['delta_d'])/2.0
    output['kappa']   = (output['Zup'] - output['Zdown'])/output['ageo']


    return output

def get_currents(R: float, z: float, br: float, bz: float, bphi: float):
    """
    Compute the currents flowing in the plasma for a given flux surface.

    :param R: Major radius.
    :param z: Vertical coordinate.
    :param br: Radial magnetic field.
    :param bz: Vertical magnetic field.
    :param bphi: Toroidal magnetic field.
    :return: Dictionary with the computed current values.
    """
    currents = dict()
    # We will use the Ampere law to determine the current.
    dr = R[1] - R[0]
    dz = z[1] - z[0]
    d_dr = FinDiff(0, dr, 1, acc=4)
    d_dz = FinDiff(1, dz, 1, acc=4)

    # The currents are the curl of the magnetic field.
    jr = - d_dz(bphi) / mu_0
    jz = + d_dr(R[:, None] * bphi) / (mu_0 * R[:, None])
    jphi = (1.0 / mu_0) * (- d_dr(bz) + d_dz(br))

    currents['jr'] = jr
    currents['jz'] = jz
    currents['jphi'] = jphi

    return currents

# ----------------------------------------------------------------------------
# MAIN CLASS FUNCTION.
# ----------------------------------------------------------------------------
class equilibrium:
    """
    Container for magnetic equilibrium data and coordinate transformations.

    This class stores tokamak equilibrium data including magnetic field
    components, flux surfaces, and provides methods to compute magnetic
    coordinates for various coordinate systems.

    Parameters
    ----------
    rgrid : np.ndarray
        Radial grid points (major radius R)
    zgrid : np.ndarray
        Vertical grid points (height z)
    br : np.ndarray
        Radial component of magnetic field (2D: R x z)
    bz : np.ndarray
        Vertical component of magnetic field (2D: R x z)
    bphi : np.ndarray
        Toroidal component of magnetic field (2D: R x z)
    psi : np.ndarray
        Poloidal magnetic flux (2D: R x z)
    Raxis : float
        Radial position of magnetic axis
    zaxis : float
        Vertical position of magnetic axis
    psi_edge : float
        Poloidal flux at plasma boundary
    psi_ax : float
        Poloidal flux at magnetic axis
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Default is True

    Attributes
    ----------
    Rgrid : xr.DataArray
        Radial grid
    zgrid : xr.DataArray
        Vertical grid
    Bdata : xr.Dataset
        Magnetic field data (Br, Bz, Bphi, Babs, Bpol)
    fluxdata : xr.Dataset
        Flux surface data (psipol, rhopol)
    boundary : xr.Dataset
        Plasma boundary data
    Jdata : xr.Dataset
        Current density data

    Examples
    --------
    >>> from pycocos import Equilibrium
    >>> eq = Equilibrium(rgrid, zgrid, br, bz, bphi, psi, Raxis, zaxis, psi_edge, psi_ax)
    >>> mag_coords = eq.compute_coordinates(coordinate_system='boozer')

    """

    def __init__(
        self,
        rgrid: np.ndarray,
        zgrid: np.ndarray,
        br: np.ndarray,
        bz: np.ndarray,
        bphi: np.ndarray,
        psi: np.ndarray,
        Raxis: float,
        zaxis: float,
        psi_edge: float,
        psi_ax: float,
        phiclockwise: bool = True
    ) -> None:
        """
        Initialize a generic equilibrium.

        Parameters
        ----------
        rgrid : np.ndarray
            Radial grid points (major radius R)
        zgrid : np.ndarray
            Vertical grid points (height z)
        br : np.ndarray
            Radial component of magnetic field (2D: R x z)
        bz : np.ndarray
            Vertical component of magnetic field (2D: R x z)
        bphi : np.ndarray
            Toroidal component of magnetic field (2D: R x z)
        psi : np.ndarray
            Poloidal magnetic flux (2D: R x z)
        Raxis : float
            Radial position of magnetic axis
        zaxis : float
            Vertical position of magnetic axis
        psi_edge : float
            Poloidal flux at plasma boundary
        psi_ax : float
            Poloidal flux at magnetic axis
        phiclockwise : bool, optional
            Whether toroidal angle increases clockwise. Default is True

        Raises
        ------
        ValueError
            If grid dimensions don't match field arrays or axis is outside domain
        """
        rgrid = np.atleast_1d(rgrid)
        zgrid = np.atleast_1d(zgrid)
        self.phiclockwise = phiclockwise
        
        # Store axis values for later use in _build_structured_data
        self._Raxis_init = Raxis
        self._zaxis_init = zaxis
        self._psi_ax_init = psi_ax
        self._psi_edge_init = psi_edge

        # Checking size consistency.
        self.nr = len(rgrid)
        self.nz = len(zgrid)

        if br.ndim > 2:
            raise ValueError(f'Dimension of Br is {br.ndim}, instead of 2!')

        if bz.ndim > 2:
            raise ValueError(f'Dimension of Bz is {bz.ndim}, instead of 2!')

        if bphi.ndim > 2:
            raise ValueError(f'Dimension of Bphi is {bphi.ndim}, instead of 2!')

        if psi.ndim > 2:
            raise ValueError(f'Dimension of Psi is {bphi.ndim}, instead of 2!')

        if br.shape[0] != self.nr:
            raise ValueError(f'First dimension of Br must be {self.nr}')

        if bz.shape[0] != self.nr:
            raise ValueError(f'First dimension of Bz must be {self.nr}')

        if bphi.shape[0] != self.nr:
            raise ValueError(f'First dimension of Bphi must be {self.nr}')

        if psi.shape[0] != self.nr:
            raise ValueError(f'First dimension of Psi must be {self.nr}')

        if br.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Br must be {self.nz}')

        if bz.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Bz must be {self.nz}')

        if bphi.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Bphi must be {self.nz}')

        if psi.shape[1] != self.nz:
            raise ValueError(f'Second dimension of Psi must be {self.nz}')

        # Checking that the magnetic axis.
        if (Raxis < rgrid.min()) or (Raxis > rgrid.max()):
            raise ValueError(f'Magnetic axis must be within the domain: {Raxis}.')

        if (zaxis < zgrid.min()) or (zaxis > zgrid.max()):
            raise ValueError(f'Magnetic axis must be within the domain: {zaxis}.')

        # Storing the data internally.
        self.Rgrid = xr.DataArray(rgrid, dims=('R',),
                                  attrs={'name': 'R',
                                         'desc': 'Major radius',
                                         'short_name': 'Major radius',
                                         'units': 'm'})
        self.zgrid = xr.DataArray(zgrid, dims=('z',),
                                  attrs={'name': 'z',
                                         'desc': 'Height',
                                         'short_name': 'Height',
                                         'units': 'm'})


        # We create a dataset to store all the magnetic-field related 2D data.
        self.Bdata = xr.Dataset()


        self.Bdata['Br'] = xr.DataArray(br, coords=(self.Rgrid, self.zgrid),
                                            attrs={'name': 'Br',
                                                'desc': 'Radial magnetic field',
                                                'short_name': '$B_R$',
                                                'units': 'T'})
        self.Bdata['Bz'] = xr.DataArray(bz, coords=(self.Rgrid, self.zgrid),
                                            attrs={'name': 'Bz',
                                                'desc': 'Vertical magnetic field',
                                                'short_name': '$B_z$',
                                                'units': 'T'})
        self.Bdata['Bphi'] = xr.DataArray(bphi, coords=(self.Rgrid, self.zgrid),
                                                attrs={'name': 'Bphi',
                                                        'desc': 'Toroidal magnetic field',
                                                        'short_name': '$B_\\varphi$',
                                                        'units': 'T'})
        self.Bdata['Babs'] = np.sqrt(self.Bdata.Br**2 + self.Bdata.Bz**2 + self.Bdata.Bphi**2)
        self.Bdata.Babs.attrs['name'] = 'Babs'
        self.Bdata.Babs.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Babs.attrs['desc'] = 'Magnetic field strenght'
        self.Bdata.Babs.attrs['short_name'] = '$B_{abs}$'

        self.Bdata['Bpol'] = np.sqrt(self.Bdata.Br**2 + self.Bdata.Bz**2)# * np.sign(self.Bdata.Bz)
        self.Bdata.Bpol.attrs['name'] = 'Bpol'
        self.Bdata.Bpol.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Bpol.attrs['desc'] = 'Poloidal magnetic field'
        self.Bdata.Bpol.attrs['short_name'] = '$B_{pol}$'


        # Building the magnetic coordinates.
        self.fluxdata = xr.Dataset()

        self.fluxdata['psipol'] = xr.DataArray(psi, coords=(self.Rgrid, self.zgrid),
                                                attrs={'name': 'Psi',
                                                        'desc': 'Poloidal magnetic flux',
                                                        'short_name': '$\\Psi$',
                                                        'units': 'Wb'})

        # Checking consistency of the axis flux.
        psiax_intrp = self.fluxdata.psipol.interp(R=Raxis, z=zaxis, method='cubic')
        if(np.abs(psiax_intrp - psi_ax) > 1.0e-4):
            raise ValueError('The specified magnetic axis is not ' +
                             'consistent with the input flux: ' +
                             '%f (evaluated) vs %f (input)' % (psiax_intrp, psi_ax))

        # With the psipol, we get the magnetic axis and the separatrix values
        # for the flux to finally get rhopol.
        psimax = psi_edge - psi_ax
        self.fluxdata['rhopol'] = np.sqrt((self.fluxdata.psipol - psi_ax) / psimax)
        self.fluxdata.rhopol.attrs['units'] = ''
        self.fluxdata.rhopol.attrs['desc'] = 'Radial magnetic coordinate'
        self.fluxdata.rhopol.attrs['name'] = 'rhopol'
        self.fluxdata.rhopol.attrs['short_name'] = '$\\rho_{pol}$'

        # From this rhopol, we can get the separatrix contour line.
        R, z = self.rhopol2rz((1.0,))
        if len(R) == 1:
            R = R[0]
            z = z[0]

        self._boundary = xr.Dataset()
        self._boundary['R'] = xr.DataArray(R, dims=('idx',),
                                           attrs={'name': 'R',
                                                  'desc': 'LCFS Radii',
                                                  'short_name': 'R',
                                                  'units': 'm'})
        self._boundary['z'] = xr.DataArray(z, dims=('idx',),
                                           attrs={'name': 'z',
                                                  'desc': 'LCFS Heights',
                                                  'short_name': 'z',
                                                  'units': 'm'})

        # Checking consistency of the flux at the LCFS.
        psiedge_intrp = self.fluxdata.psipol.interp(R=self._boundary.R[0],
                                                    z=self._boundary.z[0], method='cubic')
        if(np.abs(psiedge_intrp - psi_edge) > 1.0e-4):
            raise ValueError('The specified separatrix flux is not consistent with the equilibrium.')

        self._boundary.attrs['psi_bdy'] = psi_edge
        self._boundary.attrs['psi_ax'] = psi_ax
        self._boundary.attrs['psimax'] = psimax

        # Getting the radius of the separatrix at the geometrical midplane.
        rgrid_max = float(self.Rgrid.values[-1])
        nr_fine = int((rgrid_max - float(Raxis)) / 1.0e-3)
        rgrid_fine = np.linspace(float(Raxis), rgrid_max, nr_fine)

        # Sometimes life is hard and the coils are close to our plasma
        # and there may be places outside the confined region where 
        # psipol < psi_edge, and the corresponding rhopol is an imaginary
        # number. We will use in that case linear interpolation instead
        # of cubic.
        if np.any(np.isnan(self.fluxdata.rhopol.values)):
            method = 'linear'
        else:
            method = 'cubic'

        rhop1d = self.fluxdata.rhopol.interp(R=rgrid_fine,
                                             z=zaxis,
                                             method=method).values
        idx = np.abs(rhop1d[rhop1d <= 1.0] - 1.0).argmin()
        Raus = rgrid_fine[idx]

        # We add now this variables to fluxdata
        self.fluxdata.attrs['Raxis'] = Raxis
        self.fluxdata.attrs['zaxis'] = zaxis
        self.fluxdata.attrs['Raus']  = Raus
        self.fluxdata.attrs['aminor'] = Raus - Raxis


        # Getting the magnetic field at the axis.
        self.Bdata['Baxis'] = self.Bdata.Babs.interp(R=Raxis, z=zaxis, method='cubic')
        self.Bdata.Baxis.attrs['name'] = 'Baxis'
        self.Bdata.Baxis.attrs['units'] = self.Bdata.Br.attrs['units']
        self.Bdata.Baxis.attrs['desc'] = 'Magnetic field at the axis'
        self.Bdata.Baxis.attrs['short_name'] = '$B_{ax}$'

        # Getting the current density.
        jdata = get_currents(self.Bdata.R.values, self.Bdata.z.values,
                             self.Bdata.Br.values, self.Bdata.Bz.values,
                             self.Bdata.Bphi.values)
        self.Jdata = xr.Dataset()
        self.Jdata['Jr'] = xr.DataArray(jdata['jr'], dims = ('R', 'z'),
                                        coords={'R': self.Bdata.R, 'z': self.Bdata.z},
                                        attrs={'name': 'Jr',
                                               'desc': 'Radial current density',
                                               'short_name': '$J_R$',
                                               'units': 'A/m$^2$'})
        self.Jdata['Jz'] = xr.DataArray(jdata['jz'], dims = ('R', 'z'),
                                        attrs={'name': 'Jz',
                                               'desc': 'Vertical current density',
                                               'short_name': '$J_Z$',
                                               'units': 'A/m$^2$'})
        self.Jdata['Jphi'] = xr.DataArray(jdata['jphi'], dims = ('R', 'z'),
                                           attrs={'name': 'Jphi',
                                                  'desc': 'Toroidal current density',
                                                  'short_name': '$J_{phi}$',
                                                  'units': 'A/m$^2$'})
        
        # We now evaluate the current at the axis.
        Jraxis = self.Jdata.Jr.interp(R=Raxis, z=zaxis, method='cubic').values
        Jzaxis = self.Jdata.Jz.interp(R=Raxis, z=zaxis, method='cubic').values
        Jphiaxis = self.Jdata.Jphi.interp(R=Raxis, z=zaxis, method='cubic').values
        Jaxis = np.sqrt(Jraxis**2 + Jzaxis**2 + Jphiaxis**2) * np.sign(Jphiaxis)
        self.Jdata.attrs['Jaxis'] = Jaxis

        # Create structured data organization (Option A: sub-Datasets as views)
        # These provide convenient access while keeping backward compatibility
        self._build_structured_data()

        # Creating the plotting lists (kept for backward compatibility)
        self.plot_1d_names = dict()
        self.plot_2d_names = dict()

        # Adding the variables for the plotting.
        for ivar in self.Bdata.data_vars:
            if self.Bdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.Bdata[ivar])

        for ivar in self.fluxdata.data_vars:
            if self.fluxdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.fluxdata[ivar])
        
        for ivar in self.Jdata.data_vars:
            if self.Jdata[ivar].values.ndim == 0:
                continue
            self.add_var(ivar, self.Jdata[ivar])

    def _build_structured_data(self) -> None:
        """
        Build structured data organization as xr.Dataset views.
        
        Creates convenient sub-Datasets (grid, field, flux, profiles, geometry)
        that provide views/subsets of the underlying data for easier access.
        """
        # Get axis values from stored attributes
        Raxis_val = self._Raxis_init
        zaxis_val = self._zaxis_init
        psi_ax_val = self._psi_ax_init
        psi_bdy_val = self._psi_edge_init
        psimax_val = psi_bdy_val - psi_ax_val
        
        # Grid: R and z coordinates
        self._grid = xr.Dataset({
            'R': self.Rgrid,
            'z': self.zgrid,
        })
        
        # Field: magnetic field components
        self._field = xr.Dataset({
            'Br': self.Bdata.Br,
            'Bz': self.Bdata.Bz,
            'Bphi': self.Bdata.Bphi,
            'B': self.Bdata.Babs,  # Total magnetic field
            'Bpol': self.Bdata.Bpol,  # Poloidal magnetic field
        })
        
        # Flux: flux surfaces
        self._flux = xr.Dataset({
            'psi': self.fluxdata.psipol,
            'rho': self.fluxdata.rhopol,
        })
        
        # Geometry: axis and boundary
        self._geometry = xr.Dataset({
            'R_axis': xr.DataArray(Raxis_val, attrs={'name': 'R_axis', 'units': 'm',
                                                     'desc': 'Radial position of magnetic axis',
                                                     'short_name': '$R_{axis}$'}),
            'z_axis': xr.DataArray(zaxis_val, attrs={'name': 'z_axis', 'units': 'm',
                                                     'desc': 'Vertical position of magnetic axis',
                                                     'short_name': '$z_{axis}$'}),
            'R_boundary': self._boundary.R,
            'z_boundary': self._boundary.z,
        })
        self._geometry.attrs.update({
            'psi_ax': psi_ax_val,
            'psi_bdy': psi_bdy_val,
            'psimax': psimax_val,
        })
        
        # Profiles: 1D profiles (initially empty, populated by EQDSK or user)
        self._profiles = xr.Dataset()
        
        # Initialize profiles dict for tracking
        self._profiles_dict = {}

    @property
    def grid(self) -> xr.Dataset:
        """Grid coordinates (R, z)."""
        return self._grid
    
    @property
    def field(self) -> xr.Dataset:
        """Magnetic field components (Br, Bz, Bphi, B, Bpol)."""
        return self._field
    
    @property
    def flux(self) -> xr.Dataset:
        """Flux surfaces (psi, rho)."""
        return self._flux
    
    @property
    def geometry(self) -> xr.Dataset:
        """Geometric properties (axis, boundary)."""
        return self._geometry
    
    @property
    def profiles(self) -> xr.Dataset:
        """1D profiles (q, pres, fpol, etc.)."""
        return self._profiles
    
    # Direct property access for common quantities
    @property
    def R(self) -> xr.DataArray:
        """Radial grid coordinates."""
        return self.Rgrid
    
    @property
    def z(self) -> xr.DataArray:
        """Vertical grid coordinates."""
        return self.zgrid
    
    @property
    def Br(self) -> xr.DataArray:
        """Radial magnetic field component."""
        return self.Bdata.Br
    
    @property
    def Bz(self) -> xr.DataArray:
        """Vertical magnetic field component."""
        return self.Bdata.Bz
    
    @property
    def Bphi(self) -> xr.DataArray:
        """Toroidal magnetic field component."""
        return self.Bdata.Bphi
    
    @property
    def B(self) -> xr.DataArray:
        """Total magnetic field magnitude."""
        return self.Bdata.Babs
    
    @property
    def Bpol(self) -> xr.DataArray:
        """Poloidal magnetic field magnitude."""
        return self.Bdata.Bpol
    
    @property
    def psi(self) -> xr.DataArray:
        """Poloidal magnetic flux."""
        return self.fluxdata.psipol
    
    @property
    def rho(self) -> xr.DataArray:
        """Normalized poloidal flux coordinate."""
        return self.fluxdata.rhopol
    
    @property
    def R_axis(self) -> float:
        """Radial position of magnetic axis."""
        return float(self.geometry.R_axis.values)
    
    @property
    def z_axis(self) -> float:
        """Vertical position of magnetic axis."""
        return float(self.geometry.z_axis.values)
    
    @property
    def boundary(self) -> xr.Dataset:
        """Plasma boundary (LCFS) coordinates."""
        return self._boundary
    
    @property
    def axis(self) -> xr.Dataset:
        """Magnetic axis position."""
        return xr.Dataset({
            'R': self.geometry.R_axis,
            'z': self.geometry.z_axis,
        })
    
    # Profile properties (may be None if not loaded)
    @property
    def q(self) -> Optional[xr.DataArray]:
        """Safety factor profile (if available)."""
        return self._profiles.get('q', None)
    
    @property
    def pres(self) -> Optional[xr.DataArray]:
        """Pressure profile (if available)."""
        return self._profiles.get('pres', None)
    
    @property
    def fpol(self) -> Optional[xr.DataArray]:
        """F(psi) = R*B_phi profile (if available)."""
        return self._profiles.get('fpol', None)

    def add_var(
        self,
        varname: str,
        var: xr.DataArray,
        add_to_class: bool = False
    ) -> None:
        """
        Add a variable to the database of the class.

        The class keeps a database of variables that can be used to store
        any information. This method allows adding a new variable to the
        database and eases handling of plotting routines.

        Parameters
        ----------
        varname : str
            Name of the variable to add
        var : xr.DataArray
            Variable as DataArray with proper metadata (name, desc, units, short_name)
        add_to_class : bool, optional
            If True, also add as class attribute. Default is False

        Raises
        ------
        ValueError
            If variable name already exists or metadata is missing

        Examples
        --------
        >>> eq.add_var('custom_field', my_dataarray)
        """
        # Checking if the variable name already exists.
        if (varname in self.plot_1d_names) or (varname in self.plot_2d_names):
            raise ValueError('Variable name already exists in the database.')

        # Checking that the variable contains all the metadata.
        if not hasattr(var, 'attrs'):
            raise ValueError('Variable does not have the metadata.')

        minimalmetadata = ['name', 'desc', 'short_name', 'units']
        for ikey in minimalmetadata:
            if ikey not in var.attrs:
                raise ValueError('Variable does not have the metadata (%s)' % ikey)

        # Checking the size of the input variable.
        ndim = var.ndim
        if ndim == 1:
            self.plot_1d_names[varname] = var
        elif ndim == 2:
            self.plot_2d_names[varname] = var
        else:
            raise NotImplementedError('Only 1D and 2D variables are supported.')

        if add_to_class:
            self.__dict__[varname] = var

    def rhopol2rz(
        self,
        rhopol: Union[float, np.ndarray],
        return_all: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform rhopol into (R, z) trajectories.

        Parameters
        ----------
        rhopol : float or np.ndarray
            Value(s) of the rho poloidal to transform into (R, z) contours
        return_all : bool, optional
            If True, return all contour segments. Default is False

        Returns
        -------
        R : np.ndarray
            Radial coordinates of the contour(s)
        z : np.ndarray
            Vertical coordinates of the contour(s)

        Examples
        --------
        >>> R, z = eq.rhopol2rz(0.5)  # Get contour at rho=0.5
        >>> R, z = eq.rhopol2rz([0.3, 0.5, 0.7])  # Get multiple contours
        """
        rhopol = np.atleast_1d(rhopol)
        nrho = rhopol.size  # Number of contours to evaluate.

        # We generate the empty output.
        R = np.empty((nrho,), dtype=object)
        z = np.empty((nrho,), dtype=object)

        Rmin = self.Rgrid.values.min()
        dr   = self.Rgrid.values[1] - self.Rgrid.values[0]
        zmin = self.zgrid.values.min()
        dz   = self.zgrid.values[1] - self.zgrid.values[0]

        # Looking for the contours.
        for ii in range(nrho):
            aux = measure.find_contours(self.fluxdata.rhopol.values, rhopol[ii],
                                        fully_connected='high')

            idx = 0
            if len(aux) == 0:
                continue
            if return_all:
                R[ii] = [(Rmin + dr*aux[idx][:, 0]) for idx in range(len(aux))]
                z[ii] = [(zmin + dz*aux[idx][:, 1]) for idx in range(len(aux))]
            else:
                if len(aux) > 1:
                    # We have here to choose among different options, since
                    # several flux surfaces legs have been found.
                    # We take as a measure of the closedness of the flux surface
                    # the fact that the last and first point must the closest to
                    # each other possible.
                    dl = np.zeros((len(aux),), dtype=float)
                    for jj in range(len(aux)):
                        drr = (aux[jj][-1, 0] - aux[jj][0, 0])
                        dzz = (aux[jj][-1, 1] - aux[jj][0, 1])

                        dl[jj] = np.sqrt(drr**2 + dzz**2)

                    idx = dl.argmin()
                R[ii] = aux[idx][:, 0]
                z[ii] = aux[idx][:, 1]

                # Transforming the indices into variables R and z.
                R[ii] = Rmin + dr*R[ii]
                z[ii] = zmin + dz*z[ii]

        return R, z

    def __call__(
        self,
        R: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        grid: bool = False
    ) -> xr.Dataset:
        """
        Evaluate magnetic field at given position(s).

        This provides a callable interface similar to magnetic_coordinates:
        `eq(R, z)` returns magnetic field components.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s)
        z : float or np.ndarray
            Vertical coordinate(s)
        grid : bool, optional
            If True, create a grid from R and z. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing Br, Bz, Bphi, B, Bpol at the requested points

        Examples
        --------
        >>> B = eq(2.0, 0.0)  # Single point
        >>> B = eq([1.5, 2.0], [0.0, 0.1])  # Multiple points
        >>> B = eq(R_grid, z_grid, grid=True)  # Grid evaluation
        """
        return self.getB(R, z, grid=grid)

    def getB(
        self,
        Rin: Union[float, np.ndarray],
        zin: Union[float, np.ndarray],
        grid: bool = False
    ) -> xr.Dataset:
        """
        Evaluate the magnetic field at the input position(s).

        Parameters
        ----------
        Rin : float or np.ndarray
            Radial coordinate(s) where to evaluate B
        zin : float or np.ndarray
            Vertical coordinate(s) where to evaluate B
        grid : bool, optional
            If True, create a grid from Rin and zin. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing Br, Bz, Bphi, B, Bpol at the requested points

        Examples
        --------
        >>> B = eq.getB(2.0, 0.0)  # Evaluate at single point
        >>> B = eq.getB([1.5, 2.0, 2.5], [0.0, 0.1, 0.0])  # Evaluate at multiple points
        """
        # Creating the output dataset.
        output = xr.Dataset()

        Rin = np.atleast_1d(Rin)
        zin = np.atleast_1d(zin)

        output['R'] = xr.DataArray(Rin, attrs={'name': 'R',
                                               'units': 'm',
                                               'desc': 'Major radius'},
                                   dims=('R') if not grid else None)

        output['z'] = xr.DataArray(zin, attrs={'name': 'z',
                                               'units': 'm',
                                               'desc': 'Height'},
                                   dims=('z') if not grid else None)

        # Interpolate magnetic field components
        if grid:
            for ikey in ('Br', 'Bz', 'Bphi'):
                output[ikey] = self.Bdata[ikey].interp(R=Rin, z=zin,
                                                        method='cubic')
        else:
            for ikey in ('Br', 'Bz', 'Bphi'):
                intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                            self.Bdata[ikey].values)
                tmp = intrp(Rin, zin, grid=False)
                dims = ('point',) if tmp.ndim == 1 else None
                output[ikey] = xr.DataArray(tmp, dims=dims)
                output[ikey].attrs.update(self.Bdata[ikey].attrs)

        # Compute derived quantities
        output['B'] = np.sqrt(output['Br']**2 + output['Bz']**2 + output['Bphi']**2)
        output['B'].attrs.update({
            'name': 'B',
            'units': 'T',
            'desc': 'Total magnetic field magnitude',
            'short_name': '$B$'
        })
        
        output['Bpol'] = np.sqrt(output['Br']**2 + output['Bz']**2)
        output['Bpol'].attrs.update({
            'name': 'Bpol',
            'units': 'T',
            'desc': 'Poloidal magnetic field magnitude',
            'short_name': '$B_{pol}$'
        })

        return output

    def flux_surface(
        self,
        rho: Union[float, np.ndarray]
    ) -> xr.Dataset:
        """
        Get flux surface contour(s) at given rho value(s).

        Parameters
        ----------
        rho : float or np.ndarray
            Normalized poloidal flux coordinate value(s)

        Returns
        -------
        xr.Dataset
            Dataset with R and z coordinates of the flux surface(s)

        Examples
        --------
        >>> surface = eq.flux_surface(0.5)  # Single flux surface
        >>> surfaces = eq.flux_surface([0.3, 0.5, 0.7])  # Multiple surfaces
        """
        R, z = self.rhopol2rz(rho, return_all=False)
        
        # Handle single vs multiple surfaces
        if isinstance(R, np.ndarray) and R.dtype == object:
            # Multiple surfaces
            datasets = []
            for i in range(len(R)):
                if R[i] is not None and len(R[i]) > 0:
                    datasets.append(xr.Dataset({
                        'R': xr.DataArray(R[i], dims=('idx',),
                                         attrs={'name': 'R', 'units': 'm',
                                                'desc': 'Radial coordinate',
                                                'short_name': 'R'}),
                        'z': xr.DataArray(z[i], dims=('idx',),
                                         attrs={'name': 'z', 'units': 'm',
                                                'desc': 'Vertical coordinate',
                                                'short_name': 'z'}),
                    }))
            if len(datasets) == 1:
                return datasets[0]
            # Return as list or concatenate - for now return first if single value
            return datasets[0] if len(datasets) > 0 else xr.Dataset()
        else:
            # Single surface
            if R is not None and len(R) > 0:
                return xr.Dataset({
                    'R': xr.DataArray(R, dims=('idx',),
                                     attrs={'name': 'R', 'units': 'm',
                                            'desc': 'Radial coordinate',
                                            'short_name': 'R'}),
                    'z': xr.DataArray(z, dims=('idx',),
                                     attrs={'name': 'z', 'units': 'm',
                                            'desc': 'Vertical coordinate',
                                            'short_name': 'z'}),
                })
            else:
                return xr.Dataset()

    def interpolate(
        self,
        R: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        variables: Optional[list] = None,
        grid: bool = False
    ) -> xr.Dataset:
        """
        Generic interpolation of equilibrium variables onto (R, z) points.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s)
        z : float or np.ndarray
            Vertical coordinate(s)
        variables : list of str, optional
            Variables to interpolate. If None, interpolates all 2D variables
            (Br, Bz, Bphi, psi, rho). Default is None
        grid : bool, optional
            If True, create a grid from R and z. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing interpolated variables

        Examples
        --------
        >>> data = eq.interpolate(2.0, 0.0)  # Single point
        >>> data = eq.interpolate([1.5, 2.0], [0.0, 0.1], variables=['psi', 'Br'])
        """
        if variables is None:
            variables = ['Br', 'Bz', 'Bphi', 'psi', 'rho']
        
        output = xr.Dataset()
        R_arr = np.atleast_1d(R)
        z_arr = np.atleast_1d(z)
        
        # Map variable names to their datasets
        var_map = {
            'Br': ('Bdata', 'Br'),
            'Bz': ('Bdata', 'Bz'),
            'Bphi': ('Bdata', 'Bphi'),
            'B': ('Bdata', 'Babs'),
            'Bpol': ('Bdata', 'Bpol'),
            'psi': ('fluxdata', 'psipol'),
            'rho': ('fluxdata', 'rhopol'),
        }
        
        for var_name in variables:
            if var_name in var_map:
                ds_name, var_key = var_map[var_name]
                source_ds = getattr(self, ds_name)
                source_var = source_ds[var_key]
                
                if grid:
                    output[var_name] = source_var.interp(R=R_arr, z=z_arr, method='cubic')
                else:
                    intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                                source_var.values)
                    tmp = intrp(R_arr, z_arr, grid=False)
                    dims = ('point',) if tmp.ndim == 1 else None
                    output[var_name] = xr.DataArray(tmp, dims=dims)
                    output[var_name].attrs.update(source_var.attrs)
        
        return output

    def at_rho(
        self,
        rho: Union[float, np.ndarray]
    ) -> xr.Dataset:
        """
        Get flux-surface quantities at given rho value(s).

        Parameters
        ----------
        rho : float or np.ndarray
            Normalized poloidal flux coordinate value(s)

        Returns
        -------
        xr.Dataset
            Dataset containing flux-surface averaged or profile values
            (q, pres, fpol if available, plus geometric properties)

        Examples
        --------
        >>> fs_data = eq.at_rho(0.5)  # Get quantities at rho=0.5
        """
        rho_arr = np.atleast_1d(rho)
        output = xr.Dataset()
        
        # Interpolate profiles if available
        if len(self._profiles) > 0:
            # Profiles are typically functions of psi_n or rho
            # For now, interpolate from profiles if they exist
            for var_name in self._profiles.data_vars:
                profile_var = self._profiles[var_name]
                if profile_var.ndim == 1:
                    # Interpolate profile to requested rho values
                    # Assuming profile is on rho coordinate
                    if 'rhop' in profile_var.coords or len(profile_var.coords) == 1:
                        coord_name = list(profile_var.coords.keys())[0]
                        output[var_name] = profile_var.interp({coord_name: rho_arr},
                                                               method='linear',
                                                               kwargs={'fill_value': 'extrapolate'})
        
        # Add geometric properties
        output['rho'] = xr.DataArray(rho_arr, dims=('rho',),
                                    attrs={'name': 'rho', 'units': '',
                                           'desc': 'Normalized poloidal flux',
                                           'short_name': r'$\rho$'})
        
        return output

    def summary(self) -> None:
        """
        Print a summary of equilibrium properties.

        Examples
        --------
        >>> eq.summary()
        """
        print("=" * 60)
        print("Equilibrium Summary")
        print("=" * 60)
        print(f"\nGrid:")
        print(f"  R: [{self.Rgrid.min().values:.3f}, {self.Rgrid.max().values:.3f}] m ({self.nr} points)")
        print(f"  z: [{self.zgrid.min().values:.3f}, {self.zgrid.max().values:.3f}] m ({self.nz} points)")
        
        print(f"\nMagnetic Axis:")
        print(f"  R_axis = {self.R_axis:.3f} m")
        print(f"  z_axis = {self.z_axis:.3f} m")
        print(f"  B_axis = {float(self.Bdata.Baxis.values):.3f} T")
        
        print(f"\nFlux Surfaces:")
        print(f"  psi_ax = {self.geometry.attrs.get('psi_ax', 'N/A'):.3e} Wb")
        print(f"  psi_bdy = {self.geometry.attrs.get('psi_bdy', 'N/A'):.3e} Wb")
        print(f"  psimax = {self.geometry.attrs.get('psimax', 'N/A'):.3e} Wb")
        
        if len(self._profiles) > 0:
            print(f"\nProfiles available:")
            for var_name in self._profiles.data_vars:
                print(f"  - {var_name}")
        else:
            print(f"\nProfiles: None loaded")
        
        print(f"\nMagnetic Field Components:")
        print(f"  Br:   [{self.Br.min().values:.3f}, {self.Br.max().values:.3f}] T")
        print(f"  Bz:   [{self.Bz.min().values:.3f}, {self.Bz.max().values:.3f}] T")
        print(f"  Bphi: [{self.Bphi.min().values:.3f}, {self.Bphi.max().values:.3f}] T")
        print(f"  B:    [{self.B.min().values:.3f}, {self.B.max().values:.3f}] T")
        
        print("=" * 60)

    def plot(self, name: Optional[str] = None, ax=None, put_labels: bool=True,
             line=None, **kwargs):
        """
        Plot equilibrium variables.

        If no name is provided, lists available plottable variables.
        Otherwise, plots the specified variable (2D or 1D).

        Parameters
        ----------
        name : str, optional
            Variable name to plot. If None, lists available variables.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        put_labels : bool, optional
            If True, add axis labels and title. Default is True
        line : matplotlib Line2D or Image, optional
            Existing plot object to update. Default is None
        **kwargs
            Additional arguments passed to matplotlib plotting functions

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object
        plot_obj : matplotlib Line2D or Image
            Plot object (line or image)

        Examples
        --------
        >>> eq.plot()  # List available variables
        >>> ax, im = eq.plot('psi')  # Plot 2D variable
        >>> ax, line = eq.plot('q')  # Plot 1D profile
        """
        if name is None:
            # List available plottable variables
            plottable_2d = []
            plottable_1d = []
            
            # From structured datasets
            for var_name in self.field.data_vars:
                if self.field[var_name].ndim == 2:
                    plottable_2d.append(f"field.{var_name}")
            for var_name in self.flux.data_vars:
                if self.flux[var_name].ndim == 2:
                    plottable_2d.append(f"flux.{var_name}")
            for var_name in self.profiles.data_vars:
                if self.profiles[var_name].ndim == 1:
                    plottable_1d.append(f"profiles.{var_name}")
            
            # From legacy plot dicts (backward compatibility)
            plottable_2d.extend(self.plot_2d_names.keys())
            plottable_1d.extend(self.plot_1d_names.keys())
            
            print("Plottable 2D variables:")
            for v in sorted(set(plottable_2d)):
                print(f"  - {v}")
            print("\nPlottable 1D variables:")
            for v in sorted(set(plottable_1d)):
                print(f"  - {v}")
            return None, None
        
        # Resolve variable name from structured datasets
        resolved_var = self._resolve_plot_variable(name)
        
        if resolved_var is None:
            # Fall back to legacy plot dicts
            if name in self.plot_2d_names:
                return self.plot2d(name=name, ax=ax, put_labels=put_labels,
                                   image=line, **kwargs)
            elif name in self.plot_1d_names:
                return self.plot1d(name=name, ax=ax, put_labels=put_labels,
                                   line=line, **kwargs)
            else:
                raise ValueError(f'Cannot plot {name}: variable not found. '
                               f'Use eq.plot() to list available variables.')
        
        # Plot resolved variable
        var, is_2d = resolved_var
        if is_2d:
            return self.plot2d_var(var, name=name, ax=ax, put_labels=put_labels,
                                  image=line, **kwargs)
        else:
            return self.plot1d_var(var, name=name, ax=ax, put_labels=put_labels,
                                 line=line, **kwargs)
    
    def _resolve_plot_variable(self, name: str) -> Optional[Tuple[xr.DataArray, bool]]:
        """
        Resolve variable name to xr.DataArray from structured datasets.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'psi', 'Br', 'q', 'field.Br', 'profiles.q')

        Returns
        -------
        tuple or None
            (DataArray, is_2d) if found, None otherwise
        """
        # Try direct name first
        if name in self.field.data_vars:
            var = self.field[name]
            return (var, var.ndim == 2)
        if name in self.flux.data_vars:
            var = self.flux[name]
            return (var, var.ndim == 2)
        if name in self.profiles.data_vars:
            var = self.profiles[name]
            return (var, var.ndim == 2)
        
        # Try prefixed names (field.Br, flux.psi, profiles.q)
        if '.' in name:
            prefix, var_name = name.split('.', 1)
            if prefix == 'field' and var_name in self.field.data_vars:
                var = self.field[var_name]
                return (var, var.ndim == 2)
            if prefix == 'flux' and var_name in self.flux.data_vars:
                var = self.flux[var_name]
                return (var, var.ndim == 2)
            if prefix == 'profiles' and var_name in self.profiles.data_vars:
                var = self.profiles[var_name]
                return (var, var.ndim == 2)
        
        return None
    
    def plot1d_var(self, var: xr.DataArray, name: str, ax=None,
                   put_labels: bool=True, line=None, **kwargs):
        """
        Plot a 1D variable using xarray's plot functionality.

        Parameters
        ----------
        var : xr.DataArray
            Variable to plot (1D)
        name : str
            Variable name (for labeling)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels
        line : matplotlib Line2D, optional
            Existing line to update
        **kwargs
            Arguments passed to xarray's plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        line : matplotlib Line2D
        """
        plt = _require_matplotlib_pyplot()
        x = var[list(var.coords.keys())[0]]
        
        if line is not None:
            line.set_xdata(x.values)
            line.set_ydata(var.values)
            ax_was_none = False
        else:
            ax_was_none = ax is None
            if ax_was_none:
                fig, ax = plt.subplots(1)
            
            # Use xarray's plot if available, otherwise fall back
            try:
                line = var.plot(ax=ax, **kwargs)
                if isinstance(line, list):
                    line = line[0]
            except:
                line, = ax.plot(x.values, var.values, **kwargs)
        
        if ax_was_none and put_labels:
            fig = line.axes.figure
            x_label = x.attrs.get('short_name', x.name)
            x_units = x.attrs.get('units', '')
            y_label = var.attrs.get('short_name', var.name)
            y_units = var.attrs.get('units', '')
            ax.set_xlabel(f'{x_label} [{x_units}]' if x_units else x_label)
            ax.set_ylabel(f'{y_label} [{y_units}]' if y_units else y_label)
            ax.grid('both')
            fig.tight_layout()
        
        return ax, line
    
    def plot2d_var(self, var: xr.DataArray, name: str, ax=None,
                   put_labels: bool=True, image=None, **kwargs):
        """
        Plot a 2D variable using xarray's plot functionality.

        Parameters
        ----------
        var : xr.DataArray
            Variable to plot (2D)
        name : str
            Variable name (for labeling)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels and overlay boundary/axis
        image : matplotlib Image, optional
            Existing image to update
        **kwargs
            Arguments passed to xarray's plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        image : matplotlib Image
        """
        plt = _require_matplotlib_pyplot()
        x = var[list(var.coords.keys())[0]]
        y = var[list(var.coords.keys())[1]]
        plot_data = np.asarray(var.values).T
        extent = [x.min().values, x.max().values, y.min().values, y.max().values]
        
        if image is not None:
            # Update existing image
            image.set_data(plot_data)
            image.set_extent(extent)
            ax_was_none = False
        else:
            ax_was_none = ax is None
            if ax_was_none:
                fig, ax = plt.subplots(1)

            image = ax.imshow(plot_data, origin='lower', extent=extent, **kwargs)
            
            # Overlay boundary and axis
            if put_labels:
                ax.plot(self.geometry.R_boundary.values,
                       self.geometry.z_boundary.values, 'w-', linewidth=1.5)
                ax.plot(float(self.geometry.R_axis.values),
                       float(self.geometry.z_axis.values), 'wx', markersize=10, markeredgewidth=2)
        
        if ax_was_none and put_labels:
            fig = image.axes.figure
            x_label = x.attrs.get('short_name', x.name)
            x_units = x.attrs.get('units', '')
            y_label = y.attrs.get('short_name', y.name)
            y_units = y.attrs.get('units', '')
            ax.set_xlabel(f'{x_label} [{x_units}]' if x_units else x_label)
            ax.set_ylabel(f'{y_label} [{y_units}]' if y_units else y_label)
            
            # Add colorbar
            if hasattr(image, 'colorbar') and image.colorbar is None:
                cbar = fig.colorbar(mappable=image, ax=ax)
                z_label = var.attrs.get('short_name', var.name)
                z_units = var.attrs.get('units', '')
                cbar.set_label(f'{z_label} [{z_units}]' if z_units else z_label)
            elif not hasattr(image, 'colorbar'):
                cbar = fig.colorbar(mappable=image, ax=ax)
                z_label = var.attrs.get('short_name', var.name)
                z_units = var.attrs.get('units', '')
                cbar.set_label(f'{z_label} [{z_units}]' if z_units else z_label)
            
            ax.set_aspect('equal')
            fig.tight_layout()
        
        return ax, image

    def plot1d(self, name: str, ax=None, put_labels: bool=True,
               line=None, **kwargs):
        """
        Plot a 1D variable (explicit 1D entry point).

        This is a convenience wrapper around plot() for 1D variables.
        It maintains backward compatibility with the old API.

        Parameters
        ----------
        name : str
            Variable name to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels. Default is True
        line : matplotlib Line2D, optional
            Existing line to update
        **kwargs
            Arguments passed to matplotlib plot

        Returns
        -------
        ax : matplotlib.axes.Axes
        line : matplotlib Line2D
        """
        # Try to resolve from structured datasets first
        resolved = self._resolve_plot_variable(name)
        if resolved is not None:
            var, is_2d = resolved
            if not is_2d:
                return self.plot1d_var(var, name=name, ax=ax,
                                      put_labels=put_labels, line=line, **kwargs)
        
        # Fall back to legacy plot dicts
        if name not in self.plot_1d_names:
            raise ValueError(f'Cannot plot {name}. Variable not found. '
                           f'Use eq.plot() to list available variables.')

        var = self.plot_1d_names[name]
        return self.plot1d_var(var, name=name, ax=ax,
                              put_labels=put_labels, line=line, **kwargs)

    def plot2d(self, name: str, ax=None, put_labels: bool=True,
               image=None, **kwargs):
        """
        Plot a 2D variable (explicit 2D entry point).

        This is a convenience wrapper around plot() for 2D variables.
        It maintains backward compatibility with the old API.

        Parameters
        ----------
        name : str
            Variable name to plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        put_labels : bool, optional
            If True, add labels and overlay boundary/axis. Default is True
        image : matplotlib Image, optional
            Existing image to update
        **kwargs
            Arguments passed to matplotlib imshow/contour

        Returns
        -------
        ax : matplotlib.axes.Axes
        image : matplotlib Image
        """
        # Try to resolve from structured datasets first
        resolved = self._resolve_plot_variable(name)
        if resolved is not None:
            var, is_2d = resolved
            if is_2d:
                return self.plot2d_var(var, name=name, ax=ax,
                                      put_labels=put_labels, image=image, **kwargs)
        
        # Fall back to legacy plot dicts
        if name not in self.plot_2d_names:
            raise ValueError(f'Cannot plot {name}. Variable not found. '
                           f'Use eq.plot() to list available variables.')

        var = self.plot_2d_names[name]
        return self.plot2d_var(var, name=name, ax=ax,
                              put_labels=put_labels, image=image, **kwargs)

    def compute_coordinates(self, coordinate_system: str='boozer',
                           lpsi: int=201, ltheta: int=256,
                           dr_hr: float=1.0e-3, dz_hz: float=1.0e-3,
                           padding: float=0.05, ntht_pad: int=5,
                           rhopol_min: Optional[float]=None,
                           rhopol_max: Optional[float]=None):
        """
        Compute magnetic coordinates for the specified coordinate system.

        This is a generic method that works with any coordinate system
        registered in the Jacobian registry. The main difference between
        coordinate systems is the choice of Jacobian.

        Parameters
        ----------
        coordinate_system : str, optional
            Name of the coordinate system ('boozer', 'hamada', 'pest', etc.)
            Default is 'boozer'
        lpsi : int, optional
            Number of points along the radial direction. Default is 201
        ltheta : int, optional
            Number of points along the poloidal direction. Default is 256
        dr_hr : float, optional
            Radial step for the coordinate grid. Default is 1.0e-3
        dz_hz : float, optional
            Poloidal step for the coordinate grid. Default is 1.0e-3
        padding : float, optional
            Padding for the coordinate grid. Default is 0.05
        ntht_pad : int, optional
            Number of padding points for theta. Default is 5
        rhopol_min : float, optional
            Minimum normalized poloidal radius to include, in [0, 1].
            If provided (with or without ``rhopol_max``), this overrides
            symmetric ``padding`` behavior.
        rhopol_max : float, optional
            Maximum normalized poloidal radius to include, in [0, 1].
            If provided (with or without ``rhopol_min``), this overrides
            symmetric ``padding`` behavior.

        Returns
        -------
        MagneticCoordinates
            Magnetic coordinates object containing the transformation

        Notes
        -----
        Currently only 'boozer' is fully implemented. Other coordinate systems
        will raise NotImplementedError until their Jacobian functions are
        implemented.
        """
        # Get the Jacobian function for this coordinate system
        jacobian_func = get_jacobian_function(coordinate_system)
        
        # Build fine grid for flux surface contours
        rmin = float(self.Rgrid.values[0])
        rmax = float(self.Rgrid.values[-1])
        zmin = float(self.zgrid.values[0])
        zmax = float(self.zgrid.values[-1])

        nr_fine = int((rmax - rmin) // dr_hr)
        nz_fine = int((zmax - zmin) // dz_hz)

        R_fine = np.linspace(rmin, rmax, nr_fine)
        z_fine = np.linspace(zmin, zmax, nz_fine)

        # Evaluate on fine grid (using new structured access)
        psip = self.flux.psi.interp(R=R_fine, z=z_fine, method='cubic').values
        br_fine = self.field.Br.interp(R=R_fine, z=z_fine, method='cubic').values
        bz_fine = self.field.Bz.interp(R=R_fine, z=z_fine, method='cubic').values
        bphi_fine = self.field.Bphi.interp(R=R_fine, z=z_fine, method='cubic').values

        # Generate psi grid (using geometry attributes)
        psi_axis = float(self.geometry.attrs.get('psi_ax', self._psi_ax_init))
        psi_edge = float(self.geometry.attrs.get('psi_bdy', self._psi_edge_init))
        if rhopol_min is not None or rhopol_max is not None:
            rho_min = padding if rhopol_min is None else float(rhopol_min)
            rho_max = 1.0 - padding if rhopol_max is None else float(rhopol_max)
            if not (0.0 <= rho_min < rho_max <= 1.0):
                raise ValueError("rhopol_min/rhopol_max must satisfy 0 <= min < max <= 1.")
            eps = 1.0e-8
            rho_min = max(rho_min, eps)
            rho_max = min(rho_max, 1.0 - eps)
            if rho_max <= rho_min:
                raise ValueError("rhopol_min/rhopol_max window is too narrow after edge protection.")
            psi0 = psi_axis + rho_min**2 * (psi_edge - psi_axis)
            psi1 = psi_axis + rho_max**2 * (psi_edge - psi_axis)
        else:
            psi0 = np.amin([psi_axis, psi_edge])
            psi1 = np.amax([psi_axis, psi_edge])
            dpsi = psi1 - psi0
            psi0 += padding * dpsi
            psi1 -= padding * dpsi
        psigrid = np.linspace(psi0, psi1, lpsi)

        # Transform psigrid to radial positions at midplane (using geometry)
        R_axis_val = float(self.geometry.R_axis.values)
        z_axis_val = float(self.geometry.z_axis.values)
        R_bdy_max = float(self.geometry.R_boundary.max().values)
        Rgrid_mid = np.linspace(R_axis_val, R_bdy_max, 1000)
        psi_on_Rgrid = self.flux.psi.interp(R=Rgrid_mid,
                                            z=z_axis_val,
                                            method='cubic')
        idxsort = np.argsort(psi_on_Rgrid.values)
        frr0 = InterpolatedUnivariateSpline(psi_on_Rgrid.values[idxsort], 
                                            Rgrid_mid[idxsort])(psigrid)

        # Remove repeated values
        idx = np.where(frr0 == frr0[0])[0]
        if len(idx) > 1:
            frr0 = np.delete(frr0, idx[:-1])
            psigrid = np.delete(psigrid, idx[:-1])
        idx = np.where(frr0 == frr0[-1])[0]
        if len(idx) > 1:
            frr0 = np.delete(frr0, idx[1:])
            psigrid = np.delete(psigrid, idx[1:])

        if psi1 < psi0:
            psigrid = np.flip(psigrid, axis=0)
            frr0 = np.flip(frr0, axis=0)

        # Compute coordinates using generic function
        # Pass frr0 (radial positions at midplane) corresponding to psigrid
        out = compute_magnetic_coordinates(
            Rgrid=R_fine, zgrid=z_fine,
            br=br_fine, bz=bz_fine, bphi=bphi_fine,
            raxis=R_axis_val,
            zaxis=z_axis_val,
            psigrid=psigrid,
            ltheta=ltheta,
            phiclockwise=self.phiclockwise,
            jacobian_func=jacobian_func,
            R_at_psi=frr0,
            coordinate_system=coordinate_system,
        )

        qprof, Fprof, Iprof, thtable, nutable, jac, Rtransform, ztransform = out

        if psi1 < psi0:
            thtable = np.flip(thtable, axis=0)
            nutable = np.flip(nutable, axis=0)
            psigrid = np.flip(psigrid, axis=0)
            frr0 = np.flip(frr0, axis=0)

        # Continue with post-processing
        return self._build_magnetic_coordinates_dataset(
            psigrid, thtable, nutable, jac, Rtransform, ztransform,
            R_fine, z_fine, qprof, Fprof, Iprof, ntht_pad, coordinate_system
        )
    
    def _build_magnetic_coordinates_dataset(
        self,
        psigrid: np.ndarray,
        thtable: np.ndarray,
        nutable: np.ndarray,
        jac: np.ndarray,
        Rtransform: np.ndarray,
        ztransform: np.ndarray,
        R_fine: np.ndarray,
        z_fine: np.ndarray,
        qprof: np.ndarray,
        Fprof: np.ndarray,
        Iprof: np.ndarray,
        ntht_pad: int,
        coordinate_system: str = 'boozer'
    ) -> MagneticCoordinates:
        """
        Build the MagneticCoordinates dataset from computed coordinate arrays.

        Parameters
        ----------
        psigrid : np.ndarray
            Poloidal flux grid
        thtable : np.ndarray
            Magnetic poloidal angle table
        nutable : np.ndarray
            Magnetic toroidal angle table
        jac : np.ndarray
            Jacobian table
        Rtransform : np.ndarray
            Inverse transformation R(psi, theta)
        ztransform : np.ndarray
            Inverse transformation z(psi, theta)
        R_fine : np.ndarray
            Fine radial grid
        z_fine : np.ndarray
            Fine vertical grid
        qprof : np.ndarray
            Safety factor profile
        Fprof : np.ndarray
            F(psi) profile
        Iprof : np.ndarray
            Toroidal current profile
        ntht_pad : int
            Number of padding points for theta

        Returns
        -------
        MagneticCoordinates
            Magnetic coordinates object
        """
        ltheta = thtable.shape[1]
        
        # Build coordinate grids (using geometry for axis)
        R_axis_val = float(self.geometry.R_axis.values)
        z_axis_val = float(self.geometry.z_axis.values)
        grr, gzz = np.meshgrid(R_fine, z_fine, indexing='ij')
        thetageom = np.arctan2(gzz - z_axis_val,
                               grr - R_axis_val)
        thetageom = np.mod(thetageom + 2*np.pi, 2*np.pi)
        psirz = self.flux.psi.interp(R=R_fine, z=z_fine, method='cubic')

        # Add padding to theta grid
        thetagrid = np.linspace(0, 2*np.pi, ltheta)
        dtheta = thetagrid[1] - thetagrid[0]
        thetagrid = np.linspace(-ntht_pad*dtheta,
                                2*np.pi + ntht_pad*dtheta,
                                ltheta + 2*ntht_pad)

        # Pad thtable
        thtable_padded = thtable.copy()
        thtable_padded[:, -1] = 2*np.pi
        thtable_padded[:, 0] = 0.0
        leftside = thtable_padded[:, -ntht_pad:] - 2*np.pi
        rightside = thtable_padded[:, :ntht_pad] + 2*np.pi
        thtable_padded = np.concatenate((leftside, thtable_padded, rightside), axis=1)

        thtable_da = xr.DataArray(
            thtable_padded,
            coords={'psi0': psigrid, 'thetageom': thetagrid},
            dims=('psi0', 'thetageom'),
        )

        # Build interpolator and project to R-z plane
        thtintrp = RectBivariateSpline(psigrid, thetagrid, thtable_padded)
        thtable_Rz = thtintrp(psirz, thetageom, grid=False)
        thtable_Rz = xr.DataArray(thtable_Rz, coords=(R_fine, z_fine),
                                    dims=('R', 'z'))

        # Same for nu
        leftside = nutable[:, -ntht_pad:]
        rightside = nutable[:, :ntht_pad]
        nutable_padded = np.concatenate((leftside, nutable, rightside), axis=1)
        nutable_da = xr.DataArray(
            nutable_padded,
            coords={'psi0': psigrid, 'thetageom': thetagrid},
            dims=('psi0', 'thetageom'),
        )

        nutintrp = RectBivariateSpline(psigrid, thetagrid, nutable_padded)
        nutable_Rz = nutintrp(psirz, thetageom, grid=False)
        nutable_Rz = xr.DataArray(nutable_Rz, coords=(R_fine, z_fine),
                                    dims=('R', 'z'))

        # Project Jacobian
        leftside = jac[:, -ntht_pad:]
        rightside = jac[:, :ntht_pad]
        jacobian_padded = np.concatenate((leftside, jac, rightside), axis=1)
        jacinterp = RectBivariateSpline(psigrid, thetagrid, jacobian_padded)
        jac_Rz = jacinterp(psirz, thetageom, grid=False)

        # Compute derivatives
        d_dr = FinDiff(0, R_fine[1] - R_fine[0], 1, acc=4)
        d_dz = FinDiff(0, z_fine[1] - z_fine[0], 1, acc=4)

        dPsi_dr = d_dr(psirz)
        dPsi_dz = d_dz(psirz)
        dPsi_dphi = np.zeros_like(psirz)

        dTheta_dr = d_dr(thtable_Rz)
        dTheta_dz = d_dz(thtable_Rz)
        dTheta_dphi = np.zeros_like(thtable_Rz)

        dzeta_dr = d_dr(nutable_Rz)
        dzeta_dz = d_dz(nutable_Rz)
        dzeta_dphi = np.ones_like(nutable_Rz)

        # Inverse transformation derivatives
        dR_dpsi = dTheta_dz / jac_Rz
        dR_dtheta = -dPsi_dz / jac_Rz
        dR_dzeta = np.zeros_like(dR_dpsi)

        dz_dpsi = -dTheta_dr / jac_Rz
        dz_dtheta = dPsi_dr / jac_Rz
        dz_dzeta = np.zeros_like(dz_dpsi)

        dphi_dpsi = (dTheta_dr * dzeta_dz - dTheta_dz * dzeta_dr) / jac_Rz
        dphi_dtheta = (dPsi_dz * dzeta_dr - dPsi_dr * dzeta_dz) / jac_Rz
        dphi_dzeta = (dTheta_dz * dPsi_dr - dTheta_dr * dPsi_dz) / jac_Rz + 1.0

        # Build coordinate dataset
        magcoords = xr.Dataset()
        magcoords['psi'] = xr.DataArray(psirz, dims=('R', 'z'),
                                        coords={'R': R_fine, 'z': z_fine},
                                        attrs={'name': 'psi', 'units': 'Wb',
                                               'desc': 'Poloidal flux',
                                               'short_name': '$\\Psi$'})
        magcoords['theta'] = thtable_da
        magcoords['theta'].attrs = {'name': 'theta', 'units': 'rad',
                                    'desc': 'Magnetic poloidal angle',
                                    'short_name': '$\\Theta*$'}
        magcoords['nu'] = nutable_da
        magcoords['nu'].attrs = {'name': 'nu', 'units': 'rad',
                                  'desc': 'Magnetic toroidal angle',
                                  'short_name': '$\\nu$'}

        magcoords.R.attrs = {'name': 'R', 'units': 'm', 'desc': 'Major radius',
                             'short_name': 'R'}
        magcoords.z.attrs = {'name': 'z', 'units': 'm', 'desc': 'Height',
                             'short_name': 'z'}
        magcoords.psi0.attrs = {'name': 'psi0', 'units': 'Wb',
                                'desc': 'Poloidal flux at the magnetic axis',
                                'short_name': '$\\Psi_0$'}
        magcoords.thetageom.attrs = {'name': 'thetageom', 'units': 'rad',
                                     'desc': 'Geometrical poloidal angle',
                                     'short_name': '$\\Theta_{geom}$'}

        # Build derivatives dataset
        magdevs = xr.Dataset()
        magdevs['jacobian'] = xr.DataArray(jac_Rz, dims=('R', 'z'),
                                           coords={'R': R_fine, 'z': z_fine},
                                           attrs={'name': 'jacobian', 'units': '',
                                                  'desc': 'Jacobian of the transformation',
                                                  'short_name': '$\\mathcal{J}$'})
        
        # Add all derivative arrays with proper attributes
        derivatives = {
            'dR_dpsi': (dR_dpsi, 'm/Wb', 'Partial derivative of R with respect to poloidal flux'),
            'dR_dtheta': (dR_dtheta, 'm/rad', 'Partial derivative of R with respect to magnetic poloidal angle'),
            'dR_dzeta': (dR_dzeta, 'm/rad', 'Partial derivative of R with respect to magnetic toroidal angle'),
            'dz_dpsi': (dz_dpsi, 'm/Wb', 'Partial derivative of z with respect to poloidal flux'),
            'dz_dtheta': (dz_dtheta, 'm/rad', 'Partial derivative of z with respect to magnetic poloidal angle'),
            'dz_dzeta': (dz_dzeta, 'm/rad', 'Partial derivative of z with respect to magnetic toroidal angle'),
            'dphi_dpsi': (dphi_dpsi, 'rad/Wb', 'Partial derivative of phi with respect to poloidal flux'),
            'dphi_dtheta': (dphi_dtheta, 'rad/rad', 'Partial derivative of phi with respect to magnetic poloidal angle'),
            'dphi_dzeta': (dphi_dzeta, 'rad/rad', 'Partial derivative of phi with respect to magnetic toroidal angle'),
            'dPsi_dr': (dPsi_dr, 'Wb/m', 'Partial derivative of poloidal flux with respect to R'),
            'dPsi_dz': (dPsi_dz, 'Wb/m', 'Partial derivative of poloidal flux with respect to z'),
            'dPsi_dphi': (dPsi_dphi, 'Wb/rad', 'Partial derivative of poloidal flux with respect to phi'),
            'dTheta_dr': (dTheta_dr, 'rad/m', 'Partial derivative of magnetic poloidal angle with respect to R'),
            'dTheta_dz': (dTheta_dz, 'rad/m', 'Partial derivative of magnetic poloidal angle with respect to z'),
            'dTheta_dphi': (dTheta_dphi, 'rad/rad', 'Partial derivative of magnetic poloidal angle with respect to phi'),
            'dzeta_dr': (dzeta_dr, 'rad/m', 'Partial derivative of magnetic toroidal angle with respect to R'),
            'dzeta_dz': (dzeta_dz, 'rad/m', 'Partial derivative of magnetic toroidal angle with respect to z'),
            'dzeta_dphi': (dzeta_dphi, 'rad/rad', 'Partial derivative of magnetic toroidal angle with respect to phi'),
        }
        
        for name, (data, units, desc) in derivatives.items():
            short_name = name.replace("_", " / \\partial ")
            magdevs[name] = xr.DataArray(data, dims=('R', 'z'),
                                        coords={'R': R_fine, 'z': z_fine},
                                        attrs={'name': name, 'units': units,
                                               'desc': desc,
                                               'short_name': f'$\\partial {short_name}$'})

        magdevs.R.attrs = {'name': 'R', 'units': 'm', 'desc': 'Major radius',
                           'short_name': 'R'}
        magdevs.z.attrs = {'name': 'z', 'units': 'm', 'desc': 'Height',
                           'short_name': 'z'}

        # Add inverse transformation
        leftside = Rtransform[:, -ntht_pad:]
        rightside = Rtransform[:, :ntht_pad]
        Rtransform_padded = np.concatenate((leftside, Rtransform, rightside), axis=1)
        magcoords['R_inv'] = xr.DataArray(Rtransform_padded,
                                          dims=('psi0', 'theta_star'),
                                          coords={'psi0': psigrid,
                                                  'theta_star': thetagrid},
                                          attrs={'name': 'R_inv',
                                                 'desc': 'R = R(psi, theta*)',
                                                 'units': 'm',
                                                 'short_name': '$R(\\Psi, \\Theta^*)$'})
        
        leftside = ztransform[:, -ntht_pad:]
        rightside = ztransform[:, :ntht_pad]
        ztransform_padded = np.concatenate((leftside, ztransform, rightside), axis=1)
        magcoords['z_inv'] = xr.DataArray(ztransform_padded,
                                          dims=('psi0', 'theta_star'),
                                          coords={'psi0': psigrid,
                                                  'theta_star': thetagrid},
                                          attrs={'name': 'z_inv',
                                                 'desc': 'z = z(psi, theta*)',
                                                 'units': 'm',
                                                 'short_name': '$z(\\Psi, \\Theta^*)$'})

        magcoords.theta_star.attrs = {'name': 'theta_star', 'units': 'rad',
                                      'desc': 'Magnetic poloidal angle',
                                      'short_name': '$\\Theta^*$'}

        # Save profiles
        self.boozer_profs = xr.Dataset()
        self.boozer_profs['q'] = xr.DataArray(qprof, dims=('psi0',),
                                              coords=(psigrid,),
                                              attrs={'name': 'q', 'units': '',
                                                     'desc': 'Safety factor',
                                                     'short_name': '$q$'})
        Iprof *= 2*np.pi/(4.0*np.pi * 1e-7)
        self.boozer_profs['I'] = xr.DataArray(Iprof, dims=('psi0',),
                                              coords=(psigrid,),
                                              attrs={'name': 'I', 'units': 'A',
                                                     'desc': 'Toroidal current',
                                                     'short_name': '$I$'})
        self.boozer_profs['F'] = xr.DataArray(Fprof, dims=('psi0',),
                                              coords=(psigrid,),
                                              attrs={'name': 'F', 'units': 'T*m',
                                                     'desc': 'F(psi) function in GS equation = RB_T',
                                                     'short_name': '$F$'})

        # Map regions outside LCFS to NaN
        rhoprz = self.flux.rho.interp(R=R_fine, z=z_fine, method='cubic')
        flags = rhoprz > 1.0

        # Store computed coordinates for easy access
        mag_coords_obj = MagneticCoordinates(magcoords, magdevs,
                                              Raxis=R_axis_val,
                                              zaxis=z_axis_val,
                                              pad=ntht_pad)
        
        # Cache the result for easy access
        if not hasattr(self, '_magnetic_coordinates_cache'):
            self._magnetic_coordinates_cache = {}
        coord_sys_lower = coordinate_system.lower()
        self._magnetic_coordinates_cache[coord_sys_lower] = mag_coords_obj
        
        # Also set as main attribute if this is the first/only coordinate system
        self.magnetic_coordinates = mag_coords_obj
        self.coord_sys = coordinate_system.lower()
        
        return mag_coords_obj
    
    @property
    def coords(self) -> Dict[str, MagneticCoordinates]:
        """
        Dictionary of computed magnetic coordinate systems.
        
        Returns
        -------
        dict
            Dictionary mapping coordinate system names to MagneticCoordinates objects
            
        Examples
        --------
        >>> eq.compute_coordinates('boozer')
        >>> eq.compute_coordinates('hamada')
        >>> eq.coords['boozer']  # Access Boozer coordinates
        >>> eq.coords['hamada']  # Access Hamada coordinates
        """
        if not hasattr(self, '_magnetic_coordinates_cache'):
            self._magnetic_coordinates_cache = {}
        return self._magnetic_coordinates_cache

    def to_geqdsk(
        self,
        filename: str,
        cocos: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Save the equilibrium to a g-EQDSK file.

        When the internal object has not a clearly defined COCOS
        convention, this will try to change it to COCOS=1 by default.

        Parameters
        ----------
        filename : str
            Name of the output file
        cocos : int, optional
            COCOS convention to use. If None, uses COCOS=1. Default is None

        Returns
        -------
        dict
            Dictionary containing the g-EQDSK data that was written

        Raises
        ------
        ValueError
            If file already exists
        """

        if os.path.isfile(filename):
            raise ValueError(f'File {filename} already exists.' + 
                              'Please remove it before saving.')
        
        # Before saving the gdata.

        # Creating the gdata dictionary.
        gdata = dict()
        gdata['nx'] = self.Rgrid.size
        gdata['ny'] = self.zgrid.size
        gdata['rmagx'] = float(self.geometry.R_axis.values)
        gdata['zmagx'] = float(self.geometry.z_axis.values)
        gdata['bcentr'] = float(self.Bdata.Baxis.values)
        gdata['rleft'] = float(self.Rgrid[0].values)
        gdata['zmid'] = float(np.nanmean(self.zgrid.values))
        gdata['rdim'] = float(self.Rgrid[-1].values - self.Rgrid[0].values)
        gdata['zdim'] = float(self.zgrid[-1].values - self.zgrid[0].values)

        # Saving the Psi value.
        psip = self.flux.psi.values
        gdata['psi'] = psip
        gdata['simagx'] = float(self.geometry.attrs.get('psi_ax'))
        gdata['sibdry'] = float(self.geometry.attrs.get('psi_bdy'))
        gdata['nbdry'] = 1
        gdata['nlim'] = 1
        gdata['rbdry'] = self.geometry.R_boundary.values
        gdata['zbdry'] = self.geometry.z_boundary.values

        # We need now to get the fpol, pressure and current.
        # Getting the magnetic coordinates.
        cache = getattr(self, '_magnetic_coordinates_cache', {})
        if 'boozer' not in cache:
            self.compute_coordinates(coordinate_system='boozer', lpsi=501) # We need to compute the Boozer transformation.

        psi_ax = float(self.geometry.attrs.get('psi_ax'))
        psi_bdy = float(self.geometry.attrs.get('psi_bdy'))
        psiN = psi_bdy - psi_ax
        dpsi_nominal = psiN / (gdata['nx'] - 1)
        nx_2 = int((self.flux.rho.max().values**2 * psiN)/dpsi_nominal) + 1

        rhopol_integral = np.linspace(0, float(self.flux.rho.max().values), nx_2)
        psi_integral = rhopol_integral**2 * psiN + psi_ax
        fpol_full = np.zeros_like(rhopol_integral)

        flags_extrapolate = np.zeros_like(rhopol_integral, dtype=bool)

        # We need to manually recompute the value of the function F:
        for ii, irhop in enumerate(rhopol_integral):
            R, z = self.rhopol2rz(irhop)
            R = R[0]
            z = z[0]

            if R is None:
                flags_extrapolate[ii] = True
                continue

            Bvalue = self.getB(R, z, grid=False)

            # We need to interpolate the pressure gradient.
            fpol_full[ii] = np.nanmean(R * Bvalue.Bphi.values)
        
        # We need to extrapolate to the values that the algorithm could not
        # find.
        fpol_full[flags_extrapolate] = np.interp(rhopol_integral[flags_extrapolate],
                                                 rhopol_integral[~flags_extrapolate],
                                                 fpol_full[~flags_extrapolate])
        
        

        # From the Boozer transformation, we can get the fpol, pressure and current.
        psi1d = np.linspace(psi_ax, psi_bdy, gdata['nx'])
        psiN = psi_bdy - psi_ax
        fpol = InterpolatedUnivariateSpline(psi_integral, fpol_full, k=3)(psi1d)
        gdata['fpol'] = fpol

        # Getting the derivative.
        d_dpsi = FinDiff(0, 1, acc=4)
        ffprime = d_dpsi(fpol_full, psi_integral) * fpol_full
        idx = np.where(flags_extrapolate[:nx_2//2])[0][-1] + 3
        gdata['ffprime'] = InterpolatedUnivariateSpline(psi_integral[idx:], 
                                                        ffprime[idx:], k=1)(psi1d)

        # From the Grad-Shafranov equation, one can get the pressure
        # gradient in magnetic coordinates.
        d_dR = FinDiff(0, self.Rgrid.values[1] - self.Rgrid.values[0], 1, acc=4)
        d_dz = FinDiff(1, self.zgrid.values[1] - self.zgrid.values[0], 1, acc=4)

        tmp1 = d_dR(self.Bdata.Bz.values) / self.Rgrid.values[:, None]
        tmp2 = d_dz(self.Bdata.Br.values) / self.Rgrid.values[:, None]

        # \\mu_0 pprime = - [ FF'/R^2 + dB_z/dR / R - dB_R/dz / R ]
        ffprime_rz = InterpolatedUnivariateSpline(psi_integral[idx:], 
                                                  ffprime[idx:], ext=3,
                                                  k=3)(self.fluxdata.psipol.values)
        pprime_rz = - (ffprime_rz / self.Rgrid.values[:, None]**2 - tmp1 + tmp2)

        gdata['pprime_rz'] = pprime_rz
        gdata['ffprime_rz'] = ffprime_rz

        # We need now the average pressure gradient in the flux surface.
        pprime = np.zeros_like(rhopol_integral)
        ffprime_check = np.zeros_like(rhopol_integral)

        pprime_intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                             pprime_rz)
        ffprime_rz_intrp = RectBivariateSpline(self.Rgrid.values, self.zgrid.values,
                                               ffprime_rz)
        
        for ii, irhop in enumerate(rhopol_integral):
            R, z = self.rhopol2rz(irhop)
            R = R[0]
            z = z[0]

            if R is None:
                pprime[ii] = 0
                ffprime_check[ii] = 0
                continue

            # We need to interpolate the pressure gradient.
            tmp = pprime_intrp(R, z, grid=False)

            thetaval = np.arctan2(z - float(self.geometry.z_axis.values), 
                                  R - float(self.geometry.R_axis.values))
            # plt.plot(thetaval, tmp)
            pprime[ii] = np.nanmean(tmp)
            ffprime_check[ii] = np.nanmean(ffprime_rz_intrp(R, z, grid=False))

        
        gdata['pprime'] = InterpolatedUnivariateSpline(psi_integral, pprime,
                                                       k=3)(psi1d) / (4*np.pi * 1e-7)
        gdata['ffprime_check'] = ffprime_check

        # plt.plot(psi_integral, pprime, 'r-')
        # plt.plot(self._gdata['psi_1d'], self._gdata['pprime'], 'b--')

        # To get now the total pressure, we need to integrate the pprime
        # along the magnetic flux. Since we don't have any indication on
        # boundary condition, we will consider that for the largest value
        # of rhopol, the pressure is zero.
        ptotal = np.zeros_like(rhopol_integral)
        psi_integral = rhopol_integral**2 * psiN + psi_ax
        ptotal[-1] = 0.0

        for ii in range(len(rhopol_integral) - 2, -1, -1):
            ptotal[ii] = ptotal[ii+1] + pprime[ii] * (psi_integral[ii+1] - psi_integral[ii])

        gdata['ptotal'] = InterpolatedUnivariateSpline(psi_integral, ptotal, k=3)(psi1d)


        return gdata