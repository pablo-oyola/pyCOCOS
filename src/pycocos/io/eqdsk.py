"""
EQDSK library.

This library reads and parses the equilibrium file in the so-called EQDSK and
parses the COCOS standard.
"""

import copy
import os
from typing import Any, Dict, Optional

import freeqdsk
import numpy as np
import xarray as xr
from findiff import FinDiff
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline, interp1d

# Import from new structure
from ..core.equilibrium import equilibrium

from .cocos import FluxNormalization, fromCocosNtoCocosM, identify_cocos
from .cocos import cocos as get_cocos

__mapping = (('hdr', 'comment'),   ('Bcenter', 'bcentr'),  ('Ip', 'cpasma'),
             ('lr', 'nx'),         ('lz', 'ny'),           ('r_bdy', 'rbdry'),
             ('z_bdy', 'zbdry'),   ('Raxis', 'rmagx'),     ('zaxis', 'zmagx'),
             ('psi', 'psi'),       ('psi_ax', 'simagx'),   ('psi_bdy', 'sibdry'),
             ('psimax', 'sibdry'), ('fpol', 'fpol'),       ('prs', 'pres'),
             ('ffp', 'ffprime'),   ('pprime', 'pprime'),    ('q', 'qpsi'))


# -----------------------------------------------------------------------------
# ROUTINES TO READ THE EQDSK.
# -----------------------------------------------------------------------------
def _representative_q(qpsi: Any) -> float:
    """Return a finite, nonzero q value after checking sign consistency."""
    try:
        values = np.asarray(qpsi, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise TypeError("qpsi must contain real values") from exc

    usable = values[np.isfinite(values) & (values != 0.0)]
    if usable.size == 0:
        raise ValueError("qpsi must contain a finite, nonzero value")
    if np.any(np.sign(usable) != np.sign(usable[0])):
        raise ValueError("qpsi contains mixed nonzero signs")
    return float(usable[0])


def _resolve_loader_input_convention(
    raw: Dict[str, Any],
    cocos_in: Optional[int],
    cocos_internal: int,
    phiclockwise_in: Optional[bool],
    flux_normalization: Optional[FluxNormalization],
) -> Dict[str, Any]:
    """Resolve the source and stored COCOS conventions exactly once."""
    if phiclockwise_in is not None and not isinstance(
        phiclockwise_in,
        (bool, np.bool_),
    ):
        raise TypeError("phiclockwise_in must be a boolean or None")
    if flux_normalization not in (None, "Wb", "Wb/rad"):
        raise ValueError("flux_normalization must be either 'Wb' or 'Wb/rad'")

    internal_descriptor = get_cocos(cocos_internal)

    if cocos_in is not None:
        input_descriptor = get_cocos(cocos_in)
        expected_phiclockwise = input_descriptor.phiclockwise
        expected_flux_normalization = input_descriptor.flux_normalization

        if (
            phiclockwise_in is not None
            and bool(phiclockwise_in) != expected_phiclockwise
        ):
            raise ValueError(
                f"COCOS {cocos_in} requires phiclockwise_in="
                f"{expected_phiclockwise}, got {phiclockwise_in}"
            )
        if (
            flux_normalization is not None
            and flux_normalization != expected_flux_normalization
        ):
            raise ValueError(
                f"COCOS {cocos_in} requires flux_normalization="
                f"'{expected_flux_normalization}', got '{flux_normalization}'"
            )
    else:
        resolution = identify_cocos(
            _representative_q(raw['qpsi']),
            raw['cpasma'],
            raw['bcentr'],
            raw['simagx'],
            raw['sibdry'],
            phiclockwise=phiclockwise_in,
            flux_normalization=flux_normalization,
        )
        if not resolution.is_unique:
            raise ValueError(
                "Cannot uniquely determine the input COCOS from this EQDSK; "
                f"candidates are {resolution.candidates}. Pass cocos_in, or "
                "supply phiclockwise_in and flux_normalization."
            )
        cocos_in = resolution.require_unique()
        input_descriptor = get_cocos(cocos_in)
        expected_phiclockwise = input_descriptor.phiclockwise
        expected_flux_normalization = input_descriptor.flux_normalization

    return {
        'cocos_input': cocos_in,
        'cocos_internal': cocos_internal,
        'phiclockwise_input': expected_phiclockwise,
        'phiclockwise_internal': internal_descriptor.phiclockwise,
        'flux_normalization_input': expected_flux_normalization,
        'flux_normalization_internal': internal_descriptor.flux_normalization,
    }


def read_eqdsk(
    filename: str,
    cocos_in: Optional[int] = None,
    cocos_internal: int = 1,
    phiclockwise_in: Optional[bool] = None,
    flux_normalization: Optional[FluxNormalization] = None,
) -> Dict[str, Any]:
    """
    Read an EQDSK file using the freeqdsk library.

    This function reads a g-EQDSK file, resolves its input convention exactly
    once, and converts it exactly once to the requested internal convention.

    Parameters
    ----------
    filename : str
        Path to the EQDSK file
    cocos_in : int, optional
        Explicit COCOS convention of the input file. When supplied, detection
        is skipped and orientation/normalization are derived from this value.
    cocos_internal : int, optional
        COCOS convention used by the returned arrays. Default is 1.
    phiclockwise_in : bool, optional
        Input toroidal-angle orientation, required for detection when
        ``cocos_in`` is omitted.
    flux_normalization : {"Wb", "Wb/rad"}, optional
        Input poloidal-flux normalization, required for detection when
        ``cocos_in`` is omitted.

    Returns
    -------
    dict
        Dictionary containing equilibrium data with keys:
        - 'Rgrid', 'zgrid': Grid coordinates
        - 'psi': Poloidal flux (2D array)
        - 'fpol', 'pres', 'q': Profiles
        - 'Raxis', 'zaxis': Magnetic axis position
        - 'psi_ax', 'psi_bdy': Flux values
        - And other EQDSK quantities

    Raises
    ------
    FileNotFoundError
        If the file cannot be found

    Examples
    --------
    >>> data = read_eqdsk('equilibrium.geqdsk', cocos_in=1, cocos_internal=1)
    """
    with open(filename, 'r') as f:
        d = freeqdsk.geqdsk.read(f)

    convention = _resolve_loader_input_convention(
        d,
        cocos_in=cocos_in,
        cocos_internal=cocos_internal,
        phiclockwise_in=phiclockwise_in,
        flux_normalization=flux_normalization,
    )
    cocos_input = convention['cocos_input']

    d = fromCocosNtoCocosM(
        d,
        cocos_m=cocos_internal,
        cocos_n=cocos_input,
    )

    # We need now to transform the generated dictionary and 
    # transform it to the standard that is used in this library.
    from_freeqdsk = [ii[1] for ii in __mapping]
    to_mega = [ii[0] for ii in __mapping]
    output = dict()
    
    for ikey in d:
        if ikey in from_freeqdsk:
            idx = from_freeqdsk.index(ikey)
            output[to_mega[idx]] = d[ikey]
        else:
            output[ikey] = d[ikey]

    # We mapped everything and now we need to build other profiles.
    rleft = output.pop('rleft')
    rdim  = output.pop('rdim')
    zmid  = output.pop('zmid')
    zdim  = output.pop('zdim')
    output['Rgrid']   = np.linspace(rleft, rleft + rdim, output['lr'])
    output['zgrid']   = np.linspace(zmid - 0.5*zdim, zmid + 0.5*zdim, output['lz'])
    output['psimax']  = output['psi_bdy'] - output['psi_ax']
    output['dpsi']    = np.abs(output['psimax'])/(output['lr']-1)
    output['psirz']   = output['psi'] - output['psi_ax']
    output['rhoprz']  = np.sqrt(output['psirz']/output['psimax'])

    # --- Making the flux quantities to the grid.
    psi_1d = np.linspace(0.0, output['psimax'], num=output['lr'])
    output['psi_1d'] = psi_1d
    output['rhop_1d']  = np.sqrt(psi_1d/output['psimax'])

    # flags = output['psirz'] < output['psimax']
    flags = np.ones(output['psirz'].shape, dtype=bool)
    output['fpolrz'] = np.zeros_like(output['psirz'])
    intrp = interp1d(psi_1d, output['fpol'], kind='linear',
                     bounds_error=False, fill_value=(output['fpol'][-1], output['fpol'][-1]))
    output['fpolrz'][flags] = intrp(output['psirz'][flags])
    
    # We use linear interpolation on the edges
    flags = np.logical_not(flags)
    output['fpolrz'][flags] = output['fpol'][-1]

    output['prsrz'] = interp1d(psi_1d, output['prs'], kind='linear',
                                bounds_error=False, fill_value=0.0)
    output['prsrz'] = output['prsrz'](output['psirz'])
    output.update(convention)

    return output

def eqdsk2magnetic(eqdata: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Compute magnetic field components from EQDSK flux data.

    Uses finite differences to compute Br, Bz from the poloidal flux,
    and Bphi from the fpol profile.

    Parameters
    ----------
    eqdata : dict
        Dictionary containing equilibrium data with keys:
        - 'psi': Poloidal flux (2D array)
        - 'fpolrz': F(psi) function on R-z grid
        - 'Rgrid', 'zgrid': Grid coordinates
        - 'cocos_internal': COCOS convention of the supplied arrays

    Returns
    -------
    dict
        Dictionary containing:
        - 'br': Radial magnetic field component
        - 'bz': Vertical magnetic field component
        - 'bphi': Toroidal magnetic field component
        - 'btht': Poloidal magnetic field magnitude
        - 'babs': Total magnetic field magnitude

    Raises
    ------
    Exception
        If required keys are missing from eqdata
    """
    if 'psi' not in eqdata:
        raise Exception('The input data must contain the magnetic flux surfaces')

    if 'Rgrid' not in eqdata:
        raise Exception('The grids must be in the input data')

    if 'zgrid' not in eqdata:
        raise Exception('The grids must be in the input data')

    if 'fpolrz' not in eqdata:
        raise Exception('The fpolrz must be within the input data.')

    if 'cocos_internal' not in eqdata:
        raise Exception('The internal COCOS convention must be within the input data.')
    cocos_id = eqdata['cocos_internal']
    cocos_info = get_cocos(cocos_id)


    output = dict()
    # To compute the poloidal magnetic field, we use the psi (poloidal
    # flux surfaces and 4th order finite differences.
    dr = np.abs(eqdata['Rgrid'][1] - eqdata['Rgrid'][0])
    dz = np.abs(eqdata['zgrid'][1] - eqdata['zgrid'][0])

    Rmesh, _ = np.meshgrid(eqdata['Rgrid'], eqdata['zgrid'])

    d_dr = FinDiff(0, dr, 1, acc=4)
    d_dz = FinDiff(1, dz, 1, acc=4)

    # COCOS 11-18 store the full poloidal flux in Wb, whereas COCOS 1-8
    # store Phi_pol/(2*pi). Convert the derivative normalization while applying
    # the COCOS poloidal-field and toroidal-coordinate orientation signs.
    bpol_prefactor = (
        cocos_info.sigma_Bp
        * cocos_info.sigma_RpZ
        / (2.0 * np.pi) ** cocos_info.exp_Bp
    )

    output['br'] = + d_dz(eqdata['psi'])/Rmesh.T * bpol_prefactor
    output['bz'] = - d_dr(eqdata['psi'])/Rmesh.T * bpol_prefactor

    output['btht'] = np.sqrt(output['br']**2.0 + output['bz']**2.0)
    # Now we provide the sign to the poloidal magnetic field.
    q0 = eqdata['q'][0]

    output['btht'] *= q0/np.abs(q0)

    # We know compute the toroidal magnetic field using the fpol = R*Bphi
    # function.
    output['bphi'] = eqdata['fpolrz']/Rmesh.T
    output['babs'] = np.sqrt(output['bphi']**2.0 + output['btht']**2.0)

    return output


# ----------------------------------------------------------------------------
# CLASS FOR THE EQDSK.
# ----------------------------------------------------------------------------
class eqdsk(equilibrium):
    """
    EQDSK file handler class.

    This class extends the equilibrium class to load equilibrium data
    from g-EQDSK files with explicit or one-time COCOS detection and a single
    conversion to the requested internal convention.

    Parameters
    ----------
    fn : str
        Filename of the EQDSK file
    cocos_in : int, optional
        Explicit input COCOS convention.
    cocos_internal : int, optional
        COCOS convention used by the stored arrays. Default is 1.
    phiclockwise_in : bool, optional
        Input toroidal-angle orientation used only for detection.
    flux_normalization : {"Wb", "Wb/rad"}, optional
        Input poloidal-flux normalization used only for detection.

    Attributes
    ----------
    filename : str
        Path to the EQDSK file
    gs_profs : xr.Dataset
        Grad-Shafranov profiles (fpol, pres, q, etc.)

    Examples
    --------
    >>> from pycocos import EQDSK
    >>> eq = EQDSK('equilibrium.geqdsk', cocos_in=1, cocos_internal=1)
    >>> print(eq.gs_profs.q)  # Access q-profile
    """

    def __init__(
        self,
        fn: str,
        cocos_in: Optional[int] = None,
        cocos_internal: int = 1,
        phiclockwise_in: Optional[bool] = None,
        flux_normalization: Optional[FluxNormalization] = None,
    ) -> None:
        """
        Create an Equilibrium object starting from an EQDSK file.

        Parameters
        ----------
        fn : str
            Filename of the EQDSK file
        cocos_in : int, optional
            Explicit COCOS convention of the input file.
        cocos_internal : int, optional
            COCOS convention used by the stored arrays. Default is 1.
        phiclockwise_in : bool, optional
            Input toroidal-angle orientation used for detection.
        flux_normalization : {"Wb", "Wb/rad"}, optional
            Input poloidal-flux normalization used for detection.

        Raises
        ------
        FileNotFoundError
            If the EQDSK file cannot be found
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError(f'Cannot locate the file {fn}.')
        
        self.filename = fn

        self._gdata = read_eqdsk(
            filename=fn,
            cocos_in=cocos_in,
            cocos_internal=cocos_internal,
            phiclockwise_in=phiclockwise_in,
            flux_normalization=flux_normalization,
        )
        self._bfield = eqdsk2magnetic(self._gdata)

        self._cocos_input = self._gdata['cocos_input']
        self._cocos_internal = self._gdata['cocos_internal']
        self._phiclockwise_input = self._gdata['phiclockwise_input']
        self._phiclockwise_internal = self._gdata['phiclockwise_internal']
        self._flux_normalization_input = self._gdata['flux_normalization_input']
        self._flux_normalization_internal = self._gdata['flux_normalization_internal']

        # Using the parent class to perform the hard initializing.
        super().__init__(self._gdata['Rgrid'],   self._gdata['zgrid'],
                         self._bfield['br'],     self._bfield['bz'],
                         self._bfield['bphi'],   self._gdata['psi'],
                         self._gdata['Raxis'],   self._gdata['zaxis'],
                         self._gdata['psi_bdy'], self._gdata['psi_ax'],
                         phiclockwise=self._phiclockwise_internal,
                         flux_normalization=self._flux_normalization_internal)

        # Populate profiles in the structured data
        self._populate_profiles()
        
        # Keep gs_profs for backward compatibility (alias to _profiles)
        self.gs_profs = self._profiles
        
        # Add variables to plotting registry (backward compatibility)
        for var_name in self._profiles.data_vars:
            if var_name not in ['psi', 'rho']:  # Skip coordinate arrays
                self.add_var(var_name, self._profiles[var_name])
        
        # Add 2D profile fields if available
        if 'fpolrz' in self._gdata:
            _fpolrz = xr.DataArray(self._gdata['fpolrz'], dims=('R', 'z'),
                                    coords=(self.Rgrid, self.zgrid),
                                    attrs={ 'name': 'fpolrz',
                                            'units': r'$T\cdot m$',
                                            'desc': r'$RB_\phi$',
                                            'short_name': r'$RB_\phi$'
                                         })
            self.add_var('fpolrz', _fpolrz)
        
        if 'prsrz' in self._gdata:
            _prsrz = xr.DataArray(self._gdata['prsrz'], dims=('R', 'z'),
                                    coords=(self.Rgrid, self.zgrid),
                                    attrs={ 'name': 'prsrz',
                                            'units': 'Pa',
                                            'desc': 'Plasma pressure',
                                            'short_name': 'p'
                                         })
            self.add_var('prsrz', _prsrz)
    
    def _populate_profiles(self) -> None:
        """Populate the profiles dataset from EQDSK data."""
        tmp = np.linspace(0.0, self._gdata['psimax'], num=self._gdata['lr'])
        flux_units = self._flux_normalization_internal
        if flux_units == "Wb/rad":
            ffprime_units = "T^2*m^4*rad/Wb"
            pprime_units = "Pa*rad/Wb"
        else:
            ffprime_units = "T^2*m^4/Wb"
            pprime_units = "Pa/Wb"
        _psi1d = xr.DataArray(tmp, dims=('rhop',),
                             attrs={'name': 'psi', 'units': flux_units,
                                    'desc': 'Magnetic flux',
                                    'short_name': '$\\Psi$'})
        _rho1d = xr.DataArray(np.sqrt(tmp/self._gdata['psimax']),
                             dims=('rhop',),
                             attrs={'name': 'rhop', 'units': '',
                                    'desc': 'Radial magnetic coord.',
                                    'short_name': '$\\rho_{pol}$'})

        # Add profiles to structured _profiles dataset
        if 'fpol' in self._gdata:
            self._profiles['fpol'] = xr.DataArray(self._gdata['fpol'],
                                                  dims=('rhop',),
                                                  coords={'rhop': _rho1d},
                                                  attrs={'name': 'fpol',
                                                         'units': 'T*m',
                                                         'desc': 'F(psi) = RB_phi',
                                                         'short_name': '$F$'})
        
        if 'prs' in self._gdata:
            self._profiles['pres'] = xr.DataArray(self._gdata['prs'],
                                                 dims=('rhop',),
                                                 coords={'rhop': _rho1d},
                                                 attrs={'name': 'pres',
                                                        'units': 'Pa',
                                                        'desc': 'Plasma pressure',
                                                        'short_name': '$p$'})
        
        if 'ffp' in self._gdata:
            self._profiles['ffprime'] = xr.DataArray(self._gdata['ffp'],
                                                    dims=('rhop',),
                                                    coords={'rhop': _rho1d},
                                                    attrs={'name': 'ffprime',
                                                           'units': ffprime_units,
                                                           'desc': 'd(F*F)/dPsi',
                                                           'short_name': '$FF\'$'})
        
        if 'pprime' in self._gdata:
            self._profiles['pprime'] = xr.DataArray(self._gdata['pprime'],
                                                    dims=('rhop',),
                                                    coords={'rhop': _rho1d},
                                                    attrs={'name': 'pprime',
                                                           'units': pprime_units,
                                                           'desc': 'dp/dPsi',
                                                           'short_name': '$p\'$'})
        
        if 'q' in self._gdata:
            self._profiles['q'] = xr.DataArray(self._gdata['q'],
                                              dims=('rhop',),
                                              coords={'rhop': _rho1d},
                                              attrs={'name': 'q',
                                                     'units': '',
                                                     'desc': 'Safety factor',
                                                     'short_name': '$q$'})
        
        # Store coordinate arrays
        self._profiles['psi'] = _psi1d
        self._profiles['rho'] = _rho1d
    
    @property
    def cocos_input(self) -> int:
        """COCOS convention of the source EQDSK file."""
        return self._cocos_input

    @property
    def cocos_internal(self) -> int:
        """COCOS convention of the arrays stored by this object."""
        return self._cocos_internal

    @property
    def phiclockwise_input(self) -> bool:
        """Toroidal-angle orientation of the source EQDSK."""
        return self._phiclockwise_input

    @property
    def phiclockwise_internal(self) -> bool:
        """Toroidal-angle orientation of the stored arrays."""
        return self._phiclockwise_internal

    @property
    def flux_normalization_input(self) -> FluxNormalization:
        """Poloidal-flux normalization of the source EQDSK file."""
        return self._flux_normalization_input

    @property
    def flux_normalization_internal(self) -> FluxNormalization:
        """Poloidal-flux normalization of the stored arrays."""
        return self._flux_normalization_internal
    
    @property
    def cocos_info(self) -> Dict[str, Any]:
        """
        Full COCOS metadata.
        
        Returns
        -------
        dict
            Explicit input and internal COCOS descriptors and sign metadata.
        """
        cocos_input_obj = get_cocos(self._cocos_input)
        cocos_internal_obj = get_cocos(self._cocos_internal)
        return {
            'cocos_input': self._cocos_input,
            'cocos_internal': self._cocos_internal,
            'phiclockwise_input': self._phiclockwise_input,
            'phiclockwise_internal': self._phiclockwise_internal,
            'flux_normalization_input': self._flux_normalization_input,
            'flux_normalization_internal': self._flux_normalization_internal,
            'cocos_input_obj': cocos_input_obj,
            'cocos_internal_obj': cocos_internal_obj,
            'exp_Bp_input': cocos_input_obj.exp_Bp,
            'sigma_Bp_input': cocos_input_obj.sigma_Bp,
            'sigma_RpZ_input': cocos_input_obj.sigma_RpZ,
            'sigma_rhotp_input': cocos_input_obj.sigma_rhotp,
            'sign_q_pos_input': cocos_input_obj.sign_q_pos,
            'sign_pprime_pos_input': cocos_input_obj.sign_pprime_pos,
            'exp_Bp_internal': cocos_internal_obj.exp_Bp,
            'sigma_Bp_internal': cocos_internal_obj.sigma_Bp,
            'sigma_RpZ_internal': cocos_internal_obj.sigma_RpZ,
            'sigma_rhotp_internal': cocos_internal_obj.sigma_rhotp,
            'sign_q_pos_internal': cocos_internal_obj.sign_q_pos,
            'sign_pprime_pos_internal': cocos_internal_obj.sign_pprime_pos,
        }
    
    def _internal_geqdsk_data(self) -> Dict[str, Any]:
        """Return an independent FreeQDSK-schema snapshot of stored data."""
        rgrid = np.asarray(self._gdata['Rgrid'])
        zgrid = np.asarray(self._gdata['zgrid'])
        output = {
            'nx': int(self._gdata['lr']),
            'ny': int(self._gdata['lz']),
            'rdim': float(rgrid[-1] - rgrid[0]),
            'zdim': float(zgrid[-1] - zgrid[0]),
            'rcentr': float(self._gdata['rcentr']),
            'rleft': float(rgrid[0]),
            'zmid': float(0.5 * (zgrid[0] + zgrid[-1])),
            'rmagx': float(self._gdata['Raxis']),
            'zmagx': float(self._gdata['zaxis']),
            'simagx': float(self._gdata['psi_ax']),
            'sibdry': float(self._gdata['psi_bdy']),
            'bcentr': float(self._gdata['Bcenter']),
            'cpasma': float(self._gdata['Ip']),
            'fpol': np.array(self._gdata['fpol'], copy=True),
            'pres': np.array(self._gdata['prs'], copy=True),
            'psi': np.array(self._gdata['psi'], copy=True),
            'qpsi': np.array(self._gdata['q'], copy=True),
        }

        for output_name, internal_name in (
            ('ffprime', 'ffp'),
            ('pprime', 'pprime'),
            ('rbdry', 'r_bdy'),
            ('zbdry', 'z_bdy'),
            ('rlim', 'rlim'),
            ('zlim', 'zlim'),
        ):
            if internal_name in self._gdata:
                output[output_name] = np.array(
                    self._gdata[internal_name],
                    copy=True,
                )
        return output

    def to_geqdsk(
        self,
        cocos_out: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a non-mutating g-EQDSK view in the requested COCOS.

        Parameters
        ----------
        cocos_out : int, optional
            Output convention. The stored ``cocos_internal`` is used when
            omitted.

        Returns
        -------
        dict
            Independent FreeQDSK-compatible data in ``cocos_out``.
        """
        if cocos_out is None:
            cocos_out = self._cocos_internal
        else:
            cocos_out = get_cocos(cocos_out).cocos

        return fromCocosNtoCocosM(
            self._internal_geqdsk_data(),
            cocos_m=cocos_out,
            cocos_n=self._cocos_internal,
        )

    def save(
        self,
        filename: str,
        cocos_out: Optional[int] = None,
    ) -> None:
        """Write a g-EQDSK file in the requested COCOS.

        Parameters
        ----------
        filename : str
            Output filename. Existing files are never overwritten.
        cocos_out : int, optional
            Output convention. The stored ``cocos_internal`` is used when
            omitted.

        Examples
        --------
        >>> eq.save('output.geqdsk')
        >>> eq.save('output.geqdsk', cocos_out=3)
        """
        output = self.to_geqdsk(cocos_out=cocos_out)
        with open(filename, 'x') as stream:
            freeqdsk.geqdsk.write(output, stream)

    def to_dict(self) -> Dict[str, Any]:
        """Return an independent snapshot in the internal pyCOCOS schema."""
        return copy.deepcopy(self._gdata)
    
    @classmethod
    def load(
        cls,
        filename: str,
        cocos_in: Optional[int] = None,
        cocos_internal: int = 1,
        phiclockwise_in: Optional[bool] = None,
        flux_normalization: Optional[FluxNormalization] = None,
    ):
        """
        Load equilibrium from g-EQDSK file (factory method).

        Parameters
        ----------
        filename : str
            Path to g-EQDSK file
        cocos_in : int, optional
            Explicit input COCOS convention.
        cocos_internal : int, optional
            COCOS convention used by the stored arrays. Default is 1.
        phiclockwise_in : bool, optional
            Input toroidal-angle orientation used for detection.
        flux_normalization : {"Wb", "Wb/rad"}, optional
            Input poloidal-flux normalization used for detection.

        Returns
        -------
        EQDSK
            EQDSK instance

        Examples
        --------
        >>> eq = EQDSK.load('equilibrium.geqdsk', cocos_in=1, cocos_internal=1)
        """
        return cls(
            filename,
            cocos_in=cocos_in,
            cocos_internal=cocos_internal,
            phiclockwise_in=phiclockwise_in,
            flux_normalization=flux_normalization,
        )
