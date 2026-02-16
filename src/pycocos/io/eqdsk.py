"""
EQDSK library.

This library reads and parses the equilibrium file in the so-called EQDSK and
parses the COCOS standard.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from findiff import FinDiff
import freeqdsk
from typing import Dict, Any, Optional

# Import from new structure
from ..core.equilibrium import equilibrium

from .cocos import fromCocosNtoCocosM
from .cocos import assign as assign_cocos
from .cocos import cocos as get_cocos
import logging

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')


logger = logging.getLogger('equ.eqdsk')

if len(logger.handlers) == 0:
    hnd = logging.StreamHandler()
    hnd.setFormatter(fmt)
    logger.addHandler(hnd)

#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

__mapping = (('hdr', 'comment'),   ('Bcenter', 'bcentr'),  ('Ip', 'cpasma'),
             ('lr', 'nx'),         ('lz', 'ny'),           ('r_bdy', 'rbdry'),
             ('z_bdy', 'zbdry'),   ('Raxis', 'rmagx'),     ('zaxis', 'zmagx'),
             ('psi', 'psi'),       ('psi_ax', 'simagx'),   ('psi_bdy', 'sibdry'),
             ('psimax', 'sibdry'), ('fpol', 'fpol'),       ('prs', 'pres'),
             ('ffp', 'ffprime'),   ('pprime', 'pprime'),    ('q', 'qpsi'))


# -----------------------------------------------------------------------------
# ROUTINES TO READ THE EQDSK.
# -----------------------------------------------------------------------------
# g-file reader from Giovanni.
def ssplit(ll: str):
    """
    Read a formatted output from any FORTRAN code.

    adapted from:
    Giovanni Tardini - git@ipp.mpg.de

    :param ll: line to transform to a numeric list.
    """
    tmp = ll.replace('-', ' -')
    tmp = tmp.replace('e -', 'e-')
    tmp = tmp.replace('E -', 'E-')
    slist = tmp.split()
    a = [float(i) for i in slist]

    return a

def read_eqdsk_2(filename: str, cocos: int=2):
    """
    Read an EQDSK file and return a dictionary with the physical info.

    :param filename: file to read.
    :param cocos: COCOS standard.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Cannot locate the file {filename}.')

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Setting the constants according to the COCOS convention.
    if cocos < 10:
        pi_exp = 0
    else:
        pi_exp = 1

    if cocos in (1, 11, 2, 12, 5, 15, 6, 16):
        sigma_bp = +1
    else:
        sigma_bp = -1
    
    if cocos in (1, 11, 2, 12, 7, 17, 8, 18):
        sigma_rho = +1
    else:
        sigma_rho = -1

    print('sigma_bp', sigma_bp, 'sigma_rho', sigma_rho)


    header = lines[0].split()
    nw = int(header[-2])
    nh = int(header[-1])

    rdim, zdim, rcentr, rleft, zmid      = ssplit(lines[1])
    rdim, zdim, rcentr, rleft, zmid      = ssplit(lines[1])
    rmaxis, zmaxis, simag, sibry, bcentr = ssplit(lines[2])
    current, simag, xdum, rmaxis, xdum   = ssplit(lines[3])
    zmaxis, xdum, sibry, xdum, xdum      = ssplit(lines[4])

    jline = 5
    len_d = {'fpol': nw, 'ffp': nw, 'prs': nw, 'pprim': nw, \
             'psi': nw*nh, 'q':nw}

    output = {}
    n_lin  = 5
    geq_sig = ('fpol', 'prs', 'ffp', 'pprim', 'psi', 'q')
    for lbl in geq_sig:
        nx = len_d[lbl]
        output[lbl] = np.zeros(nx)
        jl = 0
        while jl < nx:
            jr = min([jl+n_lin, nx+1])
            output[lbl][jl: jr] = ssplit(lines[jline])
            jline += 1
            jl += n_lin
    psi = output['psi'].reshape(nh, nw).T


    nbbbs, limitr = lines[jline].split()
    n_bdy = int(nbbbs)
    n_lim = int(limitr)
    jline += 1
    len_d = {'bdy': n_bdy, 'lim': n_lim}
    for lbl in ('bdy', 'lim'):
        nx = 2*len_d[lbl]
        output[lbl] = np.zeros(nx)
        jl = 0
        while jl < nx:
            jr = min([jl+5, nx])
            count = jr - jl
            output[lbl][jl:jr] = ssplit(lines[jline])[:count]
            jline += 1
            jl += 5

    # We check now whether there is consistency in the magnetic flux at the axis.
    Rgrid = np.linspace(rleft, rleft + rdim, nw)
    zgrid = np.linspace(zmid - 0.5*zdim, zmid + 0.5*zdim, nh)
    intrp = RectBivariateSpline(Rgrid, zgrid, psi)

    simag_intrp = intrp(rmaxis, zmaxis, grid=False).squeeze()


    if abs(abs(simag_intrp) - abs(simag)) > 1e-4:
        logger.warning('The magnetic flux at the axis seems ' +
                       'not be consistent with the value on the file')
        simag = simag_intrp


    # Let's do the same with the boundary.
    bdy = output['bdy'].reshape(n_bdy, 2)
    Rbdy = bdy[:, 0]
    zbdy = bdy[:, 1]
    # sibdy_intrp = np.mean(intrp(Rbdy, zbdy, grid=False))
    # plt.hist(intrp(Rbdy, zbdy, grid=False).flatten(), bins=100)

    if simag > sibry:
        psi_sign = -1.0
    else:
        psi_sign = +1.0

    output['hdr']     = header
    output['Bcenter'] = bcentr
    output['Ip']      = current
    output['lr']      = nw
    output['lz']      = nh
    output['r_bdy']   = bdy[:, 0]
    output['z_bdy']   = bdy[:, 1]
    output['Raxis']   = rmaxis
    output['zaxis']   = zmaxis
    output['Rgrid']   = np.linspace(rleft, rleft + rdim, nw)
    output['zgrid']   = np.linspace(zmid - 0.5*zdim, zmid + 0.5*zdim, nh)
    output['psi']     = psi_sign *psi   / (2.0*np.pi)**pi_exp
    output['psi_ax']  = psi_sign * simag / (2.0*np.pi)**pi_exp
    output['psi_bdy'] = psi_sign * sibry / (2.0*np.pi)**pi_exp
    output['psimax']  = output['psi_bdy'] - output['psi_ax']
    output['dpsi']    = np.abs(output['psimax'])/(output['lr']-1)
    output['psirz']   = output['psi'] - output['psi_ax']
    output['rhoprz']  = np.sqrt(output['psirz']/output['psimax'])
    output['q']       = output['q'] * sigma_bp * sigma_rho

    # --- Making the flux quantities to the grid.
    psi_1d = np.linspace(0.0, output['psimax'], num=output['lr'])
    output['psi_1d'] = psi_1d
    output['rhop_1d']  = np.sqrt(psi_1d/output['psimax'])

    flags = output['psirz'] < output['psimax']
    output['fpolrz'] = np.zeros_like(output['psirz'])
    intrp = interp1d(psi_1d, output['fpol'], kind='cubic',
                                bounds_error=False, fill_value='extrapolate')
    output['fpolrz'][flags] = intrp(output['psirz'][flags])
    flags = np.logical_not(flags)
    output['fpolrz'][flags] = output['fpol'][-1]

    output['prsrz'] = interp1d(psi_1d, output['prs'], kind='linear',
                                bounds_error=False, fill_value=0.0)
    output['prsrz'] = output['prsrz'](output['psirz'])
    output['cocos'] = 1

    return output

def read_eqdsk(
    filename: str,
    cocos: int = 1,
    phiclockwise: bool = True
) -> Dict[str, Any]:
    """
    Read an EQDSK file using the freeqdsk library.

    This function reads g-EQDSK files and automatically handles COCOS
    detection and conversion.

    Parameters
    ----------
    filename : str
        Path to the EQDSK file
    cocos : int, optional
        Target COCOS convention. Default is 1
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Default is True

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
    >>> data = read_eqdsk('equilibrium.geqdsk', cocos=1)
    """
    with open(filename, 'r') as f:
        d = freeqdsk.geqdsk.read(f)

    # We check which is the input COCOS.
    cocos_in = assign_cocos(d['qpsi'][0], d['cpasma'], d['bcentr'],
                            d['simagx'], d['sibdry'], phiclockwise=phiclockwise)
    
    if cocos_in != cocos:
        logger.warning(f'The input COCOS is {cocos_in}. Transforming to COCOS: {cocos}')
    
    # Transforming the equilibrium to the output COCOS.
    d = fromCocosNtoCocosM(d, cocos)

    cocos_out = assign_cocos(d['qpsi'][0], d['cpasma'], d['bcentr'],
                             d['simagx'], d['sibdry'])

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
    output['cocos'] = cocos
    output['cocos_in'] = cocos_in
    output['cocos_out'] = cocos_out

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
        - 'cocos': COCOS convention

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

    if 'fpol' not in eqdata:
        raise Exception('The fpol must be within the input data.')


    output = dict()
    # To compute the poloidal magnetic field, we use the psi (poloidal
    # flux surfaces and 4th order finite differences.
    dr = np.abs(eqdata['Rgrid'][1] - eqdata['Rgrid'][0])
    dz = np.abs(eqdata['zgrid'][1] - eqdata['zgrid'][0])

    Rmesh, _ = np.meshgrid(eqdata['Rgrid'], eqdata['zgrid'])

    d_dr = FinDiff(0, dr, 1, acc=4)
    d_dz = FinDiff(1, dz, 1, acc=4)

    # Correcting the poloidal field sign.
    if eqdata['cocos'] in (1, 11, 2, 12, 5, 15, 6, 16):
        sign_bpol = +1
    else:
        sign_bpol = -1

    # Correcting whether we have or not a direct triedron.
    if eqdata['cocos'] in (1, 11, 3, 13, 5, 15, 7, 17):
        sign_bpol *= 1.0
    else:
        sign_bpol *= -1.0

    output['br'] = + d_dz(eqdata['psi'])/Rmesh.T * sign_bpol
    output['bz'] = - d_dr(eqdata['psi'])/Rmesh.T * sign_bpol

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
    from g-EQDSK files with automatic COCOS detection and conversion.

    Parameters
    ----------
    fn : str
        Filename of the EQDSK file
    cocos : int, optional
        Target COCOS convention (default: 1). The file's COCOS will be
        auto-detected and converted if necessary
    phiclockwise : bool, optional
        Whether toroidal angle increases clockwise. Default is True

    Attributes
    ----------
    filename : str
        Path to the EQDSK file
    gs_profs : xr.Dataset
        Grad-Shafranov profiles (fpol, pres, q, etc.)

    Examples
    --------
    >>> from pycocos import EQDSK
    >>> eq = EQDSK('equilibrium.geqdsk')
    >>> print(eq.gs_profs.q)  # Access q-profile
    """

    def __init__(
        self,
        fn: str,
        cocos: int = 1,
        phiclockwise: bool = True
    ) -> None:
        """
        Create an Equilibrium object starting from an EQDSK file.

        Parameters
        ----------
        fn : str
            Filename of the EQDSK file
        cocos : int, optional
            Target COCOS convention. Default is 1
        phiclockwise : bool, optional
            Whether toroidal angle increases clockwise. Default is True

        Raises
        ------
        FileNotFoundError
            If the EQDSK file cannot be found
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError(f'Cannot locate the file {fn}.')
        
        self.filename = fn

        # Launching a Deprecation Warning.
        if cocos != 1:
            logger.warning('The COCOS input is deprecated. ' +
                           'The code detects the original COCOS and ' +
                           'transforms it to the standard COCOS: 1')

        # Reading the EQDSK.
        self._gdata = read_eqdsk(filename=fn, cocos=1, phiclockwise=phiclockwise)
        self._bfield = eqdsk2magnetic(self._gdata)
        
        # Store COCOS information (detect from raw data before conversion)
        # Read raw file again to get original COCOS
        with open(fn, 'r') as f:
            d_raw = freeqdsk.geqdsk.read(f)
        
        self._cocos_detected = assign_cocos(
            d_raw.get('qpsi', [0])[0] if len(d_raw.get('qpsi', [])) > 0 else 0,
            d_raw.get('cpasma', 0),
            d_raw.get('bcentr', 0),
            d_raw.get('simagx', 0),
            d_raw.get('sibdry', 0),
            phiclockwise=phiclockwise
        )
        self._cocos_target = cocos

        # Using the parent class to perform the hard initializing.
        super().__init__(self._gdata['Rgrid'],   self._gdata['zgrid'],
                         self._bfield['br'],     self._bfield['bz'],
                         self._bfield['bphi'],   self._gdata['psi'],
                         self._gdata['Raxis'],   self._gdata['zaxis'],
                         self._gdata['psi_bdy'], self._gdata['psi_ax'],
                         phiclockwise=phiclockwise)

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
                                            'units': '$T\cdot m$',
                                            'desc': '$RB_\\phi$',
                                            'short_name': '$RB_\\phi$'
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
        _psi1d = xr.DataArray(tmp, dims=('rhop',),
                             attrs={'name': 'psi', 'units': 'Wb',
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
                                                           'units': 'T^2*m^4/Wb',
                                                           'desc': 'd(F*F)/dPsi',
                                                           'short_name': '$FF\'$'})
        
        if 'pprime' in self._gdata:
            self._profiles['pprime'] = xr.DataArray(self._gdata['pprime'],
                                                    dims=('rhop',),
                                                    coords={'rhop': _rho1d},
                                                    attrs={'name': 'pprime',
                                                           'units': 'Pa/Wb',
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
    def cocos(self) -> int:
        """
        Detected COCOS convention of the loaded file.
        
        Returns
        -------
        int
            COCOS ID number (1-18)
        """
        return self._cocos_detected
    
    @property
    def cocos_info(self) -> Dict[str, Any]:
        """
        Full COCOS metadata.
        
        Returns
        -------
        dict
            Dictionary containing COCOS information:
            - 'detected': Detected COCOS ID
            - 'target': Target COCOS ID (after conversion)
            - 'cocos_obj': COCOS object with full metadata
        """
        cocos_obj = get_cocos(self._cocos_detected)
        return {
            'detected': self._cocos_detected,
            'target': self._cocos_target,
            'cocos_obj': cocos_obj,
            'exp_Bp': cocos_obj.exp_Bp,
            'sigma_Bp': cocos_obj.sigma_Bp,
            'sigma_RpZ': cocos_obj.sigma_RpZ,
            'sigma_rhotp': cocos_obj.sigma_rhotp,
            'sign_q_pos': cocos_obj.sign_q_pos,
            'sign_pprime_pos': cocos_obj.sign_pprime_pos,
        }
    
    def save(self, filename: str, cocos: Optional[int] = None) -> None:
        """
        Save equilibrium to g-EQDSK file.

        Parameters
        ----------
        filename : str
            Output filename
        cocos : int, optional
            COCOS convention to use. If None, uses current convention.
            Default is None

        Examples
        --------
        >>> eq.save('output.geqdsk')
        >>> eq.save('output.geqdsk', cocos=3)
        """
        return self.to_geqdsk(filename, cocos=cocos)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert equilibrium to dictionary format.

        Returns
        -------
        dict
            Dictionary containing equilibrium data suitable for
            writing to EQDSK or other formats
        """
        output = {}
        
        # Grid information
        output['Rgrid'] = self.Rgrid.values
        output['zgrid'] = self.zgrid.values
        output['Raxis'] = self.R_axis
        output['zaxis'] = self.z_axis
        
        # Flux surfaces
        output['psi'] = self.psi.values
        output['psi_ax'] = self.geometry.attrs.get('psi_ax')
        output['psi_bdy'] = self.geometry.attrs.get('psi_bdy')
        output['psimax'] = self.geometry.attrs.get('psimax')
        
        # Magnetic field
        output['br'] = self.Br.values
        output['bz'] = self.Bz.values
        output['bphi'] = self.Bphi.values
        
        # Profiles if available
        if len(self._profiles) > 0:
            for var_name in self._profiles.data_vars:
                output[var_name] = self._profiles[var_name].values
        
        # COCOS info
        output['cocos'] = self._cocos_detected
        
        return output
    
    @classmethod
    def load(cls, filename: str, cocos: int = 1, phiclockwise: bool = True):
        """
        Load equilibrium from g-EQDSK file (factory method).

        Parameters
        ----------
        filename : str
            Path to g-EQDSK file
        cocos : int, optional
            Target COCOS convention. Default is 1
        phiclockwise : bool, optional
            Whether toroidal angle increases clockwise. Default is True

        Returns
        -------
        EQDSK
            EQDSK instance

        Examples
        --------
        >>> eq = EQDSK.load('equilibrium.geqdsk')
        """
        return cls(filename, cocos=cocos, phiclockwise=phiclockwise)
