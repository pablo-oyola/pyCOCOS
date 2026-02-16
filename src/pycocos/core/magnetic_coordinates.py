"""
Library for the magnetic coordinates class handler that
eases the access and control of the magnetic coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import h5py
import os
from typing import Union, Optional, Tuple, Dict, Any
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from copy import copy

import logging
logger = logging.getLogger('magneticcoords')

class magnetic_coordinates:
    """
    Container for magnetic coordinates and transformation routines.

    This class stores magnetic coordinate transformations and provides
    methods to transform between cylindrical (R, z, phi) and magnetic
    (psi, theta, nu) coordinate systems.

    Parameters
    ----------
    coords : xr.Dataset
        Dataset containing magnetic coordinates (psi, theta, nu)
    deriv : xr.Dataset
        Dataset containing derivatives of coordinate transformations
    Raxis : float
        Radial position of the magnetic axis
    zaxis : float
        Vertical position of the magnetic axis
    pad : int, optional
        Number of padding points for periodic boundary conditions. Default is 0

    Attributes
    ----------
    coords : xr.Dataset
        Magnetic coordinate data
    deriv : xr.Dataset
        Derivative data
    lame_mag : xr.Dataset
        Lamé factors for magnetic coordinates
    Raxis : float
        Radial position of magnetic axis
    zaxis : float
        Vertical position of magnetic axis
    nthtpad : int
        Number of theta padding points

    Examples
    --------
    >>> coords = mag_coords(R=2.0, z=0.0)  # Transform to magnetic coords
    >>> cyl_coords = mag_coords.transform_inverse(psi=0.5, thetamag=0.0)
    """
    def __init__(
        self,
        coords: xr.Dataset,
        deriv: xr.Dataset,
        Raxis: float,
        zaxis: float,
        pad: int = 0
    ) -> None:
        """
        Initialize the magnetic coordinate object.

        Parameters
        ----------
        coords : xr.Dataset
            Dataset containing the magnetic coordinates
        deriv : xr.Dataset
            Dataset containing the derivatives of the magnetic coordinates
        Raxis : float
            R position of the magnetic axis
        zaxis : float
            z position of the magnetic axis
        pad : int, optional
            Number of padding points for periodic boundaries. Default is 0
        """
        self.coords = coords
        self.deriv = deriv

        # Computing the Lamé factors for the magnetic derivatives.
        self.lame_mag = xr.Dataset()
        self.lame_mag['h_psi'] = np.sqrt(self.deriv.dR_dpsi**2 + 
                                         self.deriv.dz_dpsi**2 +
                                         self.deriv.R**2 * self.deriv.dphi_dpsi**2)
        self.lame_mag['h_theta'] = np.sqrt(self.deriv.dR_dtheta**2 + 
                                           self.deriv.dz_dtheta**2 +
                                           self.deriv.R**2 * self.deriv.dphi_dtheta**2)
        self.lame_mag['h_zeta'] = np.sqrt(self.deriv.dR_dzeta**2 + 
                                          self.deriv.dz_dzeta**2 +
                                          self.deriv.R**2 * self.deriv.dphi_dzeta**2)
        
        # Storing the axis position.
        self.Raxis = Raxis
        self.zaxis = zaxis

        # and the padding introduced to the geometrical poloidal angle.
        self.nthtpad = pad
        
    def __call__(
        self,
        R: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        df: int = 0,
        grid: bool = False,
        only: Optional[str] = None,
        fill_nan: bool = True
    ) -> Union[xr.Dataset, Tuple[xr.Dataset, xr.Dataset]]:
        """
        Transform cylindrical coordinates to magnetic coordinates.

        This is a contravariant transformation: (R, z, phi) -> (psi, theta, nu)

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s)
        z : float or np.ndarray
            Vertical coordinate(s)
        df : int, optional
            If 0, return only coordinates. If 1, return coordinates and derivatives.
            Default is 0
        grid : bool, optional
            If True, create a grid from R and z. Default is False
        only : str, optional
            If df=1, return only this derivative. Default is None
        fill_nan : bool, optional
            If True, fill values outside LCFS with NaN. Default is True

        Returns
        -------
        xr.Dataset or tuple of xr.Dataset
            If df=0: Dataset with (psi, theta, nu)
            If df=1: Tuple of (coordinates Dataset, derivatives Dataset)

        Examples
        --------
        >>> coords = mag_coords(R=2.0, z=0.0)  # Single point
        >>> coords, derivs = mag_coords(R, z, df=1)  # With derivatives
        """
        if df == 0:
            return self._transform(R=R, z=z, grid=grid,
                                    fill_nan=fill_nan)
        elif df == 1:
            coords = self._transform(R=R, z=z, grid=grid, 
                                     fill_nan=False)
            derivs = self._transform_deriv(R=R, z=z, grid=grid,
                                           only=only)
            if fill_nan:
                flags  = (coords.psi < self.coords.psi0.min()) | \
                         (coords.psi > self.coords.psi0.max())
                for name in derivs:
                    derivs[name].values[flags] = np.nan
                for name in coords:
                    coords[name].values[flags] = np.nan
            
            return coords, derivs
        else:
            raise ValueError("df must be 0 or 1")
    
    def _transform(self, R: float, z: float, grid: bool=False,
                   fill_nan: bool=True):
        """
        Transform the input coordinates to the output coordinates.

        :param R: Radial values to evaluate the transformation.
        :param z: Vertical values to evaluate the transformation.
        :param grid: True to return the coordinates in a grid.
        :param fill_nan: when True the values outside the LCFS
                            are filled with NaN. Otherwise, the
                            simple internal extrapolation is kept.
        """
        output = xr.Dataset()

        if grid:
            grr, gzz = np.meshgrid(R, z, indexing='ij')
        else:
            assert R.shape == z.shape, "R and z must have the same shape"
            grr = R
            gzz = z

        # Interpolating the psi coordinate.
        intrp = RectBivariateSpline(self.coords.R.values,
                                    self.coords.z.values,
                                    self.coords.psi.values, 
                                    kx=1, ky=1)
        name = 'psi'
        tmp = xr.DataArray(intrp(R, z, grid=grid))
        if grid:
            output[name] = xr.DataArray(tmp, dims=('R', 'z'),
                                        coords={'R': R, 'z': z})
        else:
            output[name] = xr.DataArray(tmp)
        output[name].attrs = {'desc': self.coords[name].desc,
                              'units': self.coords[name].units,
                              'short_name': self.coords[name].short_name,
                              'name': self.coords[name].name}

        # For the input R and z we compute the values of the
        # poloial geometrical angle.
        thetageom = np.arctan2(gzz - self.zaxis, 
                               grr - self.Raxis)
        thetageom = np.mod(thetageom + 2*np.pi, 2.0*np.pi)

        for name in ('theta', 'nu'):
            # Building the Bivariate interpolator.
            intrp = RectBivariateSpline(self.coords.psi0.values,
                                        self.coords.thetageom.values,
                                        self.coords[name].values, kx=3, ky=5)
            tmp = xr.DataArray(intrp(output.psi.values,
                                     thetageom, grid=False))
            if grid:
                output[name] = xr.DataArray(tmp, dims=('R', 'z'),
                                            coords={'R': R, 'z': z})
            else:
                output[name] = xr.DataArray(tmp)
            
            # Adding the corresponding metadata.
            output[name].attrs = {'desc': self.coords[name].desc,
                                  'units': self.coords[name].units,
                                  'short_name': self.coords[name].short_name,
                                  'name': self.coords[name].name}
            
        # Checking if the Psi is within the range where the theta, zeta
        # are properly defined.
        if fill_nan:
            flags = (output.psi < self.coords.psi0.min()) | \
                    (output.psi > self.coords.psi0.max())
            for ikey in output:
                output[ikey].values[flags.values] = np.nan

        # Storing the points where the magnetic coordinates are evaluated
        # as additional elements in the dataset.
        if not grid:
            output['R'] = R
            output['z'] = z

        # Adding the attributes to the coordinates.
        output.R.attrs = {'desc': self.coords.R.desc,
                          'units': self.coords.R.units,
                          'short_name': self.coords.R.short_name,
                          'name': self.coords.R.name}
        output.z.attrs = {'desc': self.coords.z.desc,
                          'units': self.coords.z.units,
                          'short_name': self.coords.z.short_name,
                          'name': self.coords.z.name}

        return output

    def _transform_deriv(self, R: float, z: float, grid: bool=False,
                         only: str=None):
        """
        Transform the input coordinates to the output coordinates.

        :param R: Radial values to evaluate the transformation.
        :param z: Vertical values to evaluate the transformation.
        :param grid: True to return the coordinates in a grid.
        :param only: returns only the variable provided.
        """
        output = xr.Dataset()

        if (only is not None) and (only not in self.deriv):
            raise ValueError("The coordinate %s does not exist" % only)

        if grid:
            grr, gzz = np.meshgrid(R, z, indexing='ij')
        else:
            assert R.shape == z.shape, "R and z must have the same shape"
            grr = R
            gzz = z

        name_set = self.deriv
        if only is not None:
            name_set = [only,]

        for name in name_set:
            intrp = RectBivariateSpline(self.coords.R.values,
                                        self.coords.z.values,
                                        self.deriv[name].values)
            tmp = intrp(grr, gzz, grid=False)

            if grid:
                output[name] = xr.DataArray(tmp, dims=('R', 'z'),
                                            coords={'R': R, 'z': z})
            else:
                output[name] = xr.DataArray(tmp)

            # Adding the metadata.
            output[name].attrs = {'desc': self.deriv[name].desc,
                                  'units': self.deriv[name].units,
                                  'short_name': self.deriv[name].short_name,
                                  'name': self.deriv[name].name}
        
        return output

    def _transform_covariant_cyl_to_mag(self, r: float, vec: float, axis: int=0):
        """
        Transform the input coordinates to the output coordinates.

        :param r: values where the field is defined onto.
        :param vec: vector to transform.
        :param axis: axis where the vector is defined.
        :return rmag: magnetic coordinates vector.
        :return output: transformed vector into magnetic coordinates.
        """
        # Checking that the vector and the position vector have the same
        # shape and along the axis given it has 3 components.
        if vec.shape[axis] != 3:
            raise ValueError("The vector must have 3 components")
        
        # Moving the axis to the first position.
        vec = np.moveaxis(vec, axis, 0)
        pos = np.moveaxis(r, axis, 0)
        
        # Getting the transformed vector and the Jacobian matrix.
        pos = {iname: pos[i] for i, iname in enumerate(self.coords)}
        rmag, df = self(df=1, **pos)
        # Since we are using the covariant transformation we need to
        # include the Lamé factor in the phi derivative.
        df['dPsi_dphi'] /= df.R
        df['dTheta_dphi'] /= df.R
        df['dzeta_dphi'] /= df.R

        # With the derivatives we compute the covariant transformation.
        # Withing df we have all the derivatives we may need for 
        # transformation.
        output = np.zeros_like(vec)
        output[0] = vec[0]*df['dPsi_dr'] + vec[1]*df['dPsi_dphi'] + vec[2]*df['dPsi_dz']
        output[1] = vec[0]*df['dTheta_dr'] + vec[1]*df['dTheta_dphi'] + vec[2]*df['dTheta_dz']
        output[2] = vec[0]*df['dzeta_dr'] + vec[1]*df['dzeta_dphi'] + vec[2]*df['dzeta_dz']

        # Setting back the axis to the original position.
        output = np.moveaxis(output, 0, axis)
        rmag   = np.moveaxis(rmag, 0, axis)
        r      = np.moveaxis(r, 0, axis)
        vec    = np.moveaxis(vec, 0, axis)

        return rmag, output
    
    def _transform_covariant_mag_to_cyl(self, r: float, vec: float, axis: int=0):
        """
        Transform the input coordinates to the output coordinates.

        :param r: values where the field is defined onto.
        :param vec: vector to transform.
        :param axis: axis where the vector is defined.
        :return rmag: magnetic coordinates vector.
        :return output: transformed vector into magnetic coordinates.
        """
        # Checking that the vector and the position vector have the same
        # shape and along the axis given it has 3 components.
        if vec.shape[axis] != 3:
            raise ValueError("The vector must have 3 components")
        
        # Moving the axis to the first position.
        vec = np.moveaxis(vec, axis, 0)
        pos = np.moveaxis(r, axis, 0)
        
        # Getting the transformed vector and the Jacobian matrix.
        pos = {iname: pos[i] for i, iname in enumerate(self.coords)}
        rmag, df = self(df=1, **pos)

        output = np.zeros_like(vec)
        output[0] = vec[0]*df['dR_dpsi'] * self.lame_mag.h_psi + \
                    vec[1]*df['dR_dtheta'] * self.lame_mag.h_theta+ \
                    vec[2]*df['dR_dzeta'] * self.lame_mag.h_zeta
        output[1] = vec[0]*df['dZ_dpsi'] * self.lame_mag.h_psi + \
                    vec[1]*df['dZ_dtheta'] * self.lame_mag.h_theta+ \
                    vec[2]*df['dZ_dzeta'] * self.lame_mag.h_zeta
        output[2] = vec[0]*df['dPhi_dpsi'] * self.lame_mag.h_psi + \
                    vec[1]*df['dPhi_dtheta'] * self.lame_mag.h_theta+ \
                    vec[2]*df['dPhi_dzeta'] * self.lame_mag.h_zeta
        
        # Setting back the axis to the original position.
        output = np.moveaxis(output, 0, axis)
        rmag   = np.moveaxis(rmag, 0, axis)
        r      = np.moveaxis(r, 0, axis)
        vec    = np.moveaxis(vec, 0, axis)
        return rmag, output
    
    # --------------------------------------------------------------
    # Inverse transformation: magnetics -> cylindrical.
    # --------------------------------------------------------------
    def transform_inverse(
        self,
        psi: Union[float, np.ndarray],
        thetamag: Union[float, np.ndarray],
        grid: bool = False,
        psi_is_norm: bool = False
    ) -> xr.Dataset:
        """
        Transform from magnetic to cylindrical coordinates.

        Parameters
        ----------
        psi : float or np.ndarray
            Poloidal flux values
        thetamag : float or np.ndarray
            Magnetic poloidal angle values
        grid : bool, optional
            If True, create a grid from psi and thetamag. Default is False
        psi_is_norm : bool, optional
            If True, psi is normalized between 0 and 1. Default is False

        Returns
        -------
        xr.Dataset
            Dataset containing R_inv, z_inv coordinates

        Examples
        --------
        >>> cyl = mag_coords.transform_inverse(psi=0.5, thetamag=0.0)
        >>> R = cyl.R_inv.values
        >>> z = cyl.z_inv.values
        """
        output = xr.Dataset()

        # Transforming the inputs into a numpy array.
        psi = np.atleast_1d(psi)
        thetamag = np.atleast_1d(thetamag)

        if grid:
            gpsi, gtht = np.meshgrid(psi, thetamag, indexing='ij')
        else:
            assert psi.shape == thetamag.shape, "Psi and Theta must have the same shape"
            gpsi = psi
            gtht = thetamag
        
        # Interpolating the R and z coordinates.
        for ivar in ('R_inv', 'z_inv'):
            if psi_is_norm:
                psiN = self.coords.psi0.max() - self.coords.psi0.min()
                psi0_axis = (self.coords.psi0 - self.coords.psi0.min()) / psiN
            else:
                psi0_axis = self.coords.psi0
            intrp = RectBivariateSpline(psi0_axis,
                                        self.coords.theta_star.values,
                                        self.coords[ivar].values)
            tmp = intrp(gpsi, gtht, grid=False)
            if grid:
                output[ivar] = xr.DataArray(tmp, dims=('psi', 'thetamag'),
                                            coords={'psi': psi, 'thetamag': thetamag})
            else:
                output[ivar] = xr.DataArray(tmp, dims=('id',),
                                            coords={'id': np.arange(psi.size)})
                output['psi'] = xr.DataArray(psi, dims=('id',),
                                            coords={'id': np.arange(psi.size)})
                output['thetamag'] = xr.DataArray(thetamag, dims=('id',),
                                                  coords={'id': np.arange(psi.size)})
                
            output[ivar].attrs = {'desc': self.coords[ivar].desc,
                                    'units': self.coords[ivar].units,
                                    'short_name': self.coords[ivar].short_name,
                                    'name': self.coords[ivar].name}
            
        # Adding the attributes to the coordinates.
        output.psi.attrs.update(self.coords.psi.attrs)
        output.thetamag.attrs.update(self.coords.theta_star.attrs)
        
        return output

    def mag2cyl_scalar(
        self,
        field: Union[np.ndarray, xr.DataArray],
        psi: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        nu: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None
    ) -> xr.DataArray:
        """
        Transform a scalar field from magnetic coordinates to cylindrical.

        Parameters
        ----------
        field : np.ndarray or xr.DataArray
            Field defined in magnetic coordinates (psi, theta, nu)
            If DataArray, must have dims ('psi', 'theta', 'nu')
        psi : np.ndarray, optional
            Psi coordinate grid. Required if field is not a DataArray
        theta : np.ndarray, optional
            Theta coordinate grid. Required if field is not a DataArray
        nu : np.ndarray, optional
            Nu coordinate grid. Required if field is not a DataArray
        R : np.ndarray, optional
            Radial grid for output. If None, uses internal grid
        z : np.ndarray, optional
            Vertical grid for output. If None, uses internal grid
        phi : np.ndarray, optional
            Toroidal angle grid for output. If None, uses nu grid size

        Returns
        -------
        xr.DataArray
            Field transformed to cylindrical coordinates (phi, R, z)

        Raises
        ------
        ValueError
            If field shape doesn't match coordinate grids
        """
        # Checking that the field is consistent with the 
        # input magnetic coordinate shape.
        if not isinstance(field, xr.DataArray):
            if (psi is None) or (theta is None) or (nu is None):
                raise ValueError("The field must be a xarray.DataArray or " +
                                    "the coordinates must be provided")
            if not np.all(field.shape == (psi.size, theta.size, nu.size)):
                raise ValueError('The field must have the same shape as ' +
                                 'the input coordinates:', psi.size, theta.size, nu.size,
                                 '  - got instead:', field.shape)
        else:
            # Checking if the coordinates of the input array are
            # defined and are labelled as (psi, theta, nu).
            if not np.all(field.dims == ('psi', 'theta', 'nu')):
                raise ValueError('The field must have the same shape as ' +
                                 'the input coordinates:', psi.size, theta.size, nu.size,
                                 '  - got instead:', field.shape)
            logger.warning('The input field is a xarray.DataArray. ' +
                           'Ignoring the input coordinates.')
            # We make a shortcut for the coordinates.
            psi = field.psi.values
            theta = field.theta.values
            nu = field.nu.values
        
        # We have a consistent input, building the output array.
        if R is None:
            print('Generating R coordinates')
            R = np.linspace(self.coords.R.min(), 
                            self.coords.R.max(), 
                            self.coords.R.size)
        if z is None:
            z = np.linspace(self.coords.z.min(), 
                            self.coords.z.max(), 
                            self.coords.z.size)
        if phi is None:
            phi = np.linspace(0, 2*np.pi, nu.size)

        # We obtain the magnetic coordinates for the grid (R, z, phi),
        # and interpolate onto that grid the input field.
        new_coords = self._transform(R=R, z=z, grid=True)
        psi_out = new_coords.psi.values
        theta_out = new_coords.theta.values
        nu_out = new_coords.nu.values

        # Let's make sure that theta is within the range (0, 2pi)
        # by restricting it to that range.
        theta_out = np.mod(theta_out, 2*np.pi)

        # The input grid is evaluated on phi=0 only. If the user provided
        # a phi grid, the new nu is build as nu = nu + phi. Let's tile 
        # psi, theta, nu to the new grid.
        if phi.size >= 1:
            psi_out = np.tile(psi_out, (phi.size, 1, 1))
            theta_out = np.tile(theta_out, (phi.size, 1, 1))
            nu_out = np.tile(nu_out, (phi.size, 1, 1))
            nu_out += phi[:, np.newaxis, np.newaxis]

        # We interpolate the field onto the new grid.
        intrp = RegularGridInterpolator((psi, theta, nu), field.values,
                                        method='nearest',
                                        fill_value=0.0, bounds_error=False)

        # We build the new grid.
        psi_out_shape = copy(psi_out.shape)
        field_cyl = np.zeros_like(psi_out) + np.nan
        psi_out = psi_out.ravel()
        theta_out = theta_out.ravel()
        nu_out = nu_out.ravel()

        # Before we continue, we need to purge some dimensions before we
        # make the interpolation in the case there is only a single
        # point for the evaluation of the field.
        if psi.size == 1:
            psi_out[:] = psi[0]
        if theta.size == 1:
            theta_out[:] = theta[0]
        if nu.size == 1:
            nu_out[:] = nu[0]
        
        points = np.array((psi_out, theta_out, nu_out)).T
        field_cyl = intrp(points)

        field_cyl = field_cyl.reshape(psi_out_shape)

        # Building the output array as xarray.
        output = xr.DataArray(field_cyl, dims=('phi', 'R', 'z'),
                              coords={'R': R, 'z': z, 'phi': phi},
                              attrs=field.attrs.copy())
        
        return output

    def to_hdf5_as_xarray(self, fn: str, group_name: str=None):
        """
        Save to file HDF5 using the xarray package.

        We use this function to save the magnetic coordinates
        using the xarray package. This is useful to save the
        magnetic coordinates with the corresponding coordinates.
        """
         # Checking that the file does not exist previously.
        if (group_name is None) and (not os.path.isfile(fn)):
            raise ValueError("The file already exists")
        
        mode = 'w'
        if group_name is not None:
            mode = 'a'

        encoding = dict()
        
        # We create a temporal dataset to write all the data.
        tmp = xr.Dataset()
        for name in self.coords:
            if self.coords[name].dims[0] == 'psi0':
                tmp[name.lower()] = self.coords[name].astype(np.float64).transpose('thetageom', 'psi0')
            else:
                tmp[name.lower()] = self.coords[name].astype(np.float64).transpose('z', 'R')

            encoding[name.lower()] = {'dtype': 'float64'}

            # Transforming all the string attributes to numpy arrays.
            for ikey in tmp[name.lower()].attrs:
                if isinstance(tmp[name.lower()].attrs[ikey], str):
                    tmp[name.lower()].attrs[ikey] = np.array(tmp[name.lower()].attrs[ikey].encode(), dtype='S')

        # Saving the derivatives.
        for name in self.deriv:
            tmp[name.lower()] = self.deriv[name].astype(np.float64).transpose('z', 'R')
            encoding[name.lower()] = {'dtype': 'float64'}
            # Transforming all the string attributes to numpy arrays.
            for ikey in tmp[name.lower()].attrs:
                if isinstance(tmp[name.lower()].attrs[ikey], str):
                    tmp[name.lower()].attrs[ikey] = np.array(tmp[name.lower()].attrs[ikey].encode(), dtype='S')


        # We add the rest of the elements as attributes.
        tmp.attrs['rpsimin'] = np.float64(self.coords.psi0.min())
        tmp.attrs['rpsimax'] = np.float64(self.coords.psi0.max())
        tmp.attrs['npad'] = np.int32(self.nthtpad)
        tmp.attrs['ntht'] = np.int32(self.coords.thetageom.size - 2*self.nthtpad)

        # Dumping to the file.
        tmp.to_netcdf(fn, mode=mode,
                      group=group_name, 
                      engine='h5netcdf', 
                      encoding=encoding)
          
    def to_netcdf4(self, fn: str, group_name: str=None):
        """
        Save the magnetic coordinates object to a netcdf4 file.

        :param fn: filename where to save the magnetic coordinates.
        """
        # Checking that the file does not exist previously.
        if (group_name is None) and (not os.path.isfile(fn)):
            raise ValueError("The file already exists")
        
        mode = 'w'
        if group_name is not None:
            mode = 'a'

        # Creating the file.
        f = xr.Dataset()

        # Saving the coordinates.
        for name in self.coords:
            f[name] = self.coords[name]
        
        # Saving the derivatives.
        for name in self.deriv:
            f[name] = self.deriv[name]
        
        # Closing the file.
        f.to_netcdf(fn, group=group_name,
                    mode=mode, format='NETCDF4')

    def plot(self, coord_name=None, ax=None, 
             nr: int=256, nz: int=512, **kwargs):
        """
        Plot magnetic coordinates.

        If no coord_name is provided, lists available plottable coordinates.
        Otherwise, plots the specified coordinate.

        Parameters
        ----------
        coord_name : str, optional
            Name of the coordinate to plot. If None, lists available coordinates.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        nr : int, optional
            Number of radial points for plot grid. Default is 256
        nz : int, optional
            Number of vertical points for plot grid. Default is 512
        **kwargs
            Arguments passed to matplotlib contour

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object
        im : matplotlib ContourSet
            Contour plot object
        cbar : matplotlib Colorbar, optional
            Colorbar (if created)

        Examples
        --------
        >>> mag.plot()  # List available coordinates
        >>> ax, im, cbar = mag.plot('psi')  # Plot psi coordinate
        >>> ax, im, cbar = mag.plot('theta')  # Plot theta coordinate
        """
        if coord_name is None:
            # List available plottable coordinates
            print("Plottable coordinates:")
            print("  From coords:")
            for name in self.coords.data_vars:
                if self.coords[name].ndim == 2:
                    print(f"    - {name}")
            print("  From deriv:")
            for name in self.deriv.data_vars:
                if self.deriv[name].ndim == 2:
                    print(f"    - {name}")
            return None, None, None
        
        # Checking that the coordinate exists.
        if (coord_name not in self.coords) and (coord_name not in self.deriv):
            raise ValueError(f"The coordinate '{coord_name}' does not exist. "
                           f"Use mag.plot() to list available coordinates.")

        ax_is_none = ax is None
        if ax_is_none:
            fig, ax = plt.subplots()

        # Building the grid for the plot.
        r = np.linspace(self.coords.R.min().values, self.coords.R.max().values, nr)
        z = np.linspace(self.coords.z.min().values, self.coords.z.max().values, nz)
        extent = [r.min(), r.max(), z.min(), z.max()]
        
        if coord_name in self.coords:
            data = self(R=r, z=z, grid=True, df=0)[coord_name]
        else:
            data = self(R=r, z=z, grid=True, 
                        df=1, only=coord_name)[1][coord_name]
        
        im = ax.contour(r, z, data.values.T, extent=extent, origin='lower', **kwargs)

        if ax_is_none:
            xlabel = data.coords['R'].attrs.get('short_name', 'R')
            xunits = data.coords['R'].attrs.get('units', 'm')
            ylabel = data.coords['z'].attrs.get('short_name', 'z')
            yunits = data.coords['z'].attrs.get('units', 'm')

            ax.set_xlabel(f'{xlabel} [{xunits}]')
            ax.set_ylabel(f'{ylabel} [{yunits}]')
            cbar = fig.colorbar(im, ax=ax)

            ax.set_aspect('equal')

            zlabel = data.attrs.get('short_name', coord_name)
            zunits = data.attrs.get('units', '')
            cbar.set_label(f'{zlabel} [{zunits}]' if zunits else zlabel)
        
        return ax, im, None  # Return None for cbar to match old API
    
    def rescale(self, nr: int, nz: int,
                ntht: int=None, npsi: int=None,
                rmin: float=None, rmax: float=None,
                zmin: float=None, zmax: float=None):
        """
        Rescale the magnetic coordinates to the input grid.

        :param nr: number of points in the radial direction.
        :param nz: number of points in the vertical direction.
        :param ntht: number of points in the poloidal angle.
        :param npsi: number of points in the toroidal angle.
        :param rmin: minimum value of the radial coordinate.
        :param rmax: maximum value of the radial coordinate.
        :param zmin: minimum value of the vertical coordinate.
        :param zmax: maximum value of the vertical coordinate.
        """
        # Building the new grid.
        if rmin is None:
            rmin = self.coords.R.min()
        if rmax is None:
            rmax = self.coords.R.max()
        if zmin is None:
            zmin = self.coords.z.min()
        if zmax is None:
            zmax = self.coords.z.max()

        # Checking the input size of the grid.
        if nr <= 0:
            raise ValueError("nr must be greater than 0")
        if nz <= 0:
            raise ValueError("nz must be greater than 0")
        
        # Building the new grid.
        r = np.linspace(rmin, rmax, nr)
        z = np.linspace(zmin, zmax, nz)

        # We evaluate the data on the new grid grid.
        coords, devs = self(R=r, z=z, grid=True, df=1,
                            fill_nan=False)
    

        # For the theta and nu we use a linear interpolator
        # directly on the (psi0, thetageom) grid.
        if ntht is None:
            ntht = self.coords.thetageom.size
        if npsi is None:
            npsi = self.coords.psi0.size
        
        # Building the new grid.
        psi0 = np.linspace(self.coords.psi0.min(), 
                           self.coords.psi0.max(), 
                           npsi)
        thetageom = np.linspace(0, 2*np.pi, ntht)

        # Building the interpolator.
        theta = self.coords.theta.interp(psi0=psi0, 
                                         thetageom=thetageom,
                                         method='cubic')
        theta.attrs = self.coords.theta.attrs
        nu = self.coords.nu.interp(psi0=psi0, 
                                   thetageom=thetageom,
                                   method='cubic')
        nu.attrs = self.coords.nu.attrs

        # We append to the edges of theta and nu 
        # the shifted values to fake a periodic
        # boundary condition using a linear interpolator.
        dtheta    = thetageom[1] - thetageom[0]

        thetagrid = np.linspace(-self.nthtpad * dtheta, 
                                2*np.pi + self.nthtpad * dtheta, 
                                ntht + 2*self.nthtpad)
        leftside  = theta.values[:, -self.nthtpad:] - 2*np.pi
        rightside = theta.values[:,  :self.nthtpad] + 2*np.pi
        theta     = np.concatenate((leftside, theta.values, rightside), axis=1)
        theta = xr.DataArray(theta, dims=('psi0', 'thetageom'),
                                coords={'psi0': psi0, 'thetageom': thetagrid},
                                attrs=self.coords.theta.attrs.copy())
        
        # Same for nu.
        leftside  = nu.values[:, -self.nthtpad:]
        rightside = nu.values[:,  :self.nthtpad]
        nu = np.concatenate((leftside, nu.values, rightside), axis=1)
        nu = xr.DataArray(nu, dims=('psi0', 'thetageom'),
                            coords={'psi0': psi0, 'thetageom': thetagrid},
                            attrs=self.coords.nu.attrs.copy())
        
        coords['theta'] = theta
        coords['nu'] = nu

        # Building the new class instance.
        return magnetic_coordinates(coords, devs, 
                                    self.Raxis, self.zaxis,
                                    self.nthtpad)
    
    @property
    def coordnames(self):
        """
        Return the names of the coordinates.

        """

        return np.array(list(self.coords.keys()))