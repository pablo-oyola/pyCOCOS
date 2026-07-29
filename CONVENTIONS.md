# pyCOCOS conventions

## Equilibrium loading

The loader converts once from the input convention to the caller-selected
`cocos_internal`. COCOS 1 remains the default, for which

\[
\psi=\Phi_{\mathrm{pol}}/(2\pi),\qquad
\mathbf B=\nabla\phi\times\nabla\psi+F\nabla\phi,
\]

\[
B_R=R^{-1}\partial_Z\psi,\qquad
B_Z=-R^{-1}\partial_R\psi,\qquad B_\phi=F/R.
\]

The internal toroidal orientation and flux units are derived from
`cocos_internal`; they are not fixed separately. For example, COCOS 1 has
`phiclockwise_internal=False` and `flux_normalization_internal="Wb/rad"`,
whereas COCOS 12 has `phiclockwise_internal=True` and
`flux_normalization_internal="Wb"`.

A plain EQDSK does not encode toroidal-angle orientation or whether poloidal
flux is stored in Wb or Wb/rad. Its signs alone therefore leave four valid
COCOS candidates. Either pass an authoritative `cocos_in`, or omit it and pass
both `phiclockwise_in` and `flux_normalization` for unique detection. An
explicit `cocos_in` bypasses sign-based detection, which also provides the
override path for damaged files. Input provenance is retained separately from
the selected internal convention.

`EQDSK.to_geqdsk(cocos_out=None)` returns an independent FreeQDSK-compatible
view and leaves the stored representation unchanged. Its default output is
`cocos_internal`; pass `cocos_out` to convert the output copy. `EQDSK.save`
writes the same view and refuses to overwrite an existing file. Because a
plain EQDSK cannot portably record its complete COCOS, reload an exported file
with the same value as explicit `cocos_in`.

Magnetic-coordinate datasets keep `coords.psi0` strictly increasing for
spline construction. Its `psi_axis` and `psi_boundary` attributes preserve
the physical axis-to-boundary orientation, including equilibria for which
physical \(\psi\) decreases outward. Normalized flux always means

\[
\psi_N=\frac{\psi-\psi_{\rm axis}}
             {\psi_{\rm boundary}-\psi_{\rm axis}}.
\]

## Boozer coordinates

The toroidal gauge is

\[
\zeta=\phi+\nu(\psi,\Theta),\qquad
\partial\phi/\partial\zeta=1.
\]

`coords.nu` is the two-dimensional gauge-shift table. The full magnetic
toroidal coordinate is `zeta`, with no alias between the two. The signed
physical Jacobian is

\[
J=[\nabla\psi\cdot(\nabla\Theta\times\nabla\zeta)]^{-1}.
\]

For Boozer output,

\[
J\mathbf B\cdot\nabla=\partial_\Theta+q\partial_\zeta,
\qquad F=g=B_\zeta,\qquad I=B_\Theta,
\qquad h=JB^2=I+qF.
\]

`deriv.I` is the sole exported covariant coefficient. The coordinate path does
not export alternate \(2\pi\)-scaled aliases or a duplicated `boozer_profs`
dataset.

## Metrics

`metric(..., tensor="contravariant")` returns
\(g^{ij}=\nabla x^i\cdot\nabla x^j\), while `tensor="covariant"` returns
\(g_{ij}=\partial_i\mathbf r\cdot\partial_j\mathbf r\). Cylindrical angular
components use \((1/R)\partial_\phi\) for gradients and
\(R\partial_i\phi\) for tangents. These are the only two accepted tensor
selectors.

All direct and reciprocal derivatives are assembled from the same
Fourier/radial-spline coordinate map. On the fitted annulus they obey

\[
\nabla\psi\cdot\partial_\psi\mathbf r=1,\qquad
\nabla\psi\cdot\partial_\Theta\mathbf r=0,
\]

and the covariant and contravariant metric tensors are algebraic inverses.
`coords.inside_coordinate_domain` identifies this annulus. The equilibrium
flux gradient and the pure flux metric
\(g^{\psi\psi}=|\nabla\psi|^2\) remain available on the complete finite
equilibrium grid. Magnetic-angle derivatives and metrics involving a magnetic
angle are not extrapolated outside the fitted annulus.
