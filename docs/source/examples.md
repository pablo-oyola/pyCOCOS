# Examples

The scripts are in
[`examples/`](https://github.com/pablo-oyola/pyCOCOS/tree/master/examples).
Each one writes the equilibrium it needs next to itself, so any of them runs
on its own, from any directory, in a few seconds:

```bash
python examples/boozer_coordinates.py
```

The output shown under each script is from running it.

## An analytic equilibrium

`solovev_equilibrium.py` writes a Solov'ev equilibrium, an exact solution of
the Grad-Shafranov equation with constant $p'$ and $FF' = 0$:

$$
\psi(R, z) = A\left[R^2 z^2 + \frac{\kappa^2}{4}\left(R^2 - R_0^2\right)^2\right].
$$

Here $R_0 = 3$ m, $a = 1$ m, $\kappa = 1.6$ and $B_0 = 2.5$ T, and $A$ is
chosen for $q_0 = 1.1$.  The file is COCOS 1: $\psi$ in Wb/rad, increasing
outwards, with positive current and field.  Its safety factor,
$q = (F/2\pi)\,dV/d\psi$ with $V = \int dR\,dz/R$ over the surface, reduces to
a smooth one-dimensional quadrature.  It is exact to that quadrature, which
makes it a reference for the q that pyCOCOS computes.

```{literalinclude} ../../examples/solovev_equilibrium.py
:language: python
```

```text
Wrote solovev.geqdsk: Ip = 9.871 MA, q from 1.100 to 2.658
```

## Loading an equilibrium and its COCOS

`basic_usage.py` reads the file twice.  The first read gives the COCOS
explicitly.  The second lets pyCOCOS detect it: the signs of ψ, q and the
current narrow it to COCOS 1, 2, 11 or 12, and the toroidal orientation and
flux normalization decide between them.

```{literalinclude} ../../examples/basic_usage.py
:language: python
```

```text
Detected COCOS 1, stored as COCOS 1
============================================================
Equilibrium Summary
============================================================

Grid:
  R: [1.000, 4.400] m (129 points)
  z: [-2.500, 2.500] m (129 points)

Magnetic Axis:
  R_axis = 3.000 m
  z_axis = 0.000 m
  B_axis = 2.500 T

Flux Surfaces:
  psi_ax = 0.000e+00 Wb/rad
  psi_bdy = 2.475e+00 Wb/rad
  psimax = 2.475e+00 Wb/rad

Profiles available:
  - fpol
  - pres
  - ffprime
  - pprime
  - q
  - psi
  - rho

Magnetic Field Components:
  Br:   [-1.736, 1.736] T
  Bz:   [-3.079, 1.616] T
  Bphi: [1.705, 7.500] T
  B:    [2.232, 7.672] T
============================================================
Field components: ['Br', 'Bz', 'Bphi', 'Babs', 'Bpol', 'Baxis']
q(axis) = 1.100, q(edge) = 2.658
```

## Boozer coordinates

`boozer_coordinates.py` builds Boozer coordinates on 65 surfaces × 128
poloidal points.  It compares the q of the traced surfaces with the exact one,
and maps a point both ways.

- **Inverse map.**  The ψ_N = 0.5 surface crosses the outboard midplane at
  $R^2 = R_0^2 + 2\sqrt{\psi/A}/\kappa$, which is R = 3.7349 m, exactly what
  the inverse map returns.
- **Forward map.**  ψ at (3.5, 0.3) m is
  $A(3.5^2 \cdot 0.3^2 + 0.64 \cdot 3.25^2) = 0.6205$ Wb/rad analytically.
  The forward transform returns 0.6210, about 0.1 % high.  It interpolates
  ψ on the file's 129 × 129 grid with a degree-1 spline by default, and a
  linear interpolant always overestimates a convex ψ.

```{literalinclude} ../../examples/boozer_coordinates.py
:language: python
```

```text
max relative q error: 1.1e-05
(R, z) = (3.5, 0.3) m -> psi = 0.6210 Wb/rad, theta = 6.0687 rad, nu = -0.0150 rad
(psi_N, theta) = (0.5, 0) -> R = 3.7349 m, z = -1.3e-12 m
```

```{image} _static/examples/boozer_coordinates.png
:alt: Boozer grid of the Solov'ev equilibrium, and its safety factor against the exact profile
:width: 100%
```

The lines of constant Boozer angle crowd on the inboard side.  The Jacobian
$J = h/B^2$ grows roughly as $R^2$, so on the outermost surface drawn a step in
θ covers about five times more arc on the outboard midplane than on the
inboard one.

## Converting between COCOS conventions

`eqdsk_io.py` writes the file as COCOS 11 and reads it back.  COCOS 11
differs from COCOS 1 only in the flux normalization: ψ in Wb, so the
boundary flux grows by 2π.

```{literalinclude} ../../examples/eqdsk_io.py
:language: python
```

```text
psi at the boundary: COCOS 1 2.4747 Wb/rad, COCOS 11 15.5493 Wb
Round trip through COCOS 11 recovers psi and q: True
```
