"""Unit test for the structural J_ext = curl(M_ext) fix (jext_impl_02.md).

This replicates, in plain numpy, the exact Yee staggering and finite-difference
stencils used by WarpX::AddCurlOfMToCurrent (Source/Initialization/WarpXInitData.cpp)
to build J_ext = curl(M_ext) from a B-staggered magnetization M_ext, and by the
standard Yee discrete divergence operator (the same stencil used to advance
current_fp/current_cp -> div(J) in the field solver / Gauss-law diagnostics).

The point of the structural fix is that div(curl(M)) == 0 identically on the
Yee mesh, to machine precision -- not just approximately small, as would be the
case for a directly parsed analytic J_ext. This test asserts exactly that.
"""

import math
import unittest

import numpy as np


def yee_curl_M_to_J(Mx, My, Mz, dx, dy, dz):
    """Discrete curl of a B-staggered vector field (Mx, My, Mz) onto the
    current (E-type) Yee staggering, matching AddCurlOfMToCurrent:

        Jx(i,j,k) = (Mz(i,j+1,k) - Mz(i,j,k)) / dy - (My(i,j,k+1) - My(i,j,k)) / dz
        Jy(i,j,k) = (Mx(i,j,k+1) - Mx(i,j,k)) / dz - (Mz(i+1,j,k) - Mz(i,j,k)) / dx
        Jz(i,j,k) = (My(i+1,j,k) - My(i,j,k)) / dx - (Mx(i,j+1,k) - Mx(i,j,k)) / dy

    Arrays are indexed [i, j, k] with one extra guard point in each direction
    beyond the interior region that will be evaluated, matching the ng=1
    scratch fields allocated around Mx/My/Mz in WarpX.
    """
    nx, ny, nz = Mx.shape[0] - 1, My.shape[1] - 1, Mz.shape[2] - 1
    # Interior region where all forward differences below are well defined.
    ni, nj, nk = min(nx, Mx.shape[0] - 1), min(ny, My.shape[1] - 1), min(nz, Mz.shape[2] - 1)

    Jx = ((Mz[:ni, 1:nj + 1, :nk] - Mz[:ni, :nj, :nk]) / dy
          - (My[:ni, :nj, 1:nk + 1] - My[:ni, :nj, :nk]) / dz)
    Jy = ((Mx[:ni, :nj, 1:nk + 1] - Mx[:ni, :nj, :nk]) / dz
          - (Mz[1:ni + 1, :nj, :nk] - Mz[:ni, :nj, :nk]) / dx)
    Jz = ((My[1:ni + 1, :nj, :nk] - My[:ni, :nj, :nk]) / dx
          - (Mx[:ni, 1:nj + 1, :nk] - Mx[:ni, :nj, :nk]) / dy)
    return Jx, Jy, Jz


def yee_divergence(Jx, Jy, Jz, dx, dy, dz):
    """Discrete divergence of an E-staggered current (Jx, Jy, Jz), evaluated
    at the same grid points where the Yee Ampere-law update evaluates
    div(J) (the "H"/cell-corner points diagonally forward of each J point by
    half a cell in the other two directions):

        div(J)(i,j,k) = (Jx(i+1,j,k) - Jx(i,j,k)) / dx
                      + (Jy(i,j+1,k) - Jy(i,j,k)) / dy
                      + (Jz(i,j,k+1) - Jz(i,j,k)) / dz

    This is the finite-difference divergence operator that is the algebraic
    adjoint of the forward-difference curl in yee_curl_M_to_J above: the two
    together telescope to exactly zero (div(curl) == 0 to machine precision),
    which is the discrete identity this test exercises. A divergence built
    from a differently-handed (e.g. backward-difference) stencil does *not*
    telescope against this curl and is not the operator this fix targets.
    """
    ni, nj, nk = Jx.shape[0] - 1, Jx.shape[1] - 1, Jx.shape[2] - 1
    div = ((Jx[1:ni + 1, :nj, :nk] - Jx[:ni, :nj, :nk]) / dx
           + (Jy[:ni, 1:nj + 1, :nk] - Jy[:ni, :nj, :nk]) / dy
           + (Jz[:ni, :nj, 1:nk + 1] - Jz[:ni, :nj, :nk]) / dz)
    return div


class JextCurlDivergenceTest(unittest.TestCase):
    def _check_zero_divergence(self, Mx, My, Mz, dx, dy, dz):
        Jx, Jy, Jz = yee_curl_M_to_J(Mx, My, Mz, dx, dy, dz)
        div = yee_divergence(Jx, Jy, Jz, dx, dy, dz)

        # Machine precision, not "small": this is the whole point of the
        # structural J_ext = curl(M_ext) fix (jext_impl_02.md).
        max_abs_div = np.max(np.abs(div))
        scale = max(1.0, np.max(np.abs(Jx)), np.max(np.abs(Jy)), np.max(np.abs(Jz)))
        self.assertLess(max_abs_div / scale, 1.0e-12)

    def test_divergence_of_curl_is_zero_for_random_M(self):
        rng = np.random.default_rng(1234)
        n = 12
        ng = 1
        shape = (n + 2 * ng, n + 2 * ng, n + 2 * ng)
        Mx = rng.standard_normal(shape)
        My = rng.standard_normal(shape)
        Mz = rng.standard_normal(shape)

        self._check_zero_divergence(Mx, My, Mz, dx=0.7, dy=1.1, dz=0.9)

    def test_divergence_of_curl_is_zero_for_rmf_coil_profile(self):
        # Same functional form as the RMF coil M_y/M_z profiles in
        # solar_wind_rmf_explicit_coils.txt, sampled on a small 3D grid.
        n = 16
        ng = 1
        dx = dy = dz = 1.0
        R_coil = 4.0
        w_coil = 1.5
        A_coil = 2.0
        t = 0.3
        omega_rmf = 1.0

        idx = np.arange(-ng, n + ng)
        x = (idx + 0.5) * dx - 0.5 * n * dx
        y = (idx + 0.5) * dy - 0.5 * n * dy
        z = (idx + 0.5) * dz - 0.5 * n * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        rho_xz = np.sqrt(X * X + Z * Z)
        rho_xy = np.sqrt(X * X + Y * Y)

        Mx = np.zeros_like(X)
        My = (A_coil * np.exp(-Y * Y / (2.0 * w_coil * w_coil))
              * 0.5 * (1.0 - np.tanh((rho_xz - R_coil) / w_coil))
              * math.cos(omega_rmf * t))
        Mz = (A_coil * np.exp(-Z * Z / (2.0 * w_coil * w_coil))
              * 0.5 * (1.0 - np.tanh((rho_xy - R_coil) / w_coil))
              * math.sin(omega_rmf * t))

        self._check_zero_divergence(Mx, My, Mz, dx, dy, dz)


if __name__ == "__main__":
    unittest.main()
