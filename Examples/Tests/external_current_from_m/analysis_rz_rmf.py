#!/usr/bin/env python3

import sys

import numpy as np
import openpmd_api as io
import yt
from read_raw_data import read_data


B0 = 2.0e-6
RADIUS = 2.0
WIDTH = 3.0
FREQUENCY = 2.5e8
RAMP_TIME = 2.0e-9
DT = 1.0e-9
DR = 1.0
DZ = 1.0
MU0 = 1.2566370612685e-6
NCOMPS = 5

SIGMA0 = 0.5 * (1.0 - np.tanh(-RADIUS / (2.0 * WIDTH)))
M0 = 3.0 * B0 / (2.0 * MU0 * SIGMA0)


def ramp(time):
    if time < RAMP_TIME:
        return 0.5 * (1.0 - np.cos(np.pi * time / RAMP_TIME))
    return 1.0


def sigma(r, z):
    u = (r**2 + z**2 - RADIUS**2) / (2.0 * RADIUS * WIDTH)
    return 0.5 * (1.0 - np.tanh(u))


def magnetization_modes(r, z, time):
    phase = 2.0 * np.pi * FREQUENCY * time
    amplitude = M0 * ramp(time) * sigma(r, z)
    return {
        "mr_real": amplitude * np.cos(phase),
        "mr_imag": amplitude * np.sin(phase),
        "mt_real": amplitude * np.sin(phase),
        "mt_imag": -amplitude * np.cos(phase),
    }


def expected_current(time):
    result = {}

    r, z = np.meshgrid(np.arange(8) + 0.5, np.arange(9) - 4.0, indexing="ij")
    mt_hi = magnetization_modes(r, z + 0.5 * DZ, time)
    mt_lo = magnetization_modes(r, z - 0.5 * DZ, time)
    jr = np.zeros((*r.shape, NCOMPS))
    jr[..., 1] = -(mt_hi["mt_real"] - mt_lo["mt_real"]) / DZ
    jr[..., 2] = -(mt_hi["mt_imag"] - mt_lo["mt_imag"]) / DZ
    result["jx_fp"] = jr

    r, z = np.meshgrid(np.arange(9), np.arange(9) - 4.0, indexing="ij")
    mr_hi = magnetization_modes(r, z + 0.5 * DZ, time)
    mr_lo = magnetization_modes(r, z - 0.5 * DZ, time)
    jt = np.zeros((*r.shape, NCOMPS))
    jt[..., 1] = (mr_hi["mr_real"] - mr_lo["mr_real"]) / DZ
    jt[..., 2] = (mr_hi["mr_imag"] - mr_lo["mr_imag"]) / DZ
    result["jy_fp"] = jt

    r, z = np.meshgrid(np.arange(9), np.arange(8) + 0.5 - 4.0, indexing="ij")
    mt_hi = magnetization_modes(r + 0.5 * DR, z, time)
    mt_lo = magnetization_modes(r - 0.5 * DR, z, time)
    mr = magnetization_modes(r, z, time)
    jz = np.zeros((*r.shape, NCOMPS))
    off_axis = r > 0.0
    drr_mt_real = np.zeros_like(r, dtype=float)
    drr_mt_imag = np.zeros_like(r, dtype=float)
    drr_mt_real[off_axis] = (
        (r[off_axis] + 0.5 * DR) * mt_hi["mt_real"][off_axis]
        - (r[off_axis] - 0.5 * DR) * mt_lo["mt_real"][off_axis]
    ) / (r[off_axis] * DR)
    drr_mt_imag[off_axis] = (
        (r[off_axis] + 0.5 * DR) * mt_hi["mt_imag"][off_axis]
        - (r[off_axis] - 0.5 * DR) * mt_lo["mt_imag"][off_axis]
    ) / (r[off_axis] * DR)
    jz[..., 1][off_axis] = (
        drr_mt_real[off_axis] - mr["mr_imag"][off_axis] / r[off_axis]
    )
    jz[..., 2][off_axis] = (
        drr_mt_imag[off_axis] + mr["mr_real"][off_axis] / r[off_axis]
    )
    result["jz_fp"] = jz

    return result


def divergence_m1(raw, suffix):
    jr = raw[f"jx_fp{suffix}"]
    jt = raw[f"jy_fp{suffix}"]
    jz = raw[f"jz_fp{suffix}"]
    r = (np.arange(1, 8) * DR)[:, None]

    radial_real = (
        (r + 0.5 * DR) * jr[1:8, 1:8, 1]
        - (r - 0.5 * DR) * jr[0:7, 1:8, 1]
    ) / (r * DR)
    radial_imag = (
        (r + 0.5 * DR) * jr[1:8, 1:8, 2]
        - (r - 0.5 * DR) * jr[0:7, 1:8, 2]
    ) / (r * DR)
    axial_real = (jz[1:8, 1:8, 1] - jz[1:8, 0:7, 1]) / DZ
    axial_imag = (jz[1:8, 1:8, 2] - jz[1:8, 0:7, 2]) / DZ

    return (
        radial_real + jt[1:8, 1:8, 2] / r + axial_real,
        radial_imag - jt[1:8, 1:8, 1] / r + axial_imag,
    )


def check_axis_continuity(raw, suffix, source_time):
    jr = raw[f"jx_fp{suffix}"]
    jt = raw[f"jy_fp{suffix}"]
    jz = raw[f"jz_fp{suffix}"]

    # A regular m=1 vector field satisfies Jtheta = -i*Jr on the axis. Jr is
    # radially cell-centered and has no independent r=0 degree of freedom, so
    # construct its axis limit from the same regular magnetization and compare
    # that limit with the production Jtheta value stored at r=0.
    z = np.arange(9) - 4.0
    mt_hi = magnetization_modes(np.zeros_like(z), z + 0.5 * DZ, source_time)
    mt_lo = magnetization_modes(np.zeros_like(z), z - 0.5 * DZ, source_time)
    jr_axis_real = -(mt_hi["mt_real"] - mt_lo["mt_real"]) / DZ
    jr_axis_imag = -(mt_hi["mt_imag"] - mt_lo["mt_imag"]) / DZ
    real_error = np.max(np.abs(jt[0, 1:-1, 1] - jr_axis_imag[1:-1]))
    imag_error = np.max(np.abs(jt[0, 1:-1, 2] + jr_axis_real[1:-1]))
    axial_error = np.max(np.abs(jz[0, 1:-1, 1:3]))
    axis_error = max(real_error, imag_error, axial_error)
    print(f"current_fp{suffix}: m=1 axis continuity error = {axis_error:.16e}")
    assert axis_error < 2.0e-12

    # The production RZ divergence has no independent higher-mode degree of
    # freedom at r=0: it regularizes div(J)_m to zero there. The adjacent r=dr
    # control volume is covered by the ordinary modal divergence below.
    div_real, div_imag = divergence_m1(raw, suffix)
    first_off_axis_error = max(
        np.max(np.abs(div_real[0])), np.max(np.abs(div_imag[0]))
    )
    print(
        f"current_fp{suffix}: first off-axis divergence error = "
        f"{first_off_axis_error:.16e}; axis divergence = 0 by production regularity"
    )
    assert first_off_axis_error < 2.0e-12


def check_openpmd_contract(path):
    series = io.Series(f"{path}/openpmd_%T.h5", io.Access.read_only)
    iteration = series.iterations[max(series.iterations)]
    assert "j" in iteration.meshes
    assert "j_external" in iteration.meshes

    for component in ("r", "t", "z"):
        total = iteration.meshes["j"][component].load_chunk()
        external = iteration.meshes["j_external"][component].load_chunk()
        series.flush()
        assert total.shape[0] == NCOMPS
        assert total.shape == external.shape
        error = np.max(np.abs(total - external))
        print(
            f"openPMD j_external/{component}: thetaMode shape = {total.shape}, "
            f"total/external error = {error:.16e}"
        )
        assert np.max(np.abs(external[1:3])) > 1.0e-3
        assert error < 2.0e-12


interiors = {
    "jx_fp": (slice(None), slice(1, -1), slice(None)),
    "jy_fp": (slice(0, -1), slice(1, -1), slice(None)),
    "jz_fp": (slice(0, -1), slice(1, -1), slice(None)),
}

for plotfile in sys.argv[1:-1]:
    diagnostic_time = float(yt.load(plotfile).current_time)
    source_time = diagnostic_time - 0.5 * DT
    raw = read_data(plotfile)[0]
    expected = expected_current(source_time)

    for component, reference in expected.items():
        interior = interiors[component]
        reference = reference[interior]
        for actual_component in (component, f"{component}_external"):
            actual = raw[actual_component][interior]
            error = np.max(np.abs(actual - reference))
            print(
                f"{plotfile}, {actual_component}: source time = {source_time:.6e}, "
                f"RZ RMF current error = {error:.16e}"
            )
            assert np.max(np.abs(reference[..., 1:3])) > 1.0e-3
            assert error < 2.0e-12

    for suffix in ("", "_external"):
        check_axis_continuity(raw, suffix, source_time)
        div_real, div_imag = divergence_m1(raw, suffix)
        divergence_error = max(
            np.max(np.abs(div_real)), np.max(np.abs(div_imag))
        )
        print(
            f"{plotfile}, current_fp{suffix}: "
            f"RZ RMF divergence error = {divergence_error:.16e}"
        )
        assert divergence_error < 2.0e-12

check_openpmd_contract(sys.argv[-1])
