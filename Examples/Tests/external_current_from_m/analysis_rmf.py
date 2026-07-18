#!/usr/bin/env python3

import sys

import numpy as np
import yt
from read_raw_data import read_data


A_COIL = 1.0
R_COIL = 2.0
W_COIL = 0.8
OMEGA_RMF = 5.0e8 * np.pi
T_RAMP = 2.0e-9
DT = 1.0e-9
DX = 1.0


def ramp(time):
    if time < T_RAMP:
        return 0.5 * (1.0 - np.cos(np.pi * time / T_RAMP))
    return 1.0


def my(x, y, z, time):
    profile = np.exp(-(y**2) / (2.0 * W_COIL**2))
    profile *= 0.5 * (1.0 - np.tanh((np.sqrt(x**2 + z**2) - R_COIL) / W_COIL))
    return ramp(time) * A_COIL * profile * np.cos(OMEGA_RMF * time)


def mz(x, y, z, time):
    profile = np.exp(-(z**2) / (2.0 * W_COIL**2))
    profile *= 0.5 * (1.0 - np.tanh((np.sqrt(x**2 + y**2) - R_COIL) / W_COIL))
    return ramp(time) * A_COIL * profile * np.sin(OMEGA_RMF * time)


def expected_current(time):
    cell = np.arange(8) + 0.5 - 4.0
    node = np.arange(9) - 4.0

    x, y, z = np.meshgrid(cell, node, node, indexing="ij")
    jx = (
        mz(x, y + 0.5 * DX, z, time) - mz(x, y - 0.5 * DX, z, time)
        - my(x, y, z + 0.5 * DX, time) + my(x, y, z - 0.5 * DX, time)
    ) / DX

    x, y, z = np.meshgrid(node, cell, node, indexing="ij")
    jy = -(mz(x + 0.5 * DX, y, z, time) - mz(x - 0.5 * DX, y, z, time)) / DX

    x, y, z = np.meshgrid(node, node, cell, indexing="ij")
    jz = (my(x + 0.5 * DX, y, z, time) - my(x - 0.5 * DX, y, z, time)) / DX

    return {"jx_fp": jx, "jy_fp": jy, "jz_fp": jz}


interior = (slice(1, -1), slice(1, -1), slice(1, -1))

for plotfile in sys.argv[1:]:
    diagnostic_time = float(yt.load(plotfile).current_time)
    source_time = diagnostic_time - 0.5 * DT
    raw = read_data(plotfile)[0]
    expected = expected_current(source_time)

    for component, reference in expected.items():
        actual = raw[component][interior]
        reference = reference[interior]
        error = np.max(np.abs(actual - reference))
        print(
            f"{plotfile}, {component}: source time = {source_time:.6e}, "
            f"RMF current error = {error:.16e}"
        )
        assert np.max(np.abs(reference)) > 1.0e-3
        assert error < 2.0e-12
