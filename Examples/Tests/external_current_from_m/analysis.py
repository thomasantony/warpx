#!/usr/bin/env python3

import sys

import numpy as np
import yt
from read_raw_data import read_data


def reduced_value(name):
    data = np.loadtxt(f"diags/reducedfiles/{name}.txt")
    return float(np.atleast_2d(data)[-1, 2])

ds = yt.load(sys.argv[1])
grid = ds.covering_grid(
    level=0,
    left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions,
)

x = grid["index", "x"].to_ndarray()
y = grid["index", "y"].to_ndarray()
z = grid["index", "z"].to_ndarray()

# PEC current boundary handling modifies the outermost cells after the external
# source is added. The interior includes the AMReX box interfaces because this
# test uses max_grid_size=4 on an 8^3 domain.
interior = (slice(1, -1), slice(1, -1), slice(1, -1))

expected = {
    "jx": 12.0 * y - 8.0 * z,
    "jy": 4.0 * z - 10.0 * x,
    "jz": 6.0 * x - 2.0 * y,
}

for component, reference in expected.items():
    actual = grid["boxlib", component].to_ndarray()[interior]
    coordinates = [coordinate[interior] for coordinate in (x, y, z)]
    manufactured_error = np.max(np.abs(actual - reference[interior]))
    symmetry_error = np.max(np.abs(actual + actual[::-1, ::-1, ::-1]))
    weight = np.abs(actual)
    centroids = [np.sum(weight * coordinate) / np.sum(weight) for coordinate in coordinates]
    print(
        f"{component}: manufactured error = {manufactured_error:.16e}, "
        f"odd-symmetry error = {symmetry_error:.16e}, centroids = {centroids}"
    )
    assert manufactured_error < 1.0e-12
    assert symmetry_error < 1.0e-12
    assert np.max(np.abs(centroids)) < 1.0e-12


# Check the same manufactured curl before diagnostic cell-centering, directly
# on each component's native Yee staggering. Array dimensions follow x,y,z.
raw = read_data(sys.argv[1])[0]
native_coordinates = {
    "jx_fp": np.meshgrid(
        np.arange(8) + 0.5 - 4.0,
        np.arange(9) - 4.0,
        np.arange(9) - 4.0,
        indexing="ij",
    ),
    "jy_fp": np.meshgrid(
        np.arange(9) - 4.0,
        np.arange(8) + 0.5 - 4.0,
        np.arange(9) - 4.0,
        indexing="ij",
    ),
    "jz_fp": np.meshgrid(
        np.arange(9) - 4.0,
        np.arange(9) - 4.0,
        np.arange(8) + 0.5 - 4.0,
        indexing="ij",
    ),
}
native_references = {
    "jx_fp": lambda x, y, z: 12.0 * y - 8.0 * z,
    "jy_fp": lambda x, y, z: 4.0 * z - 10.0 * x,
    "jz_fp": lambda x, y, z: 6.0 * x - 2.0 * y,
}

for component, reference_function in native_references.items():
    actual = raw[component][interior]
    coordinates = [coordinate[interior] for coordinate in native_coordinates[component]]
    reference = reference_function(*native_coordinates[component])[interior]
    manufactured_error = np.max(np.abs(actual - reference))
    symmetry_error = np.max(np.abs(actual + actual[::-1, ::-1, ::-1]))
    weight = np.abs(actual)
    centroids = [np.sum(weight * coordinate) / np.sum(weight) for coordinate in coordinates]
    print(
        f"{component} (native): manufactured error = {manufactured_error:.16e}, "
        f"odd-symmetry error = {symmetry_error:.16e}, centroids = {centroids}"
    )
    assert manufactured_error < 1.0e-12
    assert symmetry_error < 1.0e-12
    assert np.max(np.abs(centroids)) < 1.0e-12


external_l2 = reduced_value("ExternalCurrentL2")
plasma_l2 = reduced_value("PlasmaCurrentL2")
total_l2 = reduced_value("TotalCurrentL2")
assert external_l2 > 0.0
assert np.isclose(total_l2, external_l2, rtol=1.0e-14, atol=0.0)
assert abs(plasma_l2) < 1.0e-24
