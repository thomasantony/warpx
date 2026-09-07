#!/usr/bin/env python3
"""Check native Yee identities from primitive CSV products and inventories.

This test does not interpret instantaneous Yee field energy as an exactly
conserved quantity. The explicit split-stage correction must be retained.
Only Python's standard library and NumPy are required.
"""

import csv
from pathlib import Path

import numpy as np


def main():
    with Path("diags/native_yee_energy_balance.csv").open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert len(rows) == 16, "Expected every step of the 16-step fixture"
        data = {name: np.array([float(row[name]) for row in rows])
                for name in reader.fieldnames}
    assert all(np.all(np.isfinite(value)) for value in data.values())
    np.testing.assert_array_equal(data["step"], np.arange(1, 17))
    np.testing.assert_array_equal(data["domain_cells"], np.full(16, 16**3))

    # A deliberately conservative serial-summation gamma_n envelope covers
    # per-cell product/interpolation/curl arithmetic plus summation over cells.
    # The actual reduction tree is shorter. This is a floating-point bound on
    # primitive absolute operands, never a percentage of a cancelled residual.
    epsilon = np.maximum(data["real_epsilon"], np.finfo(float).eps)
    n_epsilon = 500 * data["domain_cells"] * epsilon
    assert np.all(n_epsilon < 1)
    gamma = n_epsilon / (1 - n_epsilon)

    stages = ("start", "after_B_first", "after_B_first_exchange", "after_E",
              "after_E_exchange", "after_B_second", "end")
    inventories = sum(data[f"U_{stage}_J"] for stage in stages)
    previous_u = np.r_[data["U_start_J"][0], data["U_end_J"][:-1]]
    inventories += previous_u
    field_scale = inventories + sum(data[f"{name}_abs_product_J"]
                                    for name in ("W", "CE", "CB_first", "CB_second"))
    face_scale = sum(data[f"{name}_abs_product_J"] for name in ("CE", "A", "F"))
    field_bound = gamma * field_scale
    face_bound = gamma * face_scale

    def check(label, actual, expected, bound):
        difference = np.abs(actual - expected)
        assert np.all(difference <= bound), (
            f"{label}: |difference|={difference}, arithmetic_bound={bound}"
        )

    for stage in stages:
        check(f"{stage} field inventory", data[f"U_{stage}_J"],
              data[f"UE_{stage}_J"] + data[f"UB_{stage}_J"], field_bound)
    delta_u = data["U_end_J"] - previous_u
    gap = data["U_start_J"] - previous_u
    g_b = data["U_after_B_first_exchange_J"] - data["U_after_B_first_J"]
    g_e = data["U_after_E_exchange_J"] - data["U_after_E_J"]
    g_final = data["U_end_J"] - data["U_after_B_second_J"]
    g = gap + g_b + g_e + g_final
    cb = data["CB_first_J"] + data["CB_second_J"]
    t = data["A_J"] + cb
    flux = sum(data[f"F_{axis}{side}_J"] for axis in "xyz" for side in ("lo", "hi"))
    check("delta U", data["delta_U_J"], delta_u, field_bound)
    check("solve delta U", data["delta_U_solve_J"],
          data["U_end_J"] - data["U_start_J"], field_bound)
    for name, value in (("G_gap_J", gap), ("G_B_first_exchange_J", g_b),
                        ("G_E_exchange_J", g_e), ("G_final_exchange_J", g_final),
                        ("G_J", g), ("T_J", t), ("F_J", flux)):
        check(name, data[name], value, field_bound + face_bound)

    update = delta_u + data["W_J"] - data["CE_J"] - cb - g
    face = data["CE_J"] - data["A_J"] + flux
    combined = delta_u + data["W_J"] + flux - t - g
    for name, value, bound in (("field_update", update, field_bound),
                               ("face_identity", face, face_bound),
                               ("combined", combined, field_bound + face_bound)):
        check(name, value, 0, bound)
        check(f"reported {name}", data[f"residual_{name}_J"], value, bound)

    for axis in "xyz":
        check(f"periodic {axis} faces", data[f"F_{axis}lo_J"] + data[f"F_{axis}hi_J"],
              0, face_bound)
    for name in ("W_J", "Wraw_J", "K_pre_push_J", "K_post_push_J",
                 "K_pre_boundary_J", "K_post_boundary_J", "K_outgoing_J"):
        np.testing.assert_array_equal(data[name], np.zeros(16))
    assert np.any(np.abs(t) > field_bound + face_bound), (
        "Fixture must resolve the split-stage correction above arithmetic error"
    )

    # Every cumulative column must retain the same segment origin. Bound this
    # short sum by the sum of absolute row values, including cancelled histories.
    for name in data:
        if name.startswith("cum_"):
            values = data[name[4:]]
            bound = gamma * np.cumsum(np.abs(values))
            check(name, data[name], np.cumsum(values), bound)

    old_energy = np.atleast_2d(np.loadtxt("diags/reduced/FieldEnergy.txt"))
    old_flux = np.atleast_2d(np.loadtxt("diags/reduced/FieldPoyntingFlux.txt"))
    assert np.all(np.isfinite(old_energy)) and np.all(np.isfinite(old_flux))
    for old in (old_energy, old_flux):
        np.testing.assert_array_equal(old[:, 0], np.arange(17))
        check("historical diagnostic times", old[1:, 1], data["time_end_s"],
              gamma * np.abs(data["time_end_s"]))
    check("historical/native endpoint energy", old_energy[1:, 2],
          data["U_end_J"], field_bound)
    check("historical periodic integrated flux", old_flux[1:, 8:14].sum(axis=1),
          0, np.cumsum(face_bound))
    print("PASS: native Yee stage/face/combined identities, periodic faces, "
          "cumulative histories and historical diagnostic cadence")


if __name__ == "__main__":
    main()
