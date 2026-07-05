"""
Plot solar-wind electron/proton deflection from openPMD output.

Run the short example with, for instance:

build-release-mpi-all-geometries/bin/warpx.3d.MPI.OMP.DP.PDP.OPMD.EB.QED \
  Examples/Tutorial/solar_wind_magnetic_obstacle.txt max_step=80 diag1.intervals=20

Then plot using:

uv run --no-project \
  --with openpmd-viewer \
  --with openpmd-api \
  --with matplotlib \
  python Examples/Tutorial/solar_wind_magnetic_obstacle_plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def binned_mean(x, values, bins):
    """Return bin centers and means for finite values."""
    x = np.asarray(x)
    values = np.asarray(values)
    valid = np.isfinite(x) & np.isfinite(values)
    counts, edges = np.histogram(x[valid], bins=bins)
    sums, _ = np.histogram(x[valid], bins=bins, weights=values[valid])
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def downsample(*arrays, max_particles, seed=17):
    """Apply the same deterministic random downsample to all arrays."""
    size = len(arrays[0])
    if size <= max_particles:
        return arrays
    rng = np.random.default_rng(seed)
    keep = rng.choice(size, size=max_particles, replace=False)
    return tuple(array[keep] for array in arrays)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("diags/solar_wind_magnetic_obstacle"),
        help="openPMD diagnostic directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("solar_wind_magnetic_obstacle_plots"),
        help="directory for generated PNG plots and summary text",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=25000,
        help="maximum particles per species in scatter plots",
    )
    parser.add_argument("--bins", type=int, default=80, help="number of x bins for mean momenta")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        from openpmd_viewer import OpenPMDTimeSeries
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing plotting dependency. Run with:\n"
            "uv run --no-project --with openpmd-viewer --with openpmd-api "
            "--with matplotlib python Examples/Tutorial/solar_wind_magnetic_obstacle_plot.py"
        ) from exc

    ts = OpenPMDTimeSeries(str(args.input_dir))
    summaries = []
    colors = {"electrons": "tab:blue", "protons": "tab:red"}

    for iteration in ts.iterations:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        ax_xy, ax_xz, ax_uy, ax_uz = axes.ravel()

        summaries.append(f"iteration {iteration}")

        for species in ("electrons", "protons"):
            x, y, z, ux, uy, uz = ts.get_particle(
                ["x", "y", "z", "ux", "uy", "uz"],
                species=species,
                iteration=iteration,
            )

            y_mean = float(np.mean(y))
            z_mean = float(np.mean(z))
            uy_mean = float(np.mean(uy))
            uz_mean = float(np.mean(uz))
            summaries.append(
                f"  {species:9s} <y>={y_mean: .6e} m  <z>={z_mean: .6e} m  "
                f"<uy>={uy_mean: .6e}  <uz>={uz_mean: .6e}"
            )

            x_plot, y_plot, z_plot = downsample(
                x, y, z, max_particles=args.max_particles, seed=iteration + len(species)
            )
            ax_xy.scatter(x_plot, y_plot, s=0.4, alpha=0.18, c=colors[species], label=species)
            ax_xz.scatter(x_plot, z_plot, s=0.4, alpha=0.18, c=colors[species], label=species)

            centers, uy_bin = binned_mean(x, uy, args.bins)
            _, uz_bin = binned_mean(x, uz, args.bins)
            ax_uy.plot(centers, uy_bin, color=colors[species], label=species)
            ax_uz.plot(centers, uz_bin, color=colors[species], label=species)

        for ax in (ax_xy, ax_xz):
            obstacle = plt.Circle((0.0, 0.0), 10.0, color="black", fill=False, lw=1.1, alpha=0.7)
            ax.add_patch(obstacle)
            ax.set_xlim(-100.0, 100.0)
            ax.set_ylim(-100.0, 100.0)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(markerscale=8)

        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.set_title("Transverse deflection in flow-B plane")

        ax_xz.set_xlabel("x [m]")
        ax_xz.set_ylabel("z [m]")
        ax_xz.set_title("Motion along dipole axis")

        ax_uy.axhline(0.0, color="0.3", lw=0.8)
        ax_uy.set_xlabel("x [m]")
        ax_uy.set_ylabel("<uy>")
        ax_uy.set_title("Mean y momentum vs flow position")
        ax_uy.legend()

        ax_uz.axhline(0.0, color="0.3", lw=0.8)
        ax_uz.set_xlabel("x [m]")
        ax_uz.set_ylabel("<uz>")
        ax_uz.set_title("Mean z momentum vs flow position")
        ax_uz.legend()

        fig.suptitle(f"Solar-wind magnetic obstacle, iteration {iteration}")
        fig.savefig(args.output_dir / f"solar_wind_deflection_{iteration:06d}.png", dpi=180)
        plt.close(fig)
        summaries.append("")

    (args.output_dir / "deflection_summary.txt").write_text("\n".join(summaries))


if __name__ == "__main__":
    main()
