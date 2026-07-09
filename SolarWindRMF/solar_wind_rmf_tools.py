#!/usr/bin/env python3
"""Generate paired and validation inputs for the solar-wind RMF tutorial."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


MU0 = 4.0e-7 * math.pi
CLIGHT = 299792458.0
QE = 1.602176634e-19
ME = 9.1093837015e-31
MP = 1.67262192369e-27


@dataclass(frozen=True)
class VacuumPair:
    plasma_input: Path
    vacuum_input: Path
    manifest: Path


def _replace_or_append(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    line = f"{key} = {value}"
    if pattern.search(text):
        return pattern.sub(line, text)
    return f"{text.rstrip()}\n{line}\n"


def _set_many(text: str, updates: dict[str, str]) -> str:
    for key, value in updates.items():
        text = _replace_or_append(text, key, value)
    return text


def _remove_keys(text: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        pattern = re.compile(rf"^{re.escape(key)}\s*=.*\n?", re.MULTILINE)
        text = pattern.sub("", text)
    return text


def _remove_prefixes(text: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        pattern = re.compile(rf"^{re.escape(prefix)}\..*\n?", re.MULTILINE)
        text = pattern.sub("", text)
    return text


def _with_quiet_runtime_overrides(text: str) -> str:
    return _set_many(
        text,
        {
            "warpx.verbose": "0",
            "tiny_profiler.enabled": "false",
            "tiny_profiler.memprof_enabled": "false",
            "tiny_profiler.device_synchronize_around_region": "false",
        },
    )


def _read_base(base_input: Path) -> str:
    return base_input.read_text()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")
    return path


def parse_constants(text: str) -> dict[str, float]:
    constants = {
        "pi": math.pi,
        "mu0": MU0,
        "clight": CLIGHT,
        "q_e": QE,
        "m_e": ME,
        "m_p": MP,
        "sqrt": math.sqrt,
        "tanh": math.tanh,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "abs": abs,
    }
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"\s*my_constants\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if not match:
            continue
        name, expr = match.groups()
        env = {**constants, **parsed}
        parsed[name] = float(eval(expr, {"__builtins__": {}}, env))
    return parsed


def center_field_calibration_factor(*, b0: float, r_coil: float, i_coil: float) -> float:
    return MU0 * i_coil / (2.0 * r_coil * b0)


def center_field_target_factor(
    *,
    b0: float,
    r_coil: float,
    i_coil: float,
    center_field_geometry_factor: float,
) -> float:
    return center_field_geometry_factor * center_field_calibration_factor(
        b0=b0,
        r_coil=r_coil,
        i_coil=i_coil,
    )


def _two_period_steps(constants: dict[str, float]) -> int:
    return int(round(2.0 / (constants["f_rmf"] * constants["dt"])))


def _with_common_rmf_overrides(text: str, *, prefix: str) -> str:
    return _with_quiet_runtime_overrides(
        _set_many(
            text,
            {
                "warpx.do_dive_cleaning": "0",
                "warpx.do_divb_cleaning": "0",
                "diag1.file_prefix": f"diags/{prefix}",
            },
        )
    )


def _without_plasma(text: str, *, prefix: str) -> str:
    return _remove_prefixes(
        _remove_keys(
            _set_many(
                _with_common_rmf_overrides(text, prefix=prefix),
                {
                    "diag1.fields_to_plot": "Ex Ey Ez Bx By Bz jx jy jz rho",
                    "diag1.write_species": "0",
                },
            ),
            (
                "particles.species_names",
            ),
        ),
        ("electrons", "protons"),
    )


def write_vacuum_pair(base_input: Path, output_dir: Path, stem: str = "solar_wind_rmf") -> VacuumPair:
    base = _read_base(base_input)
    plasma = _with_common_rmf_overrides(base, prefix=f"{stem}_plasma")
    vacuum = _without_plasma(base, prefix=f"{stem}_vacuum")

    plasma_path = _write(output_dir / f"{stem}_plasma.txt", plasma)
    vacuum_path = _write(output_dir / f"{stem}_vacuum.txt", vacuum)
    manifest_path = output_dir / f"{stem}_pair_manifest.json"
    manifest = {
        "observable": "delta_B = B_plasma - B_vacuum",
        "base_input": str(base_input),
        "plasma_input": str(plasma_path),
        "vacuum_input": str(vacuum_path),
        "field_components": ["Bx", "By", "Bz"],
    }
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return VacuumPair(plasma_path, vacuum_path, manifest_path)


def write_ab_cleaning_pair(base_input: Path, output_dir: Path) -> VacuumPair:
    base = _read_base(base_input)
    cleaning_on = _with_quiet_runtime_overrides(
        _set_many(
            base,
            {
                "warpx.do_dive_cleaning": "1",
                "warpx.do_divb_cleaning": "1",
                "diag1.file_prefix": "diags/solar_wind_rmf_cleaning_on",
            },
        )
    )
    cleaning_off = _with_quiet_runtime_overrides(
        _set_many(
            base,
            {
                "warpx.do_dive_cleaning": "0",
                "warpx.do_divb_cleaning": "0",
                "diag1.file_prefix": "diags/solar_wind_rmf_cleaning_off",
            },
        )
    )
    on_path = _write(output_dir / "solar_wind_rmf_cleaning_on.txt", cleaning_on)
    off_path = _write(output_dir / "solar_wind_rmf_cleaning_off.txt", cleaning_off)
    manifest_path = output_dir / "solar_wind_rmf_cleaning_ab_manifest.json"
    manifest = {
        "comparison": "Confirm fields match before relying on cleaning_off.",
        "cleaning_on_input": str(on_path),
        "cleaning_off_input": str(off_path),
        "field_components": ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
    }
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return VacuumPair(on_path, off_path, manifest_path)


def _with_center_probe(text: str) -> str:
    return (
        text.rstrip()
        + "\n\n"
        + "warpx.reduced_diags_names = CenterProbe\n"
        + "CenterProbe.type = FieldProbe\n"
        + "CenterProbe.intervals = 1\n"
        + "CenterProbe.integrate = 0\n"
        + "CenterProbe.probe_geometry = Point\n"
        + "CenterProbe.x_probe = 0.0\n"
        + "CenterProbe.y_probe = 0.0\n"
        + "CenterProbe.z_probe = 0.0\n"
    )


def write_vacuum_coil_validation(base_input: Path, output_dir: Path) -> Path:
    base = _read_base(base_input)
    constants = parse_constants(base)
    text = _without_plasma(base, prefix="solar_wind_rmf_vacuum_coil_64")
    text = _set_many(
        text,
        {
            "amr.n_cell": "64 64 64",
            "max_step": str(_two_period_steps(constants)),
            "diag1.intervals": "0:100000000:1000",
            "diag1.dump_last_timestep": "1",
        },
    )
    text = _with_center_probe(text)
    text += (
        "\n# Validation: target center |B_perp| should be B0.\n"
        "# The input scales I_coil and A_coil by 1/center_field_geometry_factor\n"
        "# so the tanh-shell M_ext geometry factor is already accounted for.\n"
        "# Validation: CenterProbe By/Bz phase gives RMF rotation sense and frequency.\n"
        "# Validation: late-time edge fields should not show PML ringing.\n"
    )
    return _write(output_dir / "solar_wind_rmf_vacuum_coil_64.txt", text)


def write_single_electron_validation(base_input: Path, output_dir: Path) -> Path:
    base = _read_base(base_input)
    constants = parse_constants(base)
    text = _without_plasma(base, prefix="solar_wind_rmf_single_electron")
    text = _set_many(
        text,
        {
            "particles.species_names": "electron",
            "diag1.fields_to_plot": "Ex Ey Ez Bx By Bz jx jy jz rho",
            "diag1.species": "electron",
            "electron.charge": "-q_e",
            "electron.mass": "m_e",
            "electron.injection_style": '"SingleParticle"',
            "electron.single_particle_pos": f"0.0 {0.5 * constants['R_coil']:.1f} 0.0",
            "electron.single_particle_u": "0.0 0.0 0.0",
            "electron.single_particle_weight": "1.0",
        },
    )
    text += "\n# Validation: particle history should show azimuthal drift in the RMF rotation sense.\n"
    return _write(output_dir / "solar_wind_rmf_single_electron.txt", text)


def write_benchmark_input(base_input: Path, output_dir: Path, precision_label: str = "fp64_a100") -> Path:
    base = _read_base(base_input)
    text = _with_common_rmf_overrides(base, prefix=f"solar_wind_rmf_benchmark_{precision_label}")
    text = _set_many(
        text,
        {
            "amr.n_cell": "128 128 128",
            "max_step": "2000",
            "electrons.initial_electrons.num_particles_per_cell_each_dim": "2 2 2",
            "protons.initial_protons.num_particles_per_cell_each_dim": "2 2 2",
            "diag1.intervals": "2000",
            "diag1.dump_last_timestep": "1",
        },
    )
    text += "\n# Benchmark: extract particle-pushes/sec and recompute cost estimates from that number.\n"
    return _write(output_dir / f"solar_wind_rmf_benchmark_{precision_label}.txt", text)


def write_all(base_input: Path, output_dir: Path) -> None:
    write_vacuum_pair(base_input, output_dir)
    write_ab_cleaning_pair(base_input, output_dir)
    write_vacuum_coil_validation(base_input, output_dir)
    write_single_electron_validation(base_input, output_dir)
    write_benchmark_input(base_input, output_dir, "fp64_a100")
    write_benchmark_input(base_input, output_dir, "fp32_5090")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).with_name("solar_wind_rmf_explicit_coils.txt"),
        help="Base RMF coil input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("SolarWindRMF/generated_rmf"),
        help="Directory for generated inputs.",
    )
    args = parser.parse_args()
    write_all(args.base, args.output_dir)


if __name__ == "__main__":
    main()
