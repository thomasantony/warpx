import math
import tempfile
import unittest
from pathlib import Path

from SolarWindRMF import solar_wind_rmf_tools as rmf


BASE = Path(__file__).resolve().parents[2] / "SolarWindRMF" / "solar_wind_rmf_explicit_coils.txt"


class SolarWindRmfToolsTest(unittest.TestCase):
    def assertQuietRuntimeOverrides(self, text):
        self.assertIn("warpx.verbose = 0", text)
        self.assertIn("tiny_profiler.enabled = false", text)
        self.assertIn("tiny_profiler.memprof_enabled = false", text)
        self.assertIn("tiny_profiler.device_synchronize_around_region = false", text)

    def test_vacuum_twin_removes_plasma_and_preserves_drive(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair = rmf.write_vacuum_pair(BASE, Path(tmp), "smoke")

            plasma = pair.plasma_input.read_text()
            vacuum = pair.vacuum_input.read_text()

        self.assertIn("particles.species_names = electrons protons", plasma)
        self.assertNotIn("particles.species_names", vacuum)
        self.assertNotIn("electrons.", vacuum)
        self.assertNotIn("protons.", vacuum)
        self.assertIn("warpx.My_external_grid_function(x,y,z,t)", vacuum)
        self.assertIn("warpx.Mz_external_grid_function(x,y,z,t)", vacuum)
        self.assertNotIn("rho_electrons", vacuum)
        self.assertNotIn("rho_protons", vacuum)
        self.assertIn("diag1.file_prefix = diags/smoke_plasma", plasma)
        self.assertIn("diag1.file_prefix = diags/smoke_vacuum", vacuum)
        self.assertQuietRuntimeOverrides(plasma)
        self.assertQuietRuntimeOverrides(vacuum)

    def test_vacuum_coil_validation_encodes_two_period_probe_and_cleaning_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = rmf.write_vacuum_coil_validation(BASE, Path(tmp))
            text = case.read_text()

        constants = rmf.parse_constants(text)
        expected_steps = round(2.0 / (constants["f_rmf"] * constants["dt"]))

        self.assertIn("amr.n_cell = 64 64 64", text)
        self.assertIn(f"max_step = {expected_steps}", text)
        self.assertNotIn("electrons.", text)
        self.assertNotIn("protons.", text)
        # The RMF drive is J_ext=curl(M_ext), so this source is
        # divergence-free on the Yee mesh by construction; see
        # test_jext_curl_divergence.py and jext_impl_02.md.
        self.assertIn("warpx.do_dive_cleaning = 0", text)
        self.assertIn("warpx.do_divb_cleaning = 0", text)
        self.assertIn("warpx.reduced_diags_names = CenterProbe", text)
        self.assertIn("CenterProbe.probe_geometry = Point", text)
        self.assertNotIn("CenterProbe.interp_order = 0", text)
        self.assertQuietRuntimeOverrides(text)
        self.assertTrue(
            math.isclose(
                rmf.center_field_calibration_factor(
                    b0=constants["B0"],
                    r_coil=constants["R_coil"],
                    i_coil=constants["I_coil"],
                ),
                1.0 / constants["center_field_geometry_factor"],
                rel_tol=1.0e-12,
            )
        )
        self.assertIn(
            "my_constants.A_coil = I_coil/(sqrt(2.0*pi)*w_coil*g0_coil)", text
        )
        self.assertNotIn("J0_coil", text)
        self.assertIn("target center |B_perp| should be B0", text)
        g0 = 0.5 * (1.0 + math.tanh(constants["R_coil"] / constants["w_coil"]))
        self.assertTrue(
            math.isclose(
                constants["A_coil"],
                constants["I_coil"] / (math.sqrt(2.0 * math.pi) * constants["w_coil"] * g0),
                rel_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                rmf.center_field_target_factor(
                    b0=constants["B0"],
                    r_coil=constants["R_coil"],
                    i_coil=constants["I_coil"],
                    center_field_geometry_factor=constants["center_field_geometry_factor"],
                ),
                1.0,
                rel_tol=1.0e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                constants["I_coil"],
                2.0
                * constants["R_coil"]
                * constants["B0"]
                / (rmf.MU0 * constants["center_field_geometry_factor"]),
                rel_tol=1.0e-12,
            )
        )

    def test_single_electron_validation_places_particle_at_half_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = rmf.write_single_electron_validation(BASE, Path(tmp))
            text = case.read_text()

        self.assertIn("particles.species_names = electron", text)
        self.assertIn('electron.injection_style = "SingleParticle"', text)
        self.assertIn("electron.single_particle_pos = 0.0 10.0 0.0", text)
        self.assertIn("electron.single_particle_u = 0.0 0.0 0.0", text)
        self.assertIn("diag1.species = electron", text)
        self.assertQuietRuntimeOverrides(text)

    def test_benchmark_input_matches_requested_size_ppc_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = rmf.write_benchmark_input(BASE, Path(tmp), precision_label="fp64_a100")
            text = case.read_text()

        self.assertIn("amr.n_cell = 128 128 128", text)
        self.assertIn("max_step = 2000", text)
        self.assertIn(
            "electrons.initial_electrons.num_particles_per_cell_each_dim = 2 2 2",
            text,
        )
        self.assertIn(
            "protons.initial_protons.num_particles_per_cell_each_dim = 2 2 2",
            text,
        )
        self.assertIn(
            "diag1.file_prefix = diags/solar_wind_rmf_benchmark_fp64_a100",
            text,
        )
        self.assertQuietRuntimeOverrides(text)


if __name__ == "__main__":
    unittest.main()
