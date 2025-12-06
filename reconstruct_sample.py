import argparse
import os
import shutil

import numpy as np

from thick_ptycho.reconstruction.ms_reconstructor import ReconstructorMS
from thick_ptycho.reconstruction.pwe_reconstructor import ReconstructorPWE
from thick_ptycho.simulation.config import ProbeConfig, ProbeType, SimulationConfig
from thick_ptycho.simulation.scan_frame import Limits
from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.utils.io import get_git_commit_hash, load_config, results_dir_name


def main(cfg_path):
    cfg = load_config(cfg_path)
    results_dir = results_dir_name("recon")

    shutil.copy(cfg_path, os.path.join(results_dir, os.path.basename(cfg_path)))
    with open(os.path.join(results_dir, "git_commit.txt"), "w") as f:
        f.write(get_git_commit_hash() + "\n")

    probe_config = ProbeConfig(
        type=ProbeType(cfg["probe_type"]),
        wave_length=float(cfg["wave_length"]),  # meters (0.635 μm). Visible light
        diameter=float(cfg["probe_diameter"]),  # [m]
        focus=float(cfg["probe_focus"]),  # focal length [m]
        tilts=[float(x) for x in cfg["probe_tilts"]],  # tilts in degrees
    )

    # if cfg does not have ylims
    # if cfg["ylims"] is None:
    limits = Limits(
        x=tuple(float(x) for x in cfg["xlims"]),
        z=tuple(float(z) for z in cfg["zlims"]),
        units="meters",
    )
    # else:
    # limits = Limits(x=cfg["xlims"], y=cfg["ylims"], z=cfg["zlims"], units="m")

    sim_config = SimulationConfig(
        probe_config=probe_config,
        # Spatial discretization
        scan_points=cfg["scan_points"],
        step_size_px=cfg["step_size_px"],
        pad_factor=cfg["pad_factor"],
        solve_reduced_domain=cfg["solve_reduced_domain"],
        points_per_wavelength=cfg["points_per_wavelength"],
        spatial_limits=limits,
        tomographic_projection_90_degree=cfg["recon_solver"]["rotate90"],
        # Refractive index or Transmission Function Constant Surrounding Medium
        medium=cfg["n_medium"],  # 1.0 for free space
        # Logging and results
        results_dir=results_dir,
        use_logging=True,
    )

    simulation_space = create_simulation_space(sim_config)
    simulation_space.summarize()

    # Load data
    data_file_path = cfg["recon_solver"]["data_file_path"]
    data = np.load(data_file_path)
    if cfg["recon_solver"]["phase_retrieval"]:
        data = data["intensity"]  # intensity data
    else:
        data = data["exit_waves"]  # complex exit waves

    # Reconstruction
    solver_type = cfg["recon_solver"]["solver_type"]
    if solver_type == "PWE":
        recon = ReconstructorPWE(
            simulation_space,
            data,
            phase_retrieval=cfg["recon_solver"]["phase_retrieval"],
            bc_type=cfg["recon_solver"]["bc_type"],
        )
    elif solver_type == "MS":
        recon = ReconstructorMS(
            simulation_space,
            data,
            phase_retrieval=cfg["recon_solver"]["phase_retrieval"],
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    n_est, wave_field, residual_hist = recon.reconstruct(
        max_iters=cfg["recon_solver"]["max_iters"]
    )

    # Visualizations
    simulation_space.viewer.plot_two_panels(
        n_est,
        view="phase_amp",
        title="Reconstructed Object",
        filename="reconstructed_object.png",
    )
    simulation_space.viewer.plot_residual(
        residual_hist,
        title="Reconstruction Residual History",
        filename="reconstruction_residual.png",
    )

    # Save reconstructed refractive index
    n_est_file = os.path.join(results_dir, "reconstruct_data.npz")
    np.savez_compressed(
        n_est_file, n_est=n_est, wave_field=wave_field, residual_history=residual_hist
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run thick ptychography simulation with configuration file."
    )
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.config_file)
