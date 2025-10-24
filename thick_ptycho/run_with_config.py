import os
import numpy as np
import shutil
from typing import Dict, Any, List, Tuple

from thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.reconstruction.pwe_reconstructor import LeastSquaresSolver
from thick_ptycho.utils.visualisations import Visualisation
from thick_ptycho.utils.utils import load_config, get_git_commit_hash, results_dir_name


# -----------------------
# Helpers
# -----------------------

def _scale_centre(centre_fraction, x_max: float, z_max: float) -> Tuple[float, float]:
    cx = float(centre_fraction[0]) * x_max
    cz = float(centre_fraction[1]) * z_max
    return (cx, cz)


# -----------------------
# Config SampleSpace
# -----------------------

def setup_sample_space(
    cfg: Dict[str, Any], results_dir: str
) -> Tuple[SampleSpace, Visualisation, List[float]]:
    """
    Parse all sample-space-related config, compute derived dimensions,
    build the SampleSpace, add objects, generate it

    Returns
    -------
    sample_space : SampleSpace
    probe_angles_list : List[float]  # passed to solver later
    """
    # === Physical params ===
    bc_type: str = cfg["bc_type"]
    probe_type: str = cfg["probe_type"]

    wl_raw = cfg.get("wavelength", None)
    if wl_raw is None:
        raise ValueError("Config must include 'wavelength' (meters).")
    wavelength = float(wl_raw)
    k0 = 2 * np.pi / wavelength

    nb = float(cfg["nb"])
    delta = float(cfg["delta"])
    beta = float(cfg["beta"])
    refractive_index_perturbation = -delta - 1j * beta

    # Spatial dimensions
    xlims = [float(v) for v in cfg.get("xlims")]
    zlims = [float(v) for v in cfg.get("zlims")]
    if "ylims" in cfg:
        ylims = [float(v) for v in cfg.get("ylims")]
        continuous_dimensions = [xlims, ylims, zlims]
    else:
        ylims = None
        continuous_dimensions = [xlims, zlims]

    # z resolution
    z_steps_per_lambda = float(cfg["z_step_per_wavelength"])
    if z_steps_per_lambda <= 0:
        raise ValueError("'z_step_per_wavelength' must be > 0.")
    dz = wavelength / z_steps_per_lambda
    z_range = zlims[1] - zlims[0]
    nz = max(1, int(z_range / dz))

    # Discrete/probe params
    probe_dimensions = list(map(int, cfg["probe_dimensions_px"]))
    probe_diameter = int(float(cfg["probe_diameter_fraction"]) * min(probe_dimensions))
    scan_points = int(cfg["scan_points"])
    step_size = int(cfg["step_size_px"])
    probe_focus = float(cfg["probe_focus"])
    probe_angles_list = [float(x) for x in cfg.get("probe_angles_list", [0.0])]

    min_nx = int(scan_points * step_size + probe_dimensions[0])
    #nx = nz if (cfg["solver"]["rotate90"] and nz >= min_nx) else min_nx
    nx = nz if nz >= min_nx else min_nx

    discrete_dimensions = [nx, nz]

    # === Build SampleSpace ===
    sample_space = SampleSpace(
        continuous_dimensions=continuous_dimensions,
        discrete_dimensions=discrete_dimensions,
        probe_dimensions=probe_dimensions,
        scan_points=scan_points,
        step_size=step_size,
        bc_type=bc_type,
        probe_type=probe_type,
        wave_number=k0,
        probe_diameter=probe_diameter,
        probe_focus=probe_focus,
        n_medium=nb,
        results_dir=results_dir,
    )

    # Add objects
    gaussian_blur = int(cfg.get("gaussian_blur_px", 0))
    sample_space_objects = cfg.get("sample_space_objects", [])

    x_max, z_max = xlims[1], zlims[1]
    for obj in sample_space_objects:
        obj_type = obj.get("type")
        centre = _scale_centre(obj["centre_fraction"], x_max, z_max)
        depth = float(obj["depth_fraction"]) * z_max
        side_length = float(obj["side_length_fraction"]) * x_max

        sample_space.add_object(
            obj_type,
            refractive_index_perturbation,
            gaussian_blur=gaussian_blur,
            centre=centre,
            depth=depth,
            side_length=side_length,
        )

    # Generate field & create Visualisation
    sample_space.generate_sample_space()

    # Optional quick summary
    sample_space.summarize_sample_space()

    return sample_space, probe_angles_list

# -----------------------
# Main
# -----------------------

def main(cfg_path: str = "config.yaml") -> None:
    cfg_path = os.path.abspath(cfg_path)
    cfg = load_config(cfg_path)

    # Results folder
    results_dir = results_dir_name()

    # Save config & commit
    shutil.copy(cfg_path, os.path.join(results_dir, os.path.basename(cfg_path)))
    commit_hash = get_git_commit_hash()
    with open(os.path.join(results_dir, "git_commit.txt"), "w") as f:
        f.write(commit_hash + "\n")
    print(f"Results: {results_dir}\nGit commit: {commit_hash}")

    # Build sample space + viz in one go
    sample_space, probe_angles_list = setup_sample_space(cfg, results_dir)

    # Solver
    solver_params = cfg["solver"]
    solver = LeastSquaresSolver(
        sample_space,
        full_system_solver=solver_params["full_system_solver"],
        probe_angles_list=probe_angles_list,
        rotate90=solver_params["rotate90"],
        results_dir=results_dir,
        poisson_noise=solver_params["poisson_noise"]
    )

    reconstructed_refractive_index, reconstructed_forward_wave, residual_history = solver.solve(
        known_phase=solver_params["known_phase"],
        max_iters=solver_params["max_iters"],
        plot_forward=solver_params["plot_forward"],
        plot_reverse=solver_params["plot_reverse"],
        solve_probe=solver_params["solve_probe"],
        sparsity_lambda=solver_params["sparsity_lambda"],
        low_pass_filter=solver_params["low_pass_filter"],
    )

    # Plots
    visualisation = Visualisation(sample_space, results_dir=results_dir)
    visualisation.plot_residual(residual_history)
    visualisation.plot_refractive_index(
        reconstructed_refractive_index, title="Reconstructed Sample Space"
    )
    visualisation.plot_auto(
        reconstructed_forward_wave[0],
        view="phase_amp",
        layout="single",
        title_prefix="reconstructed_forward_wave ",
    )
    np.save(os.path.join(results_dir, 'reconstructed_refractive_index.npy'), reconstructed_refractive_index)
    np.save(os.path.join(results_dir, 'reconstructed_forward_wave'), reconstructed_refractive_index)


if __name__ == "__main__":
    main()
