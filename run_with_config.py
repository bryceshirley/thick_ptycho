import os
import numpy as np
from thick_ptycho.simulation.config import SimulationConfig, BoundaryType, ProbeType
from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes
from thick_ptycho.forward_model import PWEIterativeLUSolver
from thick_ptycho.utils.io import load_config, results_dir_name, get_git_commit_hash
from thick_ptycho.utils.visualisations import Visualisation
import shutil


def main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    results_dir = results_dir_name()

    shutil.copy(cfg_path, os.path.join(results_dir, os.path.basename(cfg_path)))
    with open(os.path.join(results_dir, "git_commit.txt"), "w") as f:
        f.write(get_git_commit_hash() + "\n")

    wavelength = float(cfg["wavelength"])
    k0 = 2 * np.pi / wavelength

    xlims = cfg["xlims"]
    zlims = cfg["zlims"]
    continuous_dimensions = (xlims, zlims)

    # Determine NZ
    z_range = zlims[1] - zlims[0]
    dz = wavelength / cfg["z_step_per_wavelength"]
    nz = int(z_range / dz)

    probe_dim = cfg["probe_dimensions_px"][0]
    min_nx = (cfg["scan_points"] - 1) * cfg["step_size_px"] + probe_dim
    nx = max(min_nx, nz)  # ensure square for rotate90

    discrete_dimensions = (nx, nz)

    config = SimulationConfig(
        continuous_dimensions=continuous_dimensions,
        discrete_dimensions=discrete_dimensions,
        probe_dimensions=probe_dim,
        scan_points=cfg["scan_points"],
        step_size=cfg["step_size_px"],
        probe_angles=tuple(cfg["probe_angles_list"]),
        wave_number=k0,
        probe_diameter_scale=cfg["probe_diameter_fraction"],
        probe_focus=cfg["probe_focus"],
        bc_type=BoundaryType(cfg["bc_type"]),
        probe_type=ProbeType(cfg["probe_type"]),
        n_medium=cfg["nb"],
        results_dir=results_dir,
    )

    simulation_space = create_simulation_space(config)
    ptycho_object = create_ptycho_object(simulation_space)

    delta = float(cfg["delta"])
    beta = float(cfg["beta"])

    for obj_cfg in cfg["sample_space_objects"]:
        ptycho_object.add_object(
            obj_cfg["type"],
            -delta - 1j * beta,
            side_length=obj_cfg["side_length_fraction"],
            centre=obj_cfg["centre_fraction"],
            depth=obj_cfg["depth_fraction"],
            gaussian_blur=cfg["gaussian_blur_px"],
        )

    ptycho_object.build_field()
    ptycho_probes = create_ptycho_probes(simulation_space)

    forward = PWEIterativeLUSolver(simulation_space, ptycho_object, ptycho_probes)
    u = forward.solve()

    vis = Visualisation(simulation_space, results_dir)
    vis.plot_two_panels(ptycho_object.n_true, view="phase_amp")


if __name__ == "__main__":
    main()
