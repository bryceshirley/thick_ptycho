import os
import shutil

import numpy as np

from thick_ptycho.forward_model import (PWEIterativeLUSolver, PWEFullPinTSolver,
                                        MSForwardModelSolver)
from thick_ptycho.forward_model.pwe.operators import BoundaryType
from thick_ptycho.simulation.config import (ProbeConfig, ProbeType,
                                            SimulationConfig)
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes
from thick_ptycho.simulation.scan_frame import Limits
from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.utils.io import (get_git_commit_hash, load_config,
                                   results_dir_name)
from thick_ptycho.utils.visualisations import Visualisation


def main(cfg_path="config_gen_data.yaml"):
    cfg = load_config(cfg_path)
    results_dir = results_dir_name()

    shutil.copy(cfg_path, os.path.join(results_dir, os.path.basename(cfg_path)))
    with open(os.path.join(results_dir, "git_commit.txt"), "w") as f:
        f.write(get_git_commit_hash() + "\n")

    probe_config = ProbeConfig(
        type=ProbeType(cfg["probe_type"]),
        wave_length = float(cfg["wave_length"]),        # meters (0.635 μm). Visible light
        diameter = float(cfg["probe_diameter"]),               # [m]
        focus = float(cfg["probe_focus"]),                 # focal length [m]
        tilts=cfg["probe_tilts"] # tilts in degrees
    )

    # if cfg does not have ylims
    # if cfg["ylims"] is None:
    limits = Limits(x=tuple(float(x) for x in cfg["xlims"]), z=tuple(float(z) for z in cfg["zlims"]), units="meters")
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
        tomographic_projection_90_degree=cfg["solver"]["rotate90"],

        # Refractive index or Transmission Function Constant Surrounding Medium
        medium=cfg["n_medium"], # 1.0 for free space

        # Logging and results
        results_dir=results_dir,
        use_logging=True
    )

    simulation_space = create_simulation_space(sim_config)
    ptycho_object = create_ptycho_object(simulation_space)

    delta = float(cfg["delta"])
    beta = float(cfg["beta"])
    refractive_index_perturbation = complex(-delta, -beta)

    for obj_cfg in cfg["sample_space_objects"]:
        ptycho_object.add_object(
            obj_cfg["type"],
            refractive_index_perturbation,
            side_length_factor=obj_cfg["side_length_factor"],
            centre_factor=obj_cfg["centre_factor"],
            depth_factor=obj_cfg["depth_factor"],
            gaussian_blur=cfg["gaussian_blur"],
        )

    ptycho_object.build_field()
    simulation_space.viewer.plot_two_panels(ptycho_object.refractive_index,
                                            title="True Object Refractive Index",
                                            xlabel="z (px)",
                                            ylabel="x (px)",
                                            view="phase_amp",
                                            filename= "true_object.png")

    ptycho_probes = create_ptycho_probes(simulation_space)

    solver_type = cfg["solver"]["solver_type"]
    if solver_type == "iterativePWE":
        forward = PWEIterativeLUSolver(simulation_space, ptycho_probes,
                                      bc_type=cfg["solver"]["bc_type"])
    elif solver_type == "FullPintPWE":
        forward = PWEFullPinTSolver(simulation_space, ptycho_probes,
                                      bc_type=cfg["solver"]["bc_type"])
    elif solver_type == "MS":
        forward = MSForwardModelSolver(simulation_space, ptycho_probes)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    u = forward.solve(n=ptycho_object.refractive_index)
    data = forward.get_exit_waves(u)
    simulation_space.viewer.plot_two_panels(data.T,
                                        view="phase_amp", 
                                        title="Exit Waves",
                                        xlabel="Scan Number",
                                        ylabel="x (px)",
                                        filename="exit_waves.png")
    if not cfg["solver"]["phase"]:
        data = forward.get_farfield_intensities(exit_waves=data, 
                                                 poisson_noise=cfg["solver"]["poisson_noise"])
        simulation_space.viewer.plot_single_panel(np.fft.fftshift(data).T,
                                            title="Far-field Intensities",
                                            xlabel="Scan Number",
                                            ylabel="x (px)",
                                            filename="farfield_intensities.png")
    
    # Save data
    np.savez_compressed(
        os.path.join(results_dir, "simulated_data.npz"),
        data=data,
    )


if __name__ == "__main__":
    main()
