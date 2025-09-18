from pathlib import Path
import numpy as np
import yaml
import shutil
from datetime import datetime
import subprocess

from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.reconstruction.least_squares import LeastSquaresSolver
from thick_ptycho.utils.visualisations import Visualisation


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_nz(zlims_m: list[float], wavelength_m: float, axial_fraction: float) -> int:
    """nz = int( (z_max - z_min) / (wavelength * axial_fraction) )."""
    z_range = zlims_m[1] - zlims_m[0]
    dz = wavelength_m * axial_fraction
    nz = int(z_range / dz)
    return max(nz, 1)


def get_git_commit_hash() -> str:
    """Return current git commit hash, or 'unknown' if not available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def main(cfg_path: str = "config.yaml") -> None:
    cfg_path = Path(cfg_path)
    cfg = load_config(cfg_path)

    # === Results folder (timestamped) ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config file for reproducibility
    config_copy_path = run_dir / cfg_path.name
    shutil.copy(cfg_path, config_copy_path)
    print(f"Config copied to: {config_copy_path.resolve()}")

    # Save git commit hash
    commit_hash = get_git_commit_hash()
    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(commit_hash + "\n")
    print(f"Git commit hash saved: {commit_hash}")

    # === Physical params ===
    bc_type: str = cfg["bc_type"]
    probe_type: str = cfg["probe_type"]

    wavelength = float(cfg["wavelength_m"])
    k0 = 2 * np.pi / wavelength

    nb = float(cfg["nb"])
    delta = float(cfg["delta"])
    beta = float(cfg["beta"])
    refractive_index_perturbation = -delta - 1j * beta

    # === Continuous domain (meters) ===
    xlims = list(map(float, cfg["xlims_m"]))
    zlims = list(map(float, cfg["zlims_m"]))
    axial_fraction = float(cfg.get("axial_step_from_wavelength_fraction", 0.5))

    # === Discrete / sampling ===
    probe_dimensions = list(map(int, cfg["probe_dimensions_px"]))
    probe_diameter = int(float(cfg["probe_diameter_fraction"]) * min(probe_dimensions))
    scan_points = int(cfg["scan_points"])
    step_size = int(cfg["step_size_px"])
    probe_focus = float(cfg["probe_focus_m"])

    probe_angles_list = [float(x) for x in cfg.get("probe_angles_list", [0.0])]

    # === Derived grid sizes ===
    nz = compute_nz(zlims, wavelength, axial_fraction)
    print(f"nz={nz}")

    min_nx = int(scan_points * step_size + probe_dimensions[0])
    print(f"min_nx={min_nx}")

    nx = nz if nz >= min_nx else min_nx
    discrete_dimensions = [nx, nz]

    # === Build SampleSpace ===
    sample_space = SampleSpace(
        [xlims, zlims],
        discrete_dimensions,
        probe_dimensions,
        scan_points,
        step_size,
        bc_type,
        probe_type,
        k0,
        probe_diameter=probe_diameter,
        probe_focus=probe_focus,
        n_medium=nb,
    )

    sample_space.summarize_sample_space()
    visualisation = Visualisation(sample_space)

    # === Add objects ===
    gaussian_blur = int(cfg["gaussian_blur_px"])
    circles = cfg.get("circles", [])

    x_max, z_max = xlims[1], zlims[1]

    for obj in circles:
        if obj["type"] != "circle":
            continue
        side_length = float(obj["side_length_fraction"]) * x_max
        cx = float(obj["centre_fraction"][0]) * x_max
        cz = float(obj["centre_fraction"][1]) * z_max
        depth = float(obj["depth_fraction"]) * z_max

        sample_space.add_object(
            "circle",
            refractive_index_perturbation,
            side_length=side_length,
            centre=(cx, cz),
            depth=depth,
            gaussian_blur=gaussian_blur,
        )

    sample_space.generate_sample_space()

    # === Solver ===
    solver_cfg = cfg["solver"]
    least_squares = LeastSquaresSolver(
        sample_space,
        full_system_solver=bool(solver_cfg["full_system_solver"]),
        probe_angles_list=probe_angles_list,
    )

    reconstructed_sample_space, reconstructed_wave, residual_history = least_squares.solve(
        max_iters=int(solver_cfg["max_iters"]),
        plot_forward=bool(solver_cfg["plot_forward"]),
        plot_reverse=bool(solver_cfg["plot_reverse"]),
        plot_object=bool(solver_cfg["plot_object"]),
        solve_probe=bool(solver_cfg["solve_probe"]),
        sparsity_lambda=float(solver_cfg["sparsity_lambda"]),
        low_pass_filter=float(solver_cfg["low_pass_filter"]),
    )

    # Save residual history as a numpy file in run_dir
    np.save(run_dir / "residual_history.npy", residual_history)


if __name__ == "__main__":
    main()
