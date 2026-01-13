import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

def resample_simulation_data(
    exit_waves, 
    ref_index, 
    loaded_sim_config, 
    simulation_space, 
    new_empty_space_px,
    seed=42
):
    """
    Resizes and re-pads exit waves and refractive index maps to match a 
    new simulation grid.
    """
    # 1. Extract dimensions
    original_padding = loaded_sim_config["empty_space_px"]
    original_width = exit_waves.shape[1]
    
    # 2. Crop back to the "Clean" sample area
    # This removes the old padding to establish a clean baseline
    ew_clean = exit_waves[:, original_padding:-original_padding]
    clean_width = ew_clean.shape[1]
    
    print(f"Original shape: {exit_waves.shape}, Clean width: {clean_width}")

    # 3. Calculate new padding based on the ratio logic
    # We want to match the target simulation_space.effective_nx
    new_ratio = new_empty_space_px / simulation_space.effective_nx
    old_ratio = original_padding / clean_width
    print(f"Old ratio: {old_ratio}, New ratio: {new_ratio}")
    update_padding_px = int(clean_width * new_ratio)
    
    # 4. Pad the clean data to the temporary "ratio-matched" size
    ew_padded = np.pad(
        ew_clean,
        pad_width=((0, 0), (update_padding_px, update_padding_px)),
        mode="constant",
        constant_values=0.0,
    )

    # 5. Zoom Exit Waves to match the new grid resolution
    target_ew_shape = (ew_padded.shape[0], simulation_space.effective_shape[0])
    ew_zoom_factors = np.array(target_ew_shape) / np.array(ew_padded.shape)
    ew_zoomed = zoom(ew_padded, ew_zoom_factors, order=1)

    # 6. Apply final slice (the 'v_slice' logic)
    v_slice = slice(new_empty_space_px, -new_empty_space_px) if new_empty_space_px > 0 else slice(None)
    data_ew = ew_zoomed[:, v_slice]

    # Validation
    if data_ew.shape[1] != simulation_space.effective_nx:
        raise ValueError(
            f"Shape mismatch: Calculated {data_ew.shape[1]} but expected {simulation_space.effective_nx}"
        )

    # 7. Zoom Refractive Index
    # Matches target (nx, nz)
    ref_target_shape = (simulation_space.effective_nx, simulation_space.nz)
    ref_zoom_factors = np.array(ref_target_shape) / np.array(ref_index.shape)
    ref_index_zoomed = zoom(ref_index, ref_zoom_factors, order=1)

    # 8. Physics: Generate Diffraction Patterns and Noise
    data_clean = np.abs(np.fft.fft(data_ew, axis=-1)) ** 2
    rng = np.random.default_rng(seed=seed)
    data_noisy = rng.poisson(data_clean).astype(float)

    return {
        "exit_waves": data_ew,
        "ref_index": ref_index_zoomed,
        "diffraction_clean": data_clean,
        "diffraction_noisy": data_noisy
    }


def plot_solution(
    simulation_space,
    solution: np.ndarray,
):
    """Plot results from forward model simulation."""
    # Select Centre Probe
    simulation_space.viewer.plot_two_panels(
        solution[0, 0, simulation_space.num_probes // 2, ...],
        view="phase_amp",
        title="Wavefield Solution PWE",
        xlabel="z (px)",
        ylabel="x (px)",
    )
    plt.show()


def plot_data(
    simulation_space,
    data: np.ndarray,
    probes: np.ndarray,
    exitwaves: np.ndarray = None,
):
    # Recover exit waves from far-field data
    if exitwaves is None:
        exitwaves = np.fft.ifft(np.fft.ifftshift(np.abs(np.sqrt(data)), axes=-1), axis=-1)
    else:
        exitwaves = np.abs(exitwaves)

    if simulation_space.num_probes > 1:

        simulation_space.viewer.plot_single_panel(
            exitwaves.T,  # Use ifftshift to move zero frequency to the corner
            title="Recovered Exit Waves",
            xlabel="Scan Number #",
            ylabel="x (px)",
        )

        simulation_space.viewer.plot_single_panel(
            np.fft.ifftshift(data).T,
            title="Far-field Intensities",
            xlabel="Scan Number #",
            ylabel="x (px)",
        )

    if simulation_space.num_probes > 1:
        plt.figure(figsize=(8, 4))
        plot_num_probes = 10  # simulation_space.num_probes
        probe_indices = np.linspace(
            0, simulation_space.num_probes - 1, plot_num_probes, dtype=int
        )
        for p in probe_indices:
            plt.plot(
                range(simulation_space.effective_shape[0]),
                np.abs(probes[0, p, :]),
                label=f"probe {p}",
            )
        plt.title("Probe amplitudes (subset)")
        plt.xlabel("x (px)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        for p in probe_indices:
            plt.plot(
                range(exitwaves.shape[1]),
                np.abs(exitwaves[p, :]),
                label=f"probe {p}",
            )
        plt.title("Exitwave amplitudes (subset)")
        plt.xlabel("x (px)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 4))

        plt.plot(
            range(simulation_space.effective_shape[0]),
            np.abs(probes[0, simulation_space.num_probes // 2, :]),
            label=f"probe {simulation_space.num_probes // 2}",
        )
        plt.title("Probe amplitudes (subset)")
        plt.xlabel("x (px)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.plot(
            range(simulation_space.effective_shape[0]),
            np.abs(exitwaves[simulation_space.num_probes // 2, :]),
            label=f"probe {simulation_space.num_probes // 2}",
        )
        plt.title("Exitwave amplitudes (subset)")
        plt.xlabel("x (px)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def create_random_circle_object(
    simulation_space,
    ptycho_object,
    random_seed=5,
    delta=0.01,
    beta=0.001,
    delta_var=0.20,
    beta_var=0.20,
    num_circles=10,
    common_side_length=0.05,
    common_depth=0.05,
    gaussian_blur=3,
):
    """Create a random circular object in the simulation space."""

    np.random.seed(random_seed)

    # Generate random (x,z) centers
    x_centers = np.random.uniform(0.25, 0.75, size=num_circles)
    z_centers = np.random.uniform(0.1, 0.9, size=num_circles)

    for cx, cz in zip(x_centers, z_centers):
        # Randomize refractive index perturbation per circle
        delta_rand = delta * (1 + delta_var * (np.random.rand() - 0.5) * 2)
        beta_rand = beta * (1 + beta_var * (np.random.rand() - 0.5) * 2)

        refractive_index_perturbation = -delta_rand - 1j * beta_rand

        ptycho_object.add_object(
            "circle",
            refractive_index_perturbation,
            side_length_factor=common_side_length,
            centre_factor=(cx, cz),
            depth_factor=common_depth,
            gaussian_blur=gaussian_blur,
        )
    ptycho_object.build_field()
    simulation_space.viewer.plot_two_panels(
        ptycho_object.refractive_index,
        view="phase_amp",
        title="Random Circular Object",
        xlabel="z (px)",
        ylabel="x (px)",
    )
    return ptycho_object


def plot_probes(simulation_space, ptycho_probes):
    """Plot the ptychographic probes."""
    plt.figure(figsize=(8, 4))
    plot_num_probes = 10  # simulation_space.num_probes
    probe_indices = np.linspace(
        0, simulation_space.num_probes - 1, plot_num_probes, dtype=int
    )
    for p in probe_indices:
        plt.plot(
            range(simulation_space.effective_shape[0]),
            np.abs(ptycho_probes[0, p, :]),
            label=f"probe {p}",
        )
    plt.title("Probe amplitudes (subset)")
    plt.xlabel("x (px)")
    plt.legend()
    plt.tight_layout()
    plt.show()
