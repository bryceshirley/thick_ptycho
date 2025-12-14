import numpy as np
import matplotlib.pyplot as plt


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
    exitwaves: np.ndarray,
    probes: np.ndarray,
):
    # Noisy data with Poisson noise

    if simulation_space.num_probes > 1:
        rng = np.random.default_rng(seed=42)
        data_noisy = data.copy() + rng.poisson(np.abs(data))

        simulation_space.viewer.plot_single_panel(
            exitwaves.T,  # Use ifftshift to move zero frequency to the corner
            title="Update Exit Waves PWE",
            xlabel="Scan Number #",
            ylabel="x (px)",
        )

        simulation_space.viewer.plot_single_panel(
            np.fft.fftshift(data).T,
            title="Far-field Intensities PWE",
            xlabel="Scan Number #",
            ylabel="x (px)",
        )

        simulation_space.viewer.plot_single_panel(
            np.fft.fftshift(data_noisy).T,
            title="Far-field Intensities PWE (Noisy)",
            xlabel="z (m)",
            ylabel="x (m)",
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
                range(simulation_space.effective_shape[0]),
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
