import math

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
from pathlib import Path

class Visualisation:
    """
    A class for visualizing the results of the paraxial solver.
    """

    def __init__(self, sample_space):
        self.bc_type = sample_space.bc_type
        self.sample_space = sample_space

        self.num_probes = sample_space.num_probes
        self.x = sample_space.x
        self.z = sample_space.z
        if sample_space.dimension == 2:
            self.y = sample_space.y
        else:
            self.y = None
        self.sub_x = []

    def plot(self, solution, reverse=False, initial=False, slider=False,
             plot_phase=True, title=None,
             probe_index=None):
        """
        Plot the real and imaginary parts of the solution at the initial or final time step.

        Args:
            self: The instance of the class.
            solution (ndarray): The solution array.
            reverse (bool): Whether to plot the reverse solution.
            initial (bool): Whether to plot the initial time step.
        """
        if probe_index is None:
                # Select the middle probe index for 1D or 2D scans as Default
                probe_index = int(self.sample_space.num_probes * 0.5)
    
        if slider:
            if len(solution.shape) == 4:
                solution = solution[probe_index, :, :, :]
            self.plot_slider(solution, plot_phase, reverse, title, probe_index)
        elif self.sample_space.dimension == 1:
            if self.num_probes == 1:
                if len(solution.shape) == 3:
                    solution = solution[0, :, :]
            else:
                if len(solution.shape) == 3:
                    solution = solution[probe_index, :, :]
            self.plot_solution(solution, reverse, initial, plot_phase,
                               probe_index, title)
        elif self.num_probes > 1:
            self.plot_scan_2d(solution, initial, plot_phase)
        elif self.num_probes == 1:
            if len(solution.shape) == 4:
                solution = solution[0, ...]

            self.plot_solution(solution, reverse, initial, plot_phase,
                               probe_index, title)
        else:
            raise ValueError("Options are invalid. ")

    def plot_solution(
            self,
            solution,
            reverse=False,
            initial=False,
            plot_phase=False,
            probe_index=None,
            title=None):
        """
        Plot the solution (single scan) at the initial or final time step.
        """
        
        if title is None:
            title = 'solution:'
            if self.sample_space.dimension == 2:
                if initial:
                    title += ' initial time step'
                else:
                    title += ' final time step'
                if reverse:
                    title += ' (reverse)'
                else:
                    title += ' (forward)'
            else:
                title = f' (probe {probe_index}, '
                if self.bc_type == "dirichlet":
                    title += 'dirichlet, '
                elif self.bc_type == "neumann":
                    title += 'neumann, '
                else:
                    title += 'impedance, '
                title += ' reverse)' if reverse else ' forward)'

        if self.sample_space.dimension == 2:
            dim1_label = 'X'
            dim2_label = 'Y'
            if initial:
                idx = 0
            else:
                idx = -1
            plot_solution = solution[:, :, idx]
        else:
            dim1_label = 'Z'
            dim2_label = 'X'
            idx = 0
            plot_solution = solution

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        if plot_phase:
            tol = 1e-8
            real = plot_solution.real
            imag = plot_solution.imag
            mask = (np.abs(real) < tol) & (np.abs(imag) < tol)
            p1 = self.compute_phase(plot_solution)
            p1[mask] = 0.0
            p2 = np.abs(plot_solution)
            title_p1 = 'Phase ' + title
            title_p2 = 'Amplitude ' + title
        else:
            p1 = plot_solution.real
            p2 = plot_solution.imag
            title_p1 = 'Real ' + title
            title_p2 = 'Imaginary ' + title

        c1 = axs[0].imshow(
            p1,
            cmap='viridis',
            origin='lower')
        fig.colorbar(c1, ax=axs[0])
        axs[0].set_title(title_p1)
        axs[0].set_xlabel(dim1_label)
        axs[0].set_ylabel(dim2_label)

        c2 = axs[1].imshow(
            p2,
            cmap='viridis',
            origin='lower')
        fig.colorbar(c2, ax=axs[1])
        axs[1].set_title(title_p2)
        axs[1].set_xlabel(dim1_label)
        axs[1].set_ylabel(dim2_label)

        plt.tight_layout()
        plt.show()


    def plot_slider(self, solution, plot_phase, reverse=False, title=None,
                    probe_index=0):
        """
        Plot the solution using an interactive slider (Jupyter Notebook only).
        """
        if title is None:
            title = f' (probe {probe_index}, '
            if self.bc_type == "dirichlet":
                title += 'dirichlet, '
            elif self.bc_type == "neumann":
                title += 'neumann, '
            else:
                title += 'impedance, '
            title += ' reverse)' if reverse else ' forward)'

        len_z = self.sample_space.nz

        if plot_phase:
            data1 = self.compute_phase(solution)
            data2 = np.abs(solution)
            title1 = 'Phase' + title
            title2 = 'Amplitude' + title
        else:
            data1 = solution.real
            data2 = solution.imag
            title1 = 'Real' + title
            title2 = 'Imaginary' + title

        max_val1 = np.max(data1)
        min_val1 = np.min(data1)
        max_val2 = np.max(data2)
        min_val2 = np.min(data2)

        def update(frame):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            z_frame = self.z[-(frame + 1)] if reverse else self.z[frame]

            im0 = axs[0].imshow(data1[:, :, frame], cmap='viridis', origin='lower', extent=[
                        self.x.min(), self.x.max(), self.y.min(), self.y.max()], vmin=min_val1, vmax=max_val1)
            axs[0].set_title(f'{title1} at Z = {z_frame:.2f}')
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(data2[:, :, frame], cmap='viridis', origin='lower', extent=[
                        self.x.min(), self.x.max(), self.y.min(), self.y.max()], vmin=min_val2, vmax=max_val2)
            axs[1].set_title(f'{title2} at Z = {z_frame:.2f}')
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Y')
            fig.colorbar(im1, ax=axs[1])

            plt.tight_layout()
            plt.show()

        interact(update, frame=IntSlider(min=0, max=len_z - 1, step=1, value=0, description="slice"))

    # def plot_scan_1d(self, solution, reverse):
    #     """
    #     Plot the solution for each scan."
    #     """
    #     fig_amp, axes_amp = plt.subplots(
    #         1, self.num_probes, figsize=(
    #             12, 6), squeeze=False)
    #     fig_phase, axes_phase = plt.subplots(
    #         1, self.num_probes, figsize=(
    #             12, 6), squeeze=False)

    #     amplitude = np.abs(solution)
    #     phase = self.compute_phase(solution)

    #     max_val_amp = np.max(amplitude)
    #     min_val_amp = np.min(amplitude)

    #     for idx in range(self.num_probes):
    #         if len(self.sample_space.sub_x_lims) > 0:
    #             x_lims = self.sample_space.sub_x_lims[idx]
    #         else:
    #             x_lims = self.sample_space.xlims
    
    #         im_phase = axes_phase[0,
    #                               idx].imshow(phase[idx,
    #                                                 :,
    #                                                 :],
    #                                           cmap='viridis',
    #                                           origin='lower',
    #                                           extent=[*x_lims,
    #                                                   self.z.min(),
    #                                                   self.z.max()])
    #         axes_phase[0, idx].set_title(f'Scan {idx + 1}')
    #         if idx == 0:
    #             axes_phase[0, idx].set_yticks(
    #                 np.linspace(self.z.min(), self.z.max(), 5))
    #             axes_phase[0, idx].set_xlabel('X')
    #             axes_phase[0, idx].set_ylabel('Z')
    #         else:
    #             axes_phase[0, idx].set_yticks([])
    #             axes_phase[0, idx].set_xlabel('X')

    #     if reverse:
    #         fig_amp.suptitle('Backward Solution Real')  # Amplitude')
    #         fig_phase.suptitle('Backward Solution Imaginary')  # Phase')
    #     else:
    #         fig_amp.suptitle('Forward Solution Real')  # Amplitude')
    #         fig_phase.suptitle('Forward Solution Phase')  # Imaginary')

    #     # fig_amp.colorbar(im_amp, ax=axes_amp)
    #     fig_phase.colorbar(im_phase, ax=axes_phase)
    #     plt.show()

    def plot_scan_2d(self, solution, initial, plot_phase):
        """
        Plot the solution for each scan.
        If plot_phase is True, plot phase and amplitude instead of real and imaginary.
        """
        fig1, axes1 = plt.subplots(
            self.sample_space.scan_points, self.sample_space.scan_points, figsize=(15, 12), squeeze=False)
        fig2, axes2 = plt.subplots(
            self.sample_space.scan_points, self.sample_space.scan_points, figsize=(15, 12), squeeze=False)

        if initial:
            exit_or_probe = 0
        else:
            exit_or_probe = -1

        if plot_phase:
            data1 = self.compute_phase(solution[:, :, :, exit_or_probe])
            data2 = np.abs(solution[:, :, :, exit_or_probe])
            title1 = 'Phase'
            title2 = 'Amplitude'
        else:
            data1 = solution[:, :, :, exit_or_probe].real
            data2 = solution[:, :, :, exit_or_probe].imag
            title1 = 'Real'
            title2 = 'Imaginary'

        max_val1 = np.max(data1)
        min_val1 = np.min(data1)
        max_val2 = np.max(data2)
        min_val2 = np.min(data2)

        for idx in range(self.num_probes):
            row = idx // self.sample_space.scan_points
            if row % 2 == 0:  # Even row (left to right)
                col = idx % self.sample_space.scan_points
            else:  # Odd row (right to left)
                col = self.sample_space.scan_points - 1 - (idx % self.sample_space.scan_points)

            im1 = axes1[row, col].imshow(data1[idx, :, :], cmap='viridis', vmin=min_val1, vmax=max_val1, origin='lower')
            axes1[row, col].set_title(f'Scan {idx}')

            im2 = axes2[row, col].imshow(data2[idx, :, :], cmap='viridis', vmin=min_val2, vmax=max_val2, origin='lower')
            axes2[row, col].set_title(f'Scan {idx}')

        if initial:
            fig1.suptitle(f'Probe Waves {title1}')
            fig2.suptitle(f'Probe Waves {title2}')
        else:
            fig1.suptitle(f'Exit Waves {title1}')
            fig2.suptitle(f'Exit Waves {title2}')

        fig1.colorbar(im1, ax=axes1)
        fig2.colorbar(im2, ax=axes2)
        plt.show()

    def compute_phase(self, solution):
        """
        Compute the phase of the solution account for small values.
        This is used to avoid issues with np.angle() when the real and 
        imaginary parts are close to zero.
        """
        tol = 1e-8
        real = solution.real
        imag = solution.imag
        mask = (np.abs(real) < tol) & (np.abs(imag) < tol)
        p1 = np.angle(solution)
        p1[mask] = 0.0
        return p1
