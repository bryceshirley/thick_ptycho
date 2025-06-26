from typing import Dict, List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt

from .optical_objects import OpticalObject

class SampleSpace:
    """
    Interface class to create either a 1D or 2D sample space based on the detector shape.
    """
    def __new__(cls, continuous_dimensions, *args, **kwargs):
        dimension = len(continuous_dimensions) - 1
        if dimension == 1:
            return SampleSpace1D(continuous_dimensions, *args, **kwargs)
        elif dimension == 2:
            return SampleSpace2D(continuous_dimensions, *args, **kwargs)
        else:
            raise ValueError("Unsupported dimension: {}".format(dimension))
        

class SampleSpace1D:
    """
    Class representing a 1D sample space for paraxial solver.
    """

    def __init__(
            self,
            continuous_dimensions,
            discrete_dimensions,
            probe_dimensions,
            scan_points,
            step_size,
            bc_type,
            probe_type,
            wave_number,
            probe_diameter=None):
        """
        Initialize the 1D sample space.

        Parameters:
        continuous_dimensions (list): Sample space limits in continuous units (x, z).
        discrete_dimensions (list): Sample space dimensions in pixels (nx, nz).
        detector_dimensions (list): Detector shape in pixels.
        scan_points (int): Number of scan points.
        bc_type (str): Boundary condition type (e.g., 'dirichlet', 'neumann').
        wave_number (float): Wavenumber in 1/nm.
        probe_diameter (float, optional): Probe diameter in continuous units. Default is 12.
        """
        # Set the dimension to 1D
        self.dimension = 1

        # Probe shape (pixels)
        self.probe_dimensions = probe_dimensions

        # Step size for scanning (pixels)
        self.step_size = step_size

        # Number of scan points along one axis
        self.scan_points = scan_points

        # Probe Type
        self.probe_type = probe_type

        # Discrete sample space dimensions (pixels)
        self.nx = discrete_dimensions[0]
        self.nz = discrete_dimensions[1]

        # Boundary condition type (lowercase)
        self.bc_type = bc_type.lower()

        # Continuous sample space limits (x, z)
        self.xlims, self.zlims = continuous_dimensions

        # Create coordinate arrays and set sub-region sizes depending on BCs
        if self.bc_type == "dirichlet":
            self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx + 2)
        else:
            self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)

        self.sub_nx = self.probe_dimensions[0]

        # z-axis coordinates and step size
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dz = self.z[1] - self.z[0]

        # x step sizes
        self.dx = self.x[1] - self.x[0]

        # Probe diameter in continuous units and discrete pixels
        if probe_diameter is None:
            self.probe_diameter = probe_dimensions[0]
        else:
            self.probe_diameter = probe_diameter
        # Wavenumber
        self.k = wave_number

        # List to store optical objects
        self.objects = []

        # Initialize the refractive index field
        self.n_true = np.ones((self.nx, self.nz), dtype=complex)

        # Total number of probes (scan_points squared)
        self.num_probes = scan_points

        self._detector_frame_info = self._generate_scan_frames()

    
    @property
    def detector_frame_info(self) -> List[Dict[str, Any]]:
        """
        List[Dict[str, Any]]: List of dictionaries with detector frame data.
        Each dictionary in the list corresponds to a scan.
        - 'probe_centre_continuous': probe_centre_continuous,
        - 'probe_centre_discrete': probe_centre_discrete,
        - 'sub_dimensions': (sub_x, sub_y)
        """
        return self._detector_frame_info

    def summarize_sample_space(self):
        """ Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = (
            self.xlims[1] - self.xlims[0]) * (self.step_size / self.nx)
        overlap = self.probe_diameter * self.dx - continuous_stepsize
        print("Summary of the scan (continuous):")
        print(f"    Sample space x: {self.xlims[1] - self.xlims[0]} um")
        print(f"    Sample space z: {self.zlims[1] - self.zlims[0]} um")
        print(f"    Probe Diameter: {self.probe_diameter*self.dx:.2f} um")
        print(f"    Number of scan points: {self.scan_points}")
        if self.scan_points > 1:
            print(f"    Max Overlap: {overlap:.2f} um \n")

    
    def _generate_scan_frames(self) -> List[Dict[str, Any]]:
        """
        Generate detector frames along a serpentine scan path.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with detector frame data.
            Each dictionary in the list corresponds to a scan.
            'probe_centre_continuous': probe_centre_continuous,
            'probe_centre_discrete': probe_centre_discrete,
            'sub_sample_slices': sample_slices,
            'sub_dimensions': (sub_x, sub_y)
        """
        # Total distance between probe centres
        total_step = (self.scan_points - 1) * self.step_size


        # Distance from the edge to the first probe centre
        edge_margin = (self.nx - total_step) // 2

        # Validate parameters
        assert self.nx > total_step, "nx must be greater than the total scan steps. Reduce scan points or steps size, or increase nx."
        assert edge_margin >= self.probe_dimensions[
            0] // 2, "Probe shape is too large for the dimensions of the sample space. Reduce probe shape or increase nx."

        # Generate x for the scan path
        centre_x = np.floor(np.linspace(
            edge_margin, self.nx + 2 - edge_margin, self.scan_points)).astype(int)

        # --- Construct detector frames ---
        frames = []

        for k in range(self.num_probes):
            # Centre of each probe
            cx = centre_x[k]

            if self.bc_type == "dirichlet" and self.num_probes == 1:
                cx += 1
            probe_centre_discrete = (cx)

            # Continuous Centre of each probe
            probe_centre_continuous = (self.x[cx])

            # Boundaries of each probe
            x_min = int(cx - self.probe_dimensions[0] / 2)
            x_max = int(cx + self.probe_dimensions[0] / 2)

            # Continuous Boundaries of each probe
            if self.bc_type == "dirichlet":
                x_min -= 1
                x_max += 1

            sub_x = self.x[x_min:x_max]

            frames.append({
                'probe_centre_continuous': probe_centre_continuous,
                'probe_centre_discrete': probe_centre_discrete,
                'sub_dimensions': (sub_x,),
                'sub_limits': x_min
            })

        self.centre_x = centre_x

        return frames

    def add_object(self, shape, refractive_index, side_length, centre, depth,
                   guassian_blur=None):
        """
        Add an optical object to the simulation.

        Parameters:
        shape (str): Shape of the object.
        refractive_index (float): Refractive index of the object.
        side_length (float): Side length of the object.
        centre (tuple): Centre of the object.
        depth (float): Depth of the object.
        """
        if self.bc_type == "dirichlet":
            x = self.x[1:-1]
        else:
            x = self.x
        self.objects.append(
            OpticalObject(
                centre,
                shape,
                refractive_index,
                side_length,
                depth,
                self.nx,
                self.nz,
                x,
                self.z,
                guassian_blur))

    def generate_sample_space(self):
        """
        Generate the field of objects in free space.
        """
        for obj in self.objects:
            self.n_true += obj.get_refractive_index_field()

    def create_sample_slices(self, thin_sample: bool, n=None, grad=False,
                             scan_index=0):
        """
        Create the field of objects in free space.

        Parameters:
        n (ndarray): Sample refractive index field.

        Returns:
        ndarray: Object slices.
        """
        # If n is not provided, use the true refractive index field
        if n is None:
            n = self.n_true

        # If Sample is thin, restrict the domain
        if thin_sample:  # For thin samples/ sub-sampling
            # Boundaries of each probe
            x_min = self.detector_frame_info[scan_index]['sub_limits']

            # Extract the sub-sample from the object field
            n = n[x_min:x_min+self.probe_dimensions[0], :]

        # Compute the coefficient based on whether we want the gradient or not
        if grad:
            coefficient = (self.k / 1j) * n
        else:
            coefficient = ((self.k / 2j) * (n**2 - 1))

        # Compute all half time_step slices at once using vectorized operations
        # Shape: (nx, nz-1)
        object_slices = (
            self.dz / 2) * (coefficient[:, :-1] + coefficient[:, 1:]) / 2

        return object_slices


class SampleSpace2D:
    """
    Class representing a 2D sample space for paraxial solver.
    """

    def __init__(
            self,
            continuous_dimensions,
            discrete_dimensions,
            probe_dimensions,
            scan_points,
            step_size,
            bc_type,
            probe_type,
            wave_number,
            probe_diameter=None):
        """
        Initialize the 2D sample space.

        Parameters:
        continuous_dimensions (list): sample space dimensions in nanometers (x, y, z)
        discrete_dimensions (list): sample space dimensions in pixels (nx, ny, nz)
        scan_points (int): number of ptychography scan points
        bc_type (str): boundary condition type (impedance, dirichlet, neumann)
        wave_number (float): wavenumber in 1/nm
        """
        # Set the dimension to 2D
        self.dimension = 2

        # Probe shape (pixels)
        self.probe_dimensions = probe_dimensions

        # Number of scan points along one axis
        self.scan_points = scan_points

        # Discrete sample space dimensions (pixels)
        self.nx = discrete_dimensions[0]
        self.ny = discrete_dimensions[1]
        self.nz = discrete_dimensions[2]

        # Step size for scanning (pixels)
        self.step_size = step_size

        # Boundary condition type (lowercase)
        self.bc_type = bc_type.lower()

        # Probe Type
        self.probe_type = probe_type

        # Continuous sample space limits (x, y, z)
        self.xlims, self.ylims, self.zlims = continuous_dimensions

        # Create coordinate arrays and set sub-region sizes depending on BCs
        if self.bc_type == "dirichlet":
            self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx + 2)
            self.y = np.linspace(self.ylims[0], self.ylims[1], self.ny + 2)
        else:
            self.x = np.linspace(self.xlims[0], self.xlims[1], self.nx)
            self.y = np.linspace(self.ylims[0], self.ylims[1], self.ny)

        self.sub_nx = self.probe_dimensions[0]
        self.sub_ny = self.probe_dimensions[1]

        # z-axis coordinates and step size
        self.z = np.linspace(self.zlims[0], self.zlims[1], self.nz)
        self.dz = self.z[1] - self.z[0]

        # x and y step sizes
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Probe diameter in continuous units and discrete pixels
        if probe_diameter is None:
            self.probe_diameter = probe_dimensions[0]
        else:
            self.probe_diameter = probe_diameter
        # Wavenumber
        self.k = wave_number

        # List to store optical objects
        self.objects = []

        # Initialize the refractive index field
        self.n_true = np.ones((self.nx, self.ny, self.nz), dtype=complex)

        # Total number of probes (scan_points squared)
        self.num_probes = scan_points**2

        self._detector_frame_info = self._generate_scan_frames()
    
    def summarize_sample_space(self):
        """ Print a summary of the sample space and scan parameters.
        """
        # Print summary of the scan
        continuous_stepsize = (
            self.xlims[1] - self.xlims[0]) * (self.step_size / self.nx)
        overlap = self.probe_diameter * self.dx - continuous_stepsize
        print("Summary of the scan (continuous):")
        print(f"    Sample space x: {self.xlims[1] - self.xlims[0]} um")
        print(f"    Sample space y: {self.ylims[1] - self.ylims[0]} um")
        print(f"    Sample space z: {self.zlims[1] - self.zlims[0]} um")
        print(f"    Probe Diameter: {self.probe_diameter*self.dx:.2f} um")
        print(f"    Number of scan points: {self.num_probes}")
        if self.scan_points > 1:
            print(f"    Max Overlap: {overlap:.2f} um \n")

        # Plot the scan path with flipped axes
        plt.figure(figsize=(6, 6))
        plt.plot(self.centre_y, self.centre_x, marker='o', linestyle='-')
        plt.title("2D Discrete Scan Path")
        plt.xlabel("Ny")
        plt.ylabel("Nx")
        plt.xlim((0, self.ny))
        plt.ylim((0, self.nx))
        # Draw a box around the first probe area with faint fill and no outline
        y_min = int(self.centre_y[0] - self.probe_dimensions[1] / 2)
        y_max = int(self.centre_y[0] + self.probe_dimensions[1] / 2)
        x_min = int(self.centre_x[0] - self.probe_dimensions[0] / 2)
        x_max = int(self.centre_x[0] + self.probe_dimensions[0] / 2)
        rect1 = plt.Rectangle(
            (y_min, x_min), y_max - y_min, x_max - x_min,
            linewidth=0, edgecolor='none', facecolor='red', alpha=0.2, label='First Probe Area'
        )
        plt.gca().add_patch(rect1)
        if self.probe_type == "disk":
            circ1 = plt.Circle(
                (self.centre_y[0], self.centre_x[0]),
                radius=self.probe_diameter / 2,
                color='red', fill=False, alpha=0.2, label='First Probe'
            )
            plt.gca().add_patch(circ1)

        if self.scan_points > 1:
            # Draw a box around the second probe area with faint fill and no outline
            y_min = int(self.centre_y[1] - self.probe_dimensions[1] / 2)
            y_max = int(self.centre_y[1] + self.probe_dimensions[1] / 2)
            x_min = int(self.centre_x[1] - self.probe_dimensions[0] / 2)
            x_max = int(self.centre_x[1] + self.probe_dimensions[0] / 2)
            rect2 = plt.Rectangle(
                (y_min, x_min), y_max - y_min, x_max - x_min,
                linewidth=0, edgecolor='none', facecolor='green', alpha=0.2, label='Second Probe Area'
            )
            plt.gca().add_patch(rect2)
            if self.probe_type == "disk":
                circ2 = plt.Circle(
                    (self.centre_y[1], self.centre_x[1]),
                    radius=self.probe_diameter / 2,
                    color='green', fill=False, alpha=0.2, label='Second Probe')
                plt.gca().add_patch(circ2)
        plt.legend()
        plt.grid()
        plt.show()

        

    @property
    def detector_frame_info(self) -> List[Dict[str, Any]]:
        """
        List[Dict[str, Any]]: List of dictionaries with detector frame data.
        Each dictionary in the list corresponds to a scan.
        - 'probe_centre_continuous': probe_centre_continuous,
        - 'probe_centre_discrete': probe_centre_discrete,
        - 'sub_dimensions': (sub_x, sub_y)
        """
        return self._detector_frame_info

    def _generate_scan_frames(self) -> List[Dict[str, Any]]:
        """
        Generate detector frames along a serpentine scan path.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with detector frame data.
            Each dictionary in the list corresponds to a scan.
            'probe_centre_continuous': probe_centre_continuous,
            'probe_centre_discrete': probe_centre_discrete,
            'sub_sample_slices': sample_slices,
            'sub_dimensions': (sub_x, sub_y)
        """
        # Total distance between probe centres
        total_step = (self.scan_points - 1) * self.step_size

        # Distance from the edge to the first probe centre
        edge_margin = (self.nx - total_step) // 2

        # Validate parameters
        assert self.nx > total_step, "nx must be greater than the total scan steps. Reduce scan points or steps size, or increase nx."
        assert edge_margin >= self.probe_dimensions[
            0] // 2, "Probe shape is too large for the dimensions of the sample space. Reduce probe shape or increase nx."
        assert edge_margin >= self.probe_dimensions[
            1] // 2, "Probe shape is too large for the dimensions of the sample space. Reduce probe shape or increase ny."

        # Generate x and y positions for the scan path
        x_positions = np.floor(np.linspace(
            edge_margin, self.nx + 2 - edge_margin, self.scan_points)).astype(int)
        y_positions = np.floor(np.linspace(
            edge_margin, self.ny + 2 - edge_margin, self.scan_points)).astype(int)

        centre_x = []
        centre_y = []
        x_positions = np.flip(x_positions)
        for x in x_positions:
            for y in y_positions:
                centre_x.append(x)
                centre_y.append(y)
            y_positions = np.flip(y_positions)

        # --- Construct detector frames ---
        frames = []

        for k in range(self.num_probes):
            # Centre of each probe
            cx = centre_x[k]
            cy = centre_y[k]
            if self.bc_type == "dirichlet" and self.num_probes == 1:
                cx += 1
                cy += 1
            probe_centre_discrete = (cx, cy)

            # Continuous Centre of each probe
            probe_centre_continuous = (self.x[cx], self.y[cy])

            # Boundaries of each probe
            x_min = int(cx - self.probe_dimensions[0] / 2)
            x_max = int(cx + self.probe_dimensions[0] / 2)
            y_min = int(cy - self.probe_dimensions[1] / 2)
            y_max = int(cy + self.probe_dimensions[1] / 2)

            # Continuous Boundaries of each probe
            if self.bc_type == "dirichlet":
                x_min -= 1
                x_max += 1
                y_min -= 1
                y_max += 1
            sub_x = self.x[x_min:x_max]
            sub_y = self.y[y_min:y_max]

            frames.append({
                'probe_centre_continuous': probe_centre_continuous,
                'probe_centre_discrete': probe_centre_discrete,
                'sub_dimensions': (sub_x, sub_y),
                'sub_limits': (x_min, y_min)
            })

        self.centre_x = centre_x
        self.centre_y = centre_y

        return frames

    def add_object(self, shape, refractive_index, side_length, centre, depth,
                   guassian_blur=False):
        """
        Add an optical object to the simulation.

        Parameters:
        shape (str): Shape of the object.
        refractive_index (float): Refractive index of the object.
        side_length (float): Side length of the object.
        centre (tuple): Centre of the object.
        depth (float): Depth of the object.
        """
        if self.bc_type == "dirichlet":
            x = self.x[1:-1]  # Exclude the boundary points
            y = self.y[1:-1]  # Exclude the boundary points
        else:
            x = self.x
            y = self.y
        assert len(
            centre) == 3, "Centre must be a tuple of (x, y, z) coordinates."
        assert self.xlims[0] <= centre[0] <= self.xlims[1], "Centre x-coordinate out of bounds."
        assert self.ylims[0] <= centre[1] <= self.ylims[1], "Centre y-coordinate out of bounds."
        assert self.zlims[0] <= centre[2] <= self.zlims[1], "Centre z-coordinate out of bounds."
        self.objects.append(
            OpticalObject(
                centre,
                shape,
                refractive_index,
                side_length,
                depth,
                self.nx,
                self.ny,
                self.nz,
                x,
                y,
                self.z,
                guassian_blur))

    def generate_sample_space(self):
        """
        Generate the field of objects in free space.
        """
        for obj in self.objects:
            self.n_true += obj.get_refractive_index_field()

    def create_sample_slices(self, thin_sample: bool, n=None, grad=False,
                             scan_index=0):
        """
        Create the field of objects in free space.

        Parameters:
        n (ndarray): Sample refractive index field.

        Returns:
        ndarray: Object slices.
        """
        # If n is not provided, use the true refractive index field
        if n is None:
            n = self.n_true

        # If Sample is thin, restrict the domain
        if thin_sample:  # For thin samples/ sub-sampling
            # Boundaries of each probe
            x_min, y_min = self.detector_frame_info[scan_index]['sub_limits']

            # Extract the sub-sample from the object field
            n = n[x_min:x_min+self.probe_dimensions[0],
                  y_min:y_min+self.probe_dimensions[1], :]
        
        # Compute the coefficient based on whether we want the gradient or not
        if grad:
            coefficient = (self.k / 1j) * n
        else:
            coefficient = ((self.k / 2j) * (n**2 - 1))

        # Compute all half time_step slices at once using vectorized operations
        # Shape: (nx, ny, nz-1)
        object_slices = (
            self.dz / 2) * (coefficient[:, :, :-1] + coefficient[:, :, 1:]) / 2

        return object_slices
