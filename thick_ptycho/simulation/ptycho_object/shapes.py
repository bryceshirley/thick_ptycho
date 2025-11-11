import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from PIL import Image, ImageDraw, ImageFilter

from scipy.ndimage import gaussian_filter

class OpticalShape:
    """
    Interface class to create either 1D or 2D initial conditions.
    """
    def __new__(cls, centre_scale, shape, refractive_index,
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space):
        dimension = len(centre_scale) - 1
        if dimension == 1:
            return OpticalShape1D(centre_scale, shape, refractive_index,
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space)
        elif dimension == 2:
            return OpticalShape2D(centre_scale, shape, refractive_index,
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space)
        else:
            raise ValueError("Unsupported dimension: {}".format(dimension))

class OpticalShapeBase:
    """Represents an optical shape in the simulation."""
    def __init__(self, centre_scale, shape, refractive_index,
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space):
        for c in centre_scale:
            assert c <1.0 and c >0.0, "Centre scales must be between 0 and 1"
        assert side_length_scale <1.0 and side_length_scale >0.0, "Side length scales must be between 0 and 1"
        assert depth_scale <1.0 and depth_scale >0.0, "Depth scales must be between 0 and 1"
        

        # Shape properties
        self.shape = shape
        self.refractive_index = refractive_index
        self.guassian_blur = guassian_blur

        # Simulation Space properties
        self.nx = simulation_space.nx
        self.nz = simulation_space.nz
        self.x = simulation_space.x
        self.z = simulation_space.z

        # Simulation Space Bounds
        x_start, x_end = simulation_space.spatial_limits.x
        z_start, z_end = simulation_space.spatial_limits.z
        range_x = abs(x_end - x_start)
        range_z = abs(z_end - z_start)

        # Convert from normalized (0–1) to absolute spatial units
        side_length_continuous = side_length_scale * range_x
        depth_continuous = depth_scale * range_z
        self.centre_continuous = []

        # Convert from normalized (0–1) to absolute discrete units
        self.discrete_side_length = int(side_length_scale*self.nx)
        self.discrete_depth = int(depth_scale*self.nz)
        
        # Convert to discrete units (pixel-based grid indexing)
        self.discrete_side_length = int(self.nx * side_length_scale)
        self.discrete_depth = int(self.nz * depth_scale)
        self.discrete_centre = []

        # Calculate continuous and discrete centre positions
        for i, con_dim in  enumerate(simulation_space.spatial_limits.as_tuple()):
            start, end = con_dim
            range_dim = abs(end - start)
            self.centre_continuous.append(start + centre_scale[i] * range_dim)

            self.discrete_centre.append(int(np.ceil(((self.centre_continuous[i] - start) / (end - start)) * simulation_space.shape[i])))

        # Bounds check of continuous object extents against the used grid extents
        self.half_w = 0.5 * side_length_continuous
        self.half_d = 0.5 * depth_continuous
        assert side_length_scale < 0.5, "Object side length scale too large"
        assert depth_scale < 0.5, "Object depth scale too large"
        assert (self.centre_continuous[-1] - self.half_d) >= self.z[0], "Object out of bounds"
        assert (self.centre_continuous[-1] + self.half_d) <= self.z[-1], "Object out of bounds"
        assert (self.centre_continuous[0] - self.half_w) >= self.x[0], "Object out of bounds"
        assert (self.centre_continuous[0] + self.half_w) <= self.x[-1], "Object out of bounds"

        # -----------------------------------------------
        # Ensure object fits in grid
        # -----------------------------------------------
        cx = self.discrete_centre[0]
        cz = self.discrete_centre[-1]
        self.half_w_pixels = self.discrete_side_length // 2
        self.half_d_pixels = self.discrete_depth // 2

        assert cx - self.half_w_pixels >= 0, "Object out of x-bounds"
        assert cx + self.half_w_pixels < self.nx, "Object out of x-bounds"
        assert cz - self.half_d_pixels >= 0, "Object out of z-bounds"
        assert cz + self.half_d_pixels < self.nz, "Object out of z-bounds"



    def get_refractive_index_field(self):
        """Return the field of the object."""
        pass

    # def _get_field(self):
    #     """Return the field of a object."""
    #     pass


class OpticalShape1D(OpticalShapeBase):
    """Represents an optical shape in the simulation in 1D."""

    def __init__(self, centre_scale, shape, refractive_index, 
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space):
        super().__init__(centre_scale, shape, refractive_index,
                         side_length_scale, depth_scale, guassian_blur,
                         simulation_space)
        self.discrete_centre = [
            int(round(centre_scale[0] * (self.nx - 1))),
            int(round(centre_scale[1] * (self.nz - 1)))
        ]
        self.cx, self.cz = self.discrete_centre

    def get_refractive_index_field(self):
        """Return the field of the object."""
        if self.shape == 'rectangle':
            polygon_points = self._square_points()
        elif self.shape == 'triangle':
            polygon_points = self._triangle_points()
        elif self.shape == 'random':
            polygon_points = self._random_shape_points(num_points=20)
        elif self.shape == 'circle':
            polygon_points = None
        else:
            raise ValueError("Unsupported shape: {}".format(self.shape))

        return self._get_field(polygon_points)

    def _get_field(self, polygon_points=None):
        """Return the field of a object."""
        # Create a black image
        image_size = (self.nx, self.nz)
        image = Image.new(
            'RGB', image_size, color=(
                0, 0, 0))  # Black background
        draw = ImageDraw.Draw(image)

        # Draw the shape based on the polygon_points or as a circle
        if polygon_points is None:
            # polygon_points should be center and radius
            center = (self.cx, self.cz)
            radius1 = self.discrete_side_length / 2
            radius2 = self.discrete_depth / 2
            bbox = [
                center[0] - radius1, center[1] - radius2,
                center[0] + radius1, center[1] + radius2
            ]
            draw.ellipse(bbox, outline='white', fill='white')
        else:
            # Draw a white polygon using the calculated corners
            draw.polygon(
            polygon_points,
            outline='white',
            fill='white')  # White polygon

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert to a binary matrix: 1 where the image is white, 0 where it is
        # black
        binary_matrix = np.all(
            image_array == [
                255,
                255,
                255],
            axis=-
            1).astype(int).T

        # Create Refractive index field with optional Gaussian blur

        # No blur: sharp boundaries
        n = np.zeros((self.nx, self.nz), dtype=complex)
        n[np.where(binary_matrix == 1)] = self.refractive_index

        # Create Refractive index field with optional Gaussian blur
        if self.guassian_blur is not None:
            # Apply Gaussian blur to the refractive index field
            n = gaussian_filter(n, sigma=self.guassian_blur)
        return n

    def _square_points(self):
        half_side_length = self.discrete_side_length / 2
        half_depth = self.discrete_depth / 2

        # Define the corners of the square
        square_points = [
            (self.cx - half_side_length, self.cz - half_depth),  # Top-left
            (self.cx + half_side_length, self.cz - half_depth),  # Top-right
            (self.cx + half_side_length, self.cz + half_depth),  # Bottom-right
            (self.cx - half_side_length, self.cz + half_depth)   # Bottom-left
        ]

        return square_points

    def _triangle_points(self):
        half_side_length = self.discrete_side_length / 2
        half_depth = self.discrete_depth / 2

        angles_rad = np.radians([0, 120, 240])  # [90, 210, 330])

        # Calculate the three vertices based on the center and
        # side_lengthlength
        x1 = self.cx + half_side_length * math.cos(angles_rad[0])
        z1 = self.cz + half_depth * math.sin(angles_rad[0])

        x2 = self.cx + half_side_length * math.cos(angles_rad[1])
        z2 = self.cz + half_depth * math.sin(angles_rad[1])

        x3 = self.cx + half_side_length * math.cos(angles_rad[2])
        z3 = self.cz + half_depth * math.sin(angles_rad[2])

        # Define the triangle points
        triangle_points = [(x1, z1), (x2, z2), (x3, z3)]

        return triangle_points

    def _random_shape_points(
            self,
            num_points=12,
            irregularity=0.9,
            spikeyness=0.9):
        """
        Generate a list of points for a smooth random polygon centered at (self.cx, self.cz).
        num_points: Number of vertices.
        irregularity: [0,1] variance in angular spacing.
        spikeyness: [0,1] variance in radius.
        """
        center_x, center_z = self.cx, self.cz
        avg_radius = self.discrete_side_length / 2
        angle_steps = []
        lower = (2 * np.pi / num_points) * (1 - irregularity)
        upper = (2 * np.pi / num_points) * (1 + irregularity)
        sum_angles = 0
        for _ in range(num_points):
            tmp = np.random.uniform(lower, upper)
            angle_steps.append(tmp)
            sum_angles += tmp
        k = sum_angles / (2 * np.pi)
        angle_steps = [step / k for step in angle_steps]

        points = []
        angle = np.random.uniform(0, 2 * np.pi)
        for i in range(num_points):
            r = avg_radius * (1 + np.random.uniform(-spikeyness, spikeyness))
            x = center_x + r * np.cos(angle)
            z = center_z + r * np.sin(angle)
            points.append((x, z))
            angle += angle_steps[i]
        return points


class OpticalShape2D(OpticalShapeBase):
    """Represents an optical shape in the simulation in 2D."""

    def __init__(self, centre_scale, shape, refractive_index,
                 side_length_scale, depth_scale, guassian_blur,
                 simulation_space):
        super().__init__(centre_scale, shape, refractive_index,
                         side_length_scale, depth_scale, guassian_blur,
                         simulation_space)
        self.ny = simulation_space.ny
        self.y = simulation_space.y
        cy = self.discrete_centre[1]
        
        # Bounds check of continuous object extents against the used grid extents
        assert (self.centre_continuous[1] - self.half_w) >= self.y[0], "Object out of bounds"
        assert (self.centre_continuous[1] + self.half_w) <= self.y[-1], "Object out of bounds"
        assert cy - self.half_w_pixels >= 0, "Object out of y-bounds"
        assert cy + self.half_w_pixels < self.ny, "Object out of y-bounds"
        self.cx, self.cy, self.cz = self.discrete_centre

    def get_refractive_index_field(self):
        """Return the field of the object."""
        if self.shape == 'cuboid':
            polygon_points = self._square_points()
        elif self.shape == 'prism':
            polygon_points = self._triangle_points()
        elif self.shape == 'cylinder':
            polygon_points = None
        else:
            raise ValueError("Unsupported shape: {}".format(self.shape))

        return self._get_field(polygon_points)

    def _get_field(self, polygon_points):
        """Return the field of a object."""
        half_depth = int(self.discrete_depth / 2)

        # Create a black image
        image_size = (self.nx, self.ny)
        image = Image.new(
            'RGB', image_size, color=(
                0, 0, 0))  # Black background
        draw = ImageDraw.Draw(image)

        # Draw a white polygon using the calculated corners
        # Draw the shape based on the polygon_points or as a circle
        if polygon_points is None:
            # polygon_points should be center and radius
            center = (self.cx, self.cy)
            radius = self.discrete_side_length / 2
            bbox = [
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ]
            draw.ellipse(bbox, outline='white', fill='white')
        else:
            # Draw a white polygon using the calculated corners
            draw.polygon(
            polygon_points,
            outline='white',
            fill='white')  # White polygon

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert to a binary matrix: 1 where the image is white, 0 where it is
        # black
        binary_matrix = np.all(
            image_array == [
                255,
                255,
                255],
            axis=-
            1).astype(int)

        # Reshape into a 1D array
        binary_matrix = binary_matrix.reshape(self.nx * self.ny)

        # Repeat the shape through slices to create a mask for the 3D object
        mask = np.zeros((self.nx * self.ny, self.nz), dtype=bool)
        obj = np.tile(binary_matrix[:, np.newaxis], (1, self.discrete_depth))
        obj = obj.reshape(self.nx * self.ny, self.discrete_depth)
        half_depth = self.discrete_depth // 2
        start = self.cz - half_depth
        end = self.cz + half_depth + 1  # include center slice
        mask[:, start:end] = obj
        mask = mask.reshape((self.nx, self.ny, self.nz))

        # Create Refractive index field
        n = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        n[np.where(mask == 1)] = self.refractive_index

        # Create Refractive index field with optional Gaussian blur
        if self.guassian_blur is not None:
            # Apply Gaussian blur to the refractive index field
            n = gaussian_filter(n, sigma=self.guassian_blur)

        return n

    def _square_points(self):
        half_side_length = self.discrete_side_length / 2
        square_points = [
            (self.cx - half_side_length, self.cy - half_side_length),  # Top-left
            (self.cx + half_side_length, self.cy - half_side_length),  # Top-right
            (self.cx + half_side_length, self.cy +
             half_side_length),  # Bottom-right
            (self.cx - half_side_length, self.cy + half_side_length)   # Bottom-left
        ]
        return square_points

    def _triangle_points(self):
        half_side_length = self.discrete_side_length / 2

        angles_rad = np.radians([90, 210, 330])

        # Calculate the three vertices based on the center and
        # side_lengthlength
        x1 = self.cx + half_side_length * math.cos(angles_rad[0])
        y1 = self.cy + half_side_length * math.sin(angles_rad[0])

        x2 = self.cx + half_side_length * math.cos(angles_rad[1])
        y2 = self.cy + half_side_length * math.sin(angles_rad[1])

        x3 = self.cx + half_side_length * math.cos(angles_rad[2])
        y3 = self.cy + half_side_length * math.sin(angles_rad[2])

        # Define the triangle points
        triangle_points = [(x1, y1), (x2, y2), (x3, y3)]

        return triangle_points
