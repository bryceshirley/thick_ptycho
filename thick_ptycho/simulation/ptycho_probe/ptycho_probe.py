from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import jv


def u0_nm_neumann(n, m):
    """Return exact Neumann solution basis u(x,y) = cos(nπx)cos(mπy)."""
    return lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u0_nm_dirichlet(n, m):
    """Return exact Dirichlet solution basis u(x,y) = sin(nπx)sin(mπy)."""
    return lambda x, y: np.sin(n * np.pi * x) * np.sin(m * np.pi * y)


class PtychoProbes:
    """
    Probe field generator with reusable phase utilities.

    This class contains all probe profile definitions (constant, gaussian, disk,
    airy_disk, etc.) and the phase-operations used to modify them (linear tilt and
    quadratic focus). It is designed to be stateless w.r.t. solver internals and
    only depends on `simulation_space` and per-scan metadata to assemble fields.

    Parameters
    ----------
    simulation_space : object
        An object providing geometric and discretization info, e.g.
        - dimension : int (1 or 2)
        - nx[, ny] : int
        - x[, y] : ndarray (grid coordinates)
        - dx[, dy] : float (pixel size)
        - k : float (wavenumber)
        - probe_diameter : float
        - scan_frame_info[scan] : dict with
            * "probe_centre_continuous": float in 1D, (cx, cy) in 2D
            * "probe_centre_discrete": int in 1D, (cx, cy) in 2D
            * "reduced_limits_continuous": (x_min, x_max) in 1D or (x_min, x_max, y_min, y_max) in 2D
            * "reduced_limits_discrete": (x_min, x_max) in 1D or (x_min, x_max, y_min, y_max) in 2D
    solve_reduced_domain : bool
        If True, use sub-sampled grids from `scan_frame_info`.

    Attributes
    ----------
    disk_edge_blur : float
        Fraction of radius used for the cosine-ramp edge in the "blurred_disk"
        probe (default 0.5).
    """

    def __init__(self, simulation_space):
        self.probe_tilts = list(simulation_space.probe_angles)
        self.probe_type = simulation_space.probe_type
        self.probe_focus = simulation_space.probe_focus
        self.probe_diameter = simulation_space.probe_diameter
        self.simulation_space = simulation_space

        if self.probe_diameter is not None:
            for i, lim in enumerate(
                [
                    self.simulation_space.spatial_limits.x,
                    self.simulation_space.spatial_limits.y,
                ]
            ):
                if lim is None:
                    continue
                start, end = lim
                if self.probe_diameter > end - start:
                    raise ValueError(
                        f"Probe diameter must be smaller than the range of dimension {i}."
                    )

        self.num_probes = simulation_space.num_probes
        self.num_tilts = len(self.probe_tilts)

        self.simulation_space = simulation_space
        self.solve_reduced_domain = simulation_space.solve_reduced_domain
        self.disk_edge_blur = 0.5  # matches previous behavior (_disk_blur)

    # ---------------------------- public ---------------------------------

    def build_probes(self) -> np.ndarray:
        """

        Build the (tilt, scan, [...space...]) probe tensor.

        Returns
        -------
        ndarray (complex)
            Shape:
              1D: (num_tilts, num_probes, nx)
              2D: (num_tilts, num_probes, nx, ny)
        """
        probes = self.create_empty_probe_stack()
        for a_idx, tilt in enumerate(self.probe_tilts):
            for s_idx in range(self.num_probes):
                probes[a_idx, s_idx, ...] = self.make_single_probe(
                    scan=s_idx,
                    probe_tilt=tilt,
                )
        return probes

    def create_empty_probe_stack(self):
        return np.zeros(
            (
                self.num_tilts,
                self.num_probes,
                *self.simulation_space.effective_shape[:-1],
            ),
            dtype=complex,
        )

    def make_single_probe(
        self,
        scan: int,
        probe_type: Optional[str] = None,
        probe_focus: Optional[float] = None,
        probe_tilt: Optional[float] = 0.0,
    ) -> np.ndarray:
        """
        Build a single probe field for a given scan with optional phase terms.

        Parameters
        ----------
        scan : int
            Index into `simulation_space.scan_frame_info`.
        probe_type : str, optional
            Name of the base probe profile. If None, uses
            `simulation_space.probe_type`. Supported:
            {"constant","gaussian","sinusoidal","complex_exp",
             "dirichlet_test","neumann_test","airy_disk","disk","blurred_disk"}.
        probe_focus : float, optional
            Focal length (meters or consistent units). If provided and finite,
            a quadratic phase exp[-i k r^2/(2 f)] is applied.
        probe_tilt : float or (float, float), optional
            Linear-tilt tilts in radians. In 1D, provide a single float (x tilt).
            In 2D, provide a float (x tilt only) or (tilt_x, tilt_y).

        Returns
        -------
        field : ndarray of complex
            Complex-valued probe field with the correct dimensionality
            (1D: shape (nx,) ; 2D: shape (nx, ny)).
        """
        # Defaults from simulation_space if not given
        if probe_type is None:
            probe_type = self.probe_type
        if probe_focus is None:
            probe_focus = self.probe_focus

        # Coordinates and probe center
        coord = self._get_coordinates(scan)
        center = self._get_center(scan)

        # Base amplitude/profile
        if self.probe_diameter is not None:
            radius = max(self.probe_diameter / 2.0, 1e-12)
        else:
            radius = None

        field = self._build_profile(probe_type, coord, center, radius)

        # Phase terms
        k = self.simulation_space.k
        if probe_focus is not None and np.isfinite(probe_focus) and probe_focus != 0.0:
            field = self.add_quadratic_focus(field, coord, center, k, probe_focus)

        tilt_x, tilt_y = self._normalize_tilts(probe_tilt)
        if tilt_x != 0.0 or (self.simulation_space.dimension == 3 and tilt_y != 0.0):
            field = self.add_linear_tilt(field, coord, center, k, tilt_x, tilt_y)

        return field

    def add_linear_tilt(
        self,
        field: np.ndarray,
        coord: Tuple[np.ndarray, Optional[np.ndarray]],
        center: Union[float, Tuple[float, float]],
        k: float,
        tilt_x: float,
        tilt_y: float = 0.0,
    ) -> np.ndarray:
        """
        Apply a linear (tilt) phase exp[i k ((x-cx) sin(ax) + (y-cy) sin(ay))].

        Parameters
        ----------
        field : ndarray (complex)
            Input complex field (1D or 2D).
        coord : tuple
            (x,) in 1D or (x_mesh, y_mesh) in 2D, with 'ij' indexing.
        center : float or (float, float)
            Probe center in continuous coordinates.
        k : float
            Wavenumber (2π/λ).
        tilt_x : float
            Tilt tilt around x in radians (0 → no tilt).
        tilt_y : float, optional
            Tilt tilt around y in radians. Ignored in 1D.

        Returns
        -------
        ndarray (complex)
            Field with linear tilt applied.
        """
        if self.simulation_space.dimension == 2:
            (x_mesh,) = coord
            cx = center
            phase = np.exp(1j * k * (x_mesh - cx) * np.sin(tilt_x))
            return field * phase

        x_mesh, y_mesh = coord
        cx, cy = center
        phase = np.exp(
            1j
            * (k * (x_mesh - cx) * np.sin(tilt_x) + k * (y_mesh - cy) * np.sin(tilt_y))
        )
        return field * phase

    def add_quadratic_focus(
        self,
        field: np.ndarray,
        coord: Tuple[np.ndarray, Optional[np.ndarray]],
        center: Union[float, Tuple[float, float]],
        k: float,
        focal_length: float,
    ) -> np.ndarray:
        """
        Apply a quadratic (defocus) phase exp[-i k r^2 / (2 f)].

        Parameters
        ----------
        field : ndarray (complex)
            Input complex field (1D or 2D).
        coord : tuple
            (x,) in 1D or (x_mesh, y_mesh) in 2D, with 'ij' indexing.
        center : float or (float, float)
            Probe center in continuous coordinates.
        k : float
            Wavenumber (2π/λ).
        focal_length : float
            Focal length f. Positive for focusing (convention as used here).

        Returns
        -------
        ndarray (complex)
            Field with quadratic phase applied.
        """
        if self.simulation_space.dimension == 2:
            (x_mesh,) = coord
            cx = center
            r2 = (x_mesh - cx) ** 2
        else:
            x_mesh, y_mesh = coord
            cx, cy = center
            r2 = (x_mesh - cx) ** 2 + (y_mesh - cy) ** 2

        phase = np.exp(-1j * k * r2 / (2.0 * focal_length))
        return field * phase

    # ------------------------ Probe Types -----------------------------

    def _build_profile(
        self,
        probe_type: str,
        coord: Tuple[np.ndarray, Optional[np.ndarray]],
        center: Union[float, Tuple[float, float]],
        radius: float,
    ) -> np.ndarray:
        """Dispatch to a specific base-profile generator."""
        if probe_type == "constant":
            return self._constant(coord)

        if probe_type == "gaussian":
            return self._gaussian(
                coord, center, sd=(radius) / 3
            )  # at 3*sd a gaussian is ~0.01

        if probe_type == "sinusoidal":
            return self._sinusoidal(coord)

        if probe_type == "complex_exp":
            return self._complex_exp(coord)

        if probe_type == "dirichlet_test":
            return self._dirichlet_test(coord)

        if probe_type == "neumann_test":
            return self._neumann_test(coord)

        if probe_type == "airy_disk":
            return self._airy_disk(coord, center, radius)

        if probe_type == "disk":
            return self._disk(coord, center, radius)

        if probe_type == "blurred_disk":
            return self._blurred_disk(coord, center, radius)

        raise ValueError(f"Unknown probe_type: {probe_type}")

    def _constant(self, coord):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            return np.ones_like(x, dtype=complex)
        x, y = coord
        return np.ones_like(x, dtype=complex)

    def _gaussian(self, coord, center, sd: float):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            cx = center
            dx = (x - cx) / sd
            return np.exp(-0.5 * dx**2).astype(complex)
        x, y = coord
        cx, cy = center
        dx = (x - cx) / sd
        dy = (y - cy) / sd
        return np.exp(-0.5 * (dx**2 + dy**2)).astype(complex)

    def _sinusoidal(self, coord):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            return np.sin(np.pi * x).astype(complex)
        x, y = coord
        return (np.sin(np.pi * x) * np.sin(np.pi * y)).astype(complex)

    def _complex_exp(self, coord):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            return (-1j * np.exp(1j * np.pi * x)).astype(complex)
        x, y = coord
        return np.exp(1j * np.pi * (x + y)).astype(complex)

    def _dirichlet_test(self, coord):
        # np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
        if self.simulation_space.dimension == 2:
            (x,) = coord
            return (
                u0_nm_dirichlet(1, 1)(x, 0.5)
                + 0.5 * u0_nm_dirichlet(5, 5)(x, 0.5)
                + 0.2 * u0_nm_dirichlet(9, 9)(x, 0.5)
            ).astype(complex)
        x, y = coord
        return (
            u0_nm_dirichlet(1, 1)(x, y)
            + 0.5 * u0_nm_dirichlet(5, 5)(x, y)
            + 0.2 * u0_nm_dirichlet(9, 9)(x, y)
        ).astype(complex)

    def _neumann_test(self, coord):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            return (
                u0_nm_neumann(1, 1)(x, 0)
                + 0.5 * u0_nm_neumann(2, 2)(x, 0)
                + 0.2 * u0_nm_neumann(5, 5)(x, 0)
            ).astype(complex)
        x, y = coord
        return (
            u0_nm_neumann(1, 1)(x, y)
            + 0.5 * u0_nm_neumann(2, 2)(x, y)
            + 0.2 * u0_nm_neumann(5, 5)(x, y)
        ).astype(complex)

    def _airy_disk(self, coord, center, radius):
        # Amp only (intensity version commonly uses [2 J1(ξ)/ξ]^2; here amplitude)
        if self.simulation_space.dimension == 2:
            (x,) = coord
            r = np.abs(x - center)
            scaled_r = np.pi * r / radius
            amp = np.ones_like(r)
            mask = scaled_r != 0
            amp[mask] = (2 * jv(1, scaled_r[mask]) / scaled_r[mask]) ** 2
            return amp.astype(complex)
        x, y = coord
        cx, cy = center
        r = np.hypot(x - cx, y - cy)
        scaled_r = np.pi * r / radius
        amp = np.ones_like(r)
        mask = scaled_r != 0
        amp[mask] = (2 * jv(1, scaled_r[mask]) / scaled_r[mask]) ** 2
        return amp.astype(complex)

    def _disk(self, coord, center, radius):
        if self.simulation_space.dimension == 2:
            (x,) = coord
            r = np.abs(x - center)
        else:
            x, y = coord
            cx, cy = center
            r = np.hypot(x - cx, y - cy)
        return np.where(r <= radius, 1.0, 0.0).astype(complex)

    def _blurred_disk(self, coord, center, radius):
        """Cosine-ramped disk with energy normalization."""
        if self.simulation_space.dimension == 2:
            (x,) = coord
            r = np.abs(x - center)
            pix_area = self.simulation_space.dx
            area = 2 * radius
        else:
            x, y = coord
            cx, cy = center
            r = np.hypot(x - cx, y - cy)
            pix_area = self.simulation_space.dx * self.simulation_space.dy
            area = np.pi * radius**2

        portion_blur = self.disk_edge_blur
        inner = radius - portion_blur * radius
        denom = max(portion_blur * radius, 1e-12)
        t = np.clip((r - inner) / denom, 0.0, 1.0)
        rim = portion_blur * (1 + np.cos(np.pi * t))
        amp = np.where(r <= inner, 1.0, np.where(r >= radius, 0.0, rim))

        # Normalize approximate energy to match ideal disk
        target = area / pix_area
        s = np.sqrt(target / (amp**2).sum())
        return (amp * s).astype(complex)

    def _get_coordinates(self, scan: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (x,) in 1D or (x_mesh, y_mesh) in 2D with 'ij' indexing."""
        if self.simulation_space.dimension == 2:
            if self.solve_reduced_domain:
                x_min, x_max = self.simulation_space.scan_frame_info[
                    scan
                ].reduced_limits_continuous.x
                x = np.linspace(x_min, x_max, self.simulation_space.effective_nx)
            else:
                x = self.simulation_space.x
            return (x,)
        # 2D
        if self.solve_reduced_domain:
            x_min, x_max = self.simulation_space.scan_frame_info[
                scan
            ].reduced_limits_continuous.x
            y_min, y_max = self.simulation_space.scan_frame_info[
                scan
            ].reduced_limits_continuous.y
            x = np.linspace(x_min, x_max, self.simulation_space.effective_nx)
            y = np.linspace(y_min, y_max, self.simulation_space.effective_ny)
        else:
            x, y = self.simulation_space.x, self.simulation_space.y
        return np.meshgrid(x, y, indexing="ij")

    def _get_center(self, scan: int) -> Union[float, Tuple[float, float]]:
        """Return probe center (cx) in 1D or (cx, cy) in 2D."""
        return self.simulation_space.scan_frame_info[
            scan
        ].probe_centre_continuous.as_tuple()

    def _normalize_tilts(
        self, probe_tilt: Union[float, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Return (tilt_x, tilt_y) for 1D/2D convenience."""
        if self.simulation_space.dimension == 2:
            if not isinstance(probe_tilt, (int, float)):
                raise ValueError("In 1D, probe_tilt must be a single float.")
            return float(probe_tilt), 0.0

        # 2D
        if isinstance(probe_tilt, (tuple, list)) and len(probe_tilt) == 2:
            ax, ay = probe_tilt
            return float(ax), float(ay)
        if isinstance(probe_tilt, (int, float)):
            return float(probe_tilt), 0.0
        raise ValueError("In 2D, probe_tilt must be a float or (tilt_x, tilt_y).")
