import numpy as np
from scipy.special import jv
from typing import Optional, Tuple, Union

def u0_nm_neumann(n, m):
    """Return exact Neumann solution basis u(x,y) = cos(nπx)cos(mπy)."""
    return lambda x, y: np.cos(n * np.pi * x) * np.cos(m * np.pi * y)


def u0_nm_dirichlet(n, m):
    """Return exact Dirichlet solution basis u(x,y) = sin(nπx)sin(mπy)."""
    return lambda x, y: np.sin(n * np.pi * x) * np.sin(m * np.pi * y)


class Probes:
    """
    Probe field generator with reusable phase utilities.

    This class contains all probe profile definitions (constant, gaussian, disk,
    airy_disk, etc.) and the phase-operations used to modify them (linear tilt and
    quadratic focus). It is designed to be stateless w.r.t. solver internals and
    only depends on `sample_space` and per-scan metadata to assemble fields.

    Parameters
    ----------
    sample_space : object
        An object providing geometric and discretization info, e.g.
        - dimension : int (1 or 2)
        - nx[, ny] : int
        - x[, y] : ndarray (grid coordinates)
        - dx[, dy] : float (pixel size)
        - k : float (wavenumber)
        - probe_diameter_continuous : float
        - detector_frame_info[scan] : dict with
            * "sub_dimensions": (x,) in 1D or (x, y) in 2D when thin_sample=True
            * "probe_centre_continuous": float in 1D, (cx, cy) in 2D
    thin_sample : bool
        If True, use sub-sampled grids from `detector_frame_info`.

    Attributes
    ----------
    disk_edge_blur : float
        Fraction of radius used for the cosine-ramp edge in the "blurred_disk"
        probe (default 0.5).
    """

    def __init__(self, sample_space, thin_sample: bool, angles_list=(0.0,)):
        self.sample_space = sample_space
        self.thin_sample = thin_sample
        self.disk_edge_blur = 0.5  # matches previous behavior (_disk_blur)
        self.probe_angles_list = list(angles_list)
        self.num_angles = len(self.probe_angles_list)

    # ---------------------------- public ---------------------------------

    def build_probes(self) -> np.ndarray:
        """
        Build the (angle, scan, [...space...]) probe tensor.

        Returns
        -------
        ndarray (complex)
            Shape:
              1D: (num_angles, num_probes, nx)
              2D: (num_angles, num_probes, nx, ny)
        """
        if self.sample_space.dimension == 1:
            probes = np.zeros((self.num_angles, self.sample_space.num_probes, self.nx), dtype=complex)
        else:
            probes = np.zeros((self.num_angles, self.sample_space.num_probes, self.nx, self.ny), dtype=complex)

        for a_idx, angle in enumerate(self.probe_angles_list):
            for s_idx in range(self.sample_space.num_probes):
                probes[a_idx, s_idx, ...] = self.make_single_probe(
                    scan=s_idx,
                    probe_type=self.sample_space.probe_type,
                    probe_focus=self.sample_space.probe_focus,
                    probe_angle=angle,
                )
        return probes

    def make_single_probe(
        self,
        scan: int,
        probe_type: Optional[str] = None,
        probe_focus: Optional[float] = None,
        probe_angle: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """
        Build a single probe field for a given scan with optional phase terms.

        Parameters
        ----------
        scan : int
            Index into `sample_space.detector_frame_info`.
        probe_type : str, optional
            Name of the base probe profile. If None, uses
            `sample_space.probe_type`. Supported:
            {"constant","gaussian","sinusoidal","complex_exp",
             "dirichlet_test","neumann_test","airy_disk","disk","blurred_disk"}.
        probe_focus : float, optional
            Focal length (meters or consistent units). If provided and finite,
            a quadratic phase exp[-i k r^2/(2 f)] is applied.
        probe_angle : float or (float, float), optional
            Linear-tilt angles in radians. In 1D, provide a single float (x tilt).
            In 2D, provide a float (x tilt only) or (angle_x, angle_y).

        Returns
        -------
        field : ndarray of complex
            Complex-valued probe field with the correct dimensionality
            (1D: shape (nx,) ; 2D: shape (nx, ny)).
        """
        # Defaults from sample_space if not given
        if probe_type is None:
            probe_type = self.sample_space.probe_type
        if probe_focus is None:
            probe_focus = self.sample_space.probe_focus
        if probe_angle is None:
            probe_angle = self.sample_space.probe_angle

        # Coordinates and probe center
        coord = self._get_coordinates(scan)
        center = self._get_center(scan)

        # Base amplitude/profile
        radius = self.sample_space.probe_diameter_continuous / 2.0
        field = self._build_profile(probe_type, coord, center, radius)

        # Phase terms
        k = self.sample_space.k
        if probe_focus is not None and np.isfinite(probe_focus) and probe_focus != 0.0:
            field = self.add_quadratic_focus(field, coord, center, k, probe_focus)

        angle_x, angle_y = self._normalize_angles(probe_angle)
        if angle_x != 0.0 or (self.sample_space.dimension == 2 and angle_y != 0.0):
            field = self.add_linear_tilt(field, coord, center, k, angle_x, angle_y)

        return field

    def add_linear_tilt(
        self,
        field: np.ndarray,
        coord: Tuple[np.ndarray, Optional[np.ndarray]],
        center: Union[float, Tuple[float, float]],
        k: float,
        angle_x: float,
        angle_y: float = 0.0,
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
        angle_x : float
            Tilt angle around x in radians (0 → no tilt).
        angle_y : float, optional
            Tilt angle around y in radians. Ignored in 1D.

        Returns
        -------
        ndarray (complex)
            Field with linear tilt applied.
        """
        if self.sample_space.dimension == 1:
            (x_mesh,) = coord
            cx = center
            phase = np.exp(1j * k * (x_mesh - cx) * np.sin(angle_x))
            return field * phase

        x_mesh, y_mesh = coord
        cx, cy = center
        phase = np.exp(1j * (k * (x_mesh - cx) * np.sin(angle_x) +
                             k * (y_mesh - cy) * np.sin(angle_y)))
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
        if self.sample_space.dimension == 1:
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
            return self._gaussian(coord, center, sd=max(radius / 2.0, 1e-12))

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
        if self.sample_space.dimension == 1:
            (x,) = coord
            return np.ones_like(x, dtype=complex)
        x, y = coord
        return np.ones_like(x, dtype=complex)

    def _gaussian(self, coord, center, sd: float):
        if self.sample_space.dimension == 1:
            (x,) = coord
            cx = center
            dx = (x - cx) / sd
            return np.exp(-0.5 * dx ** 2).astype(complex)
        x, y = coord
        cx, cy = center
        dx = (x - cx) / sd
        dy = (y - cy) / sd
        return np.exp(-0.5 * (dx ** 2 + dy ** 2)).astype(complex)

    def _sinusoidal(self, coord):
        if self.sample_space.dimension == 1:
            (x,) = coord
            return np.sin(np.pi * x).astype(complex)
        x, y = coord
        return (np.sin(np.pi * x) * np.sin(np.pi * y)).astype(complex)

    def _complex_exp(self, coord):
        if self.sample_space.dimension == 1:
            (x,) = coord
            return (-1j * np.exp(1j * np.pi * x)).astype(complex)
        x, y = coord
        return np.exp(1j * np.pi * (x + y)).astype(complex)

    def _dirichlet_test(self, coord):
        # np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
        if self.sample_space.dimension == 1:
            (x,) = coord
            return (u0_nm_dirichlet(1, 1)(x, 0.5)+
                    0.5 * u0_nm_dirichlet(5, 5)(x, 0.5) +
                    0.2 * u0_nm_dirichlet(9, 9)(x, 0.5)).astype(complex)
        x, y = coord
        return (
            u0_nm_dirichlet(1, 1)(x, y) +
            0.5 * u0_nm_dirichlet(5, 5)(x, y) +
            0.2 * u0_nm_dirichlet(9, 9)(x, y)
        ).astype(complex)

    def _neumann_test(self, coord):
        if self.sample_space.dimension == 1:
            (x,) = coord
            return (u0_nm_neumann(1, 1)(x, 0) +
                    0.5 * u0_nm_neumann(2, 2)(x, 0) +
                    0.2 * u0_nm_neumann(5, 5)(x, 0)
                 ).astype(complex)
        x, y = coord
        return (u0_nm_neumann(1, 1)(x, y) +
                0.5 * u0_nm_neumann(2, 2)(x, y) +
                0.2 * u0_nm_neumann(5, 5)(x, y)
                ).astype(complex)

    def _airy_disk(self, coord, center, radius):
        # Amp only (intensity version commonly uses [2 J1(ξ)/ξ]^2; here amplitude)
        if self.sample_space.dimension == 1:
            (x,) = coord
            r = np.abs(x - center)
            s = np.pi * r / max(radius, 1e-12)
            amp = np.ones_like(r)
            m = s != 0
            amp[m] = (2 * jv(1, s[m]) / s[m]) ** 2
            return amp.astype(complex)
        x, y = coord
        cx, cy = center
        r = np.hypot(x - cx, y - cy)
        s = np.pi * r / max(radius, 1e-12)
        amp = np.ones_like(r)
        m = s != 0
        amp[m] = (2 * jv(1, s[m]) / s[m]) ** 2
        return amp.astype(complex)

    def _disk(self, coord, center, radius):
        if self.sample_space.dimension == 1:
            (x,) = coord
            r = np.abs(x - center)
        else:
            x, y = coord
            cx, cy = center
            r = np.hypot(x - cx, y - cy)
        return np.where(r <= radius, 1.0, 0.0).astype(complex)

    def _blurred_disk(self, coord, center, radius):
        """Cosine-ramped disk with energy normalization."""
        if self.sample_space.dimension == 1:
            (x,) = coord
            r = np.abs(x - center)
            pix_area = self.sample_space.dx
            area = (2 * radius)
        else:
            x, y = coord
            cx, cy = center
            r = np.hypot(x - cx, y - cy)
            pix_area = self.sample_space.dx * self.sample_space.dy
            area = np.pi * radius ** 2

        portion_blur = self.disk_edge_blur
        inner = radius - portion_blur * radius
        denom = max(portion_blur * radius, 1e-12)
        t = np.clip((r - inner) / denom, 0.0, 1.0)
        rim = portion_blur * (1 + np.cos(np.pi * t))
        amp = np.where(r <= inner, 1.0, np.where(r >= radius, 0.0, rim))

        # Normalize approximate energy to match ideal disk
        target = area / pix_area
        s = np.sqrt(target / (amp ** 2).sum())
        return (amp * s).astype(complex)

    def _get_coordinates(self, scan: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (x,) in 1D or (x_mesh, y_mesh) in 2D with 'ij' indexing."""
        if self.sample_space.dimension == 1:
            if self.thin_sample:
                x = self.sample_space.detector_frame_info[scan]["sub_dimensions"][0]
            else:
                x = self.sample_space.x
            return (x,)
        # 2D
        if self.thin_sample:
            x, y = self.sample_space.detector_frame_info[scan]["sub_dimensions"]
        else:
            x, y = self.sample_space.x, self.sample_space.y
        return np.meshgrid(x, y, indexing="ij")

    def _get_center(self, scan: int) -> Union[float, Tuple[float, float]]:
        """Return probe center (cx) in 1D or (cx, cy) in 2D."""
        info = self.sample_space.detector_frame_info[scan]["probe_centre_continuous"]
        return info if self.sample_space.dimension == 2 else float(info)

    def _normalize_angles(
        self, probe_angle: Union[float, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Return (angle_x, angle_y) for 1D/2D convenience."""
        if self.sample_space.dimension == 1:
            if not isinstance(probe_angle, (int, float)):
                raise ValueError("In 1D, probe_angle must be a single float.")
            return float(probe_angle), 0.0

        # 2D
        if isinstance(probe_angle, (tuple, list)) and len(probe_angle) == 2:
            ax, ay = probe_angle
            return float(ax), float(ay)
        if isinstance(probe_angle, (int, float)):
            return float(probe_angle), 0.0
        raise ValueError("In 2D, probe_angle must be a float or (angle_x, angle_y).")