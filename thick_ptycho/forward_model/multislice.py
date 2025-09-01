import numpy as np
from typing import Optional
#from .linear_system import LinearSystemSetup  # Removed import
from thick_ptycho.sample_space.sample_space import SampleSpace

class MultiSliceForwardModel:
    """
    Fresnel / angular spectrum multislice forward model (1D x-z).
    """

    def __init__(self,
                 sample_space,
                 pad_factor: float = 1.5,
                 angular_spectrum: bool = True,
                 remove_global_phase: bool = True,
                 dtype=np.complex64,
                 normalize_probes: bool = True):
        assert sample_space.dimension == 1, "MultiSliceForwardModel expects 1D SampleSpace."
        self.sample_space = sample_space
        self.k = sample_space.k
        self.wavelength = 2 * np.pi / self.k
        self.nx = sample_space.nx
        self.nz = sample_space.nz
        self.dx = sample_space.dx
        self.dz = sample_space.dz
        self.num_probes = sample_space.num_probes
        self.n_medium = sample_space.n_medium
        self.dtype = dtype
        self.pad_factor = pad_factor
        self.angular_spectrum = angular_spectrum
        self.remove_global_phase = remove_global_phase
        self.normalize_probes = normalize_probes

        # Object field
        self.n_field = sample_space.n_true  # (nx, nz)

        # Detector frame info (scan positions)
        self._detector_info = sample_space.detector_frame_info

        # Probe half width (integer)
        self._probe_half_width = sample_space.probe_dimensions[0] // 2

        # Caches
        self._kernel_cache = {}
        self._kx = self._compute_kx()

        # Build initial probe stack
        self.probes = self._get_probe_stack()

    @property
    def _padded_nx(self):
        return int(np.ceil(self.nx * self.pad_factor))

    def _compute_kx(self):
        nx_eff = self._padded_nx
        fx = np.fft.fftfreq(nx_eff, d=self.dx)  # cycles / unit
        kx = 2 * np.pi * fx                     # spatial angular freq
        return kx.astype(np.float32)

    def _get_kernel(self, dz):
        key = (dz, self._padded_nx, self.angular_spectrum)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        kx = self._kx
        if self.angular_spectrum:
            inside = self.k**2 - kx**2
            kz = np.sqrt(np.clip(inside, 0.0, None))  # evanescent truncated
            H = np.exp(1j * kz * dz)
            if self.remove_global_phase:
                H *= np.exp(-1j * self.k * dz)
        else:
            # Paraxial Fresnel (drop global phase if requested)
            H = np.exp(-1j * (kx**2) * dz / (2 * self.k))
        self._kernel_cache[key] = H.astype(self.dtype)
        return self._kernel_cache[key]

    def _pad(self, f):
        if self.pad_factor <= 1.0:
            return f, (0, 0)
        nxp = self._padded_nx
        pad_total = nxp - self.nx
        l = pad_total // 2
        r = pad_total - l
        return np.pad(f, (l, r)), (l, r)

    def _unpad(self, f, pads):
        if self.pad_factor <= 1.0:
            return f
        l, r = pads
        return f[l:-r] if r > 0 else f[l:]

    def _slice_transmission(self, n_slice, dz):
        # transmission: exp(i k Δn dz), Δn = n - n_medium
        delta_n = n_slice - self.n_medium
        phase = self.k * np.real(delta_n) * dz
        absorption = - self.k * np.imag(delta_n) * dz  # if imag(n)>0 → attenuation
        return np.exp(absorption + 1j * phase).astype(self.dtype)

    def _stability_warn(self, n):
        max_phase = np.max(np.abs((n - self.n_medium).real)) * self.k * self.dz
        if max_phase > 2 * np.pi:
            print(f"[MultiSliceForwardModel] Warning: per-slice phase shift {max_phase/np.pi:.2f}π > 2π; consider reducing dz or Δn.")
        fresnel_no = self.dx**2 / (self.wavelength * self.dz)
        if fresnel_no > 1.0:
            print(f"[MultiSliceForwardModel] Warning: dx^2/(λ dz)={fresnel_no:.2f} > 1; aliasing risk. Increase pad_factor or adjust sampling.")

    def forward(self, n: Optional[np.ndarray] = None):
        if n is None:
            n = self.n_field
        assert n.shape == (self.nx, self.nz)
        self._stability_warn(n)
        field = np.zeros((self.num_probes, self.nx, self.nz), dtype=self.dtype)
        # Mid-slice RI
        n_mid = 0.5 * (n[:, :-1] + n[:, 1:])
        dz = self.dz
        H = self._get_kernel(dz)

        for p in range(self.num_probes):
            psi = self.probes[p].astype(self.dtype)
            field[p, :, 0] = psi
            for z_idx in range(self.nz - 1):
                psi *= self._slice_transmission(n_mid[:, z_idx], dz)
                psi_pad, pads = self._pad(psi)
                Psi_f = np.fft.fft(psi_pad)
                Psi_f *= H
                psi_prop = np.fft.ifft(Psi_f)
                psi = self._unpad(psi_prop, pads)
                field[p, :, z_idx + 1] = psi
        return field

    def _get_probe_stack(self):
        return [self._build_probe(p) for p in range(self.num_probes)]

    def refresh_probes(self):
        """Rebuild probes (call if scan geometry or probe definition changes)."""
        self._detector_info = self.sample_space.detector_frame_info
        self.num_probes = self.sample_space.num_probes
        self.probes = self._get_probe_stack()

    def forward_cached(self, n: Optional[np.ndarray] = None):
        """
        Forward pass caching per-slice fields & transmissions for adjoint gradient.
        Returns field (P,nx,nz), caches:
          self._psi_slices[p][k] = field BEFORE transmission/prop at slice k (z index k)
          self._T_slices[p][k]   = transmission applied between z_k and z_{k+1}
        """
        if n is None:
            n = self.n_field
        assert n.shape == (self.nx, self.nz)
        # Ensure probe stack matches current scan config
        if len(self.probes) != self.num_probes:
            self.refresh_probes()
        local_probes = self.probes  # avoid repeated attribute lookups
        self._psi_slices = []
        self._T_slices = []
        field = np.zeros((self.num_probes, self.nx, self.nz), dtype=self.dtype)
        n_mid = 0.5 * (n[:, :-1] + n[:, 1:])
        dz = self.dz
        H = self._get_kernel(dz)

        for p in range(self.num_probes):
            psi = local_probes[p]
            psi_hist = [psi.copy()]
            T_hist = []
            field[p, :, 0] = psi
            for z_idx in range(self.nz - 1):
                T = self._slice_transmission(n_mid[:, z_idx], dz)
                T_hist.append(T)
                psi = psi * T
                psi_pad, pads = self._pad(psi)
                Psi_f = np.fft.fft(psi_pad)
                Psi_f *= H
                psi_prop = np.fft.ifft(Psi_f)
                psi = self._unpad(psi_prop, pads)
                field[p, :, z_idx + 1] = psi
                psi_hist.append(psi.copy())
            self._psi_slices.append(psi_hist)
            self._T_slices.append(T_hist)
        return field

    def forward(self, n: Optional[np.ndarray] = None):
        """Stateless forward (returns field only)."""
        return self.forward_cached(n=n)

    def full_amplitude_gradient(self, meas_amp: np.ndarray, n: Optional[np.ndarray] = None):
        """
        Full (real & imag) Wirtinger-style gradient for loss:
            L = 0.5 * || |psi_L| - meas_amp ||^2
        Returns:
            grad_real (nx, nz), grad_imag (nx, nz)
        Notes:
            - Transmission T = exp( -k * Im(Δn) * dz + i k Re(Δn) * dz )
              with Δn = n - n_medium.
              dT/d(Re n) =  i k dz * T
              dT/d(Im n) = - k dz * T
            - Forward uses midpoint refractive index n_mid = 0.5(n_j + n_{j+1}).
              Gradient at each midpoint is split equally to its two adjacent planes.
        """
        if n is None:
            n = self.n_field
        assert n.shape == (self.nx, self.nz)
        field = self.forward_cached(n=n)  # caches psi & T
        dz = self.dz
        H = self._get_kernel(dz)
        H_conj = np.conj(H)

        grad_mid_real = np.zeros((self.nx, self.nz - 1), dtype=np.float64)
        grad_mid_imag = np.zeros((self.nx, self.nz - 1), dtype=np.float64)

        for p in range(self.num_probes):
            psi_final = field[p, :, -1]
            amp = np.abs(psi_final) + 1e-12
            diff = amp - meas_amp
            # dL/d psi_L (Wirtinger) = diff * psi_L / |psi_L|
            sens = diff * (psi_final / amp)

            # Backward over slices
            for j in range(self.nz - 2, -1, -1):
                # Propagate sensitivity back: sens_before_transmission
                sens_pad, pads = self._pad(sens)
                Sf = np.fft.fft(sens_pad)
                Sf *= H_conj
                s_prop = np.fft.ifft(Sf)
                s_before_trans = self._unpad(s_prop, pads)  # before propagation, after transmission

                psi_before = self._psi_slices[p][j]          # before transmission
                T = self._T_slices[p][j]                     # transmission between j and j+1

                # Chain rule:
                # dL/dT = s_before_trans * conj(psi_before)
                dL_dT = s_before_trans * np.conj(psi_before)

                # dT/d(Re n_mid) = i k dz T
                # dT/d(Im n_mid) = - k dz T
                dT_dRe = 1j * self.k * dz * T
                dT_dIm = - self.k * dz * T

                # For real-valued loss, gradient w.r.t real variables:
                # grad_Re += Re( conj(dL_dT) * dT_dRe )
                # grad_Im += Re( conj(dL_dT) * dT_dIm )
                grad_mid_real[:, j] += (np.conj(dL_dT) * dT_dRe).real
                grad_mid_imag[:, j] += (np.conj(dL_dT) * dT_dIm).real

                # Propagate sensitivity further back (to previous slice) through transmission:
                sens = s_before_trans * np.conj(T)

        # Distribute midpoint gradients to object planes:
        grad_real = np.zeros_like(n, dtype=np.float64)
        grad_imag = np.zeros_like(n, dtype=np.float64)
        # Midpoint j corresponds to between z_j and z_{j+1}
        grad_real[:, :-1] += 0.5 * grad_mid_real
        grad_real[:, 1:]  += 0.5 * grad_mid_real
        grad_imag[:, :-1] += 0.5 * grad_mid_imag
        grad_imag[:, 1:]  += 0.5 * grad_mid_imag

        # Average over probes
        grad_real /= max(1, self.num_probes)
        grad_imag /= max(1, self.num_probes)
        return grad_real, grad_imag

    def adjoint_amplitude_gradient(self, meas_amp: np.ndarray, n: Optional[np.ndarray] = None):
        """
        Compute approximate gradient dL/d(Re(n)) for loss:
          L = 0.5 * || |psi_L| - meas_amp ||^2  (summed over probes, x)
        Uses single-pass adjoint (amplitude) ignoring ∂|psi|/∂Im(n) separation.
        Returns grad_n shape (nx, nz) real-valued.
        """
        field = self.forward_cached(n=n)
        dz = self.dz
        H = self._get_kernel(dz)
        H_conj = np.conj(H)
        grad_n = np.zeros((self.nx, self.nz), dtype=np.float64)

        for p in range(self.num_probes):
            psi_final = field[p, :, -1]
            amp = np.abs(psi_final) + 1e-12
            diff = (amp - meas_amp)
            # dL/dpsi_L
            sens = diff * (psi_final / amp)  # complex
            # Backprop slice-by-slice
            for z_rev in range(self.nz - 2, -1, -1):
                # Current slice (before transmission/prop stored in psi_slices)
                psi_before = self._psi_slices[p][z_rev]
                if z_rev < self.nz - 1:
                    T = self._T_slices[p][z_rev]
                    # Contribution to gradient at mid-slice (approx real part)
                    # d(psi_after)/d(delta_n) ≈ psi_before * T * (1j*k*dz)
                    dT_factor = (1j * self.k * dz)
                    contrib = psi_before * T * dT_factor
                    # projection: real sensitivity
                    grad_n[:, z_rev + 1] += (np.conj(sens) * contrib).real
                    # Propagate sensitivity backwards:
                    # sens_before = conj(T) * adjointProp(sens_after)
                    sens_pad, pads = self._pad(sens)
                    Sens_f = np.fft.fft(sens_pad)
                    Sens_f *= H_conj
                    sens_prop = np.fft.ifft(Sens_f)
                    sens = self._unpad(sens_prop, pads)
                    sens = sens * np.conj(T)
            # (Optionally could accumulate at z=0 mid-slice; skipped)
        # Normalize by number of probes
        grad_n /= max(1, self.num_probes)
        return grad_n

    def propagate_probe_set(self, probes: np.ndarray, n: Optional[np.ndarray] = None):
        saved = self.probes
        self.probes = probes
        out = self.forward(n=n)
        self.probes = saved
        return out

