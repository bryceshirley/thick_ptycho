
import typing

from typing import Optional
import numpy as np
from scipy.fft import dct, idct, dst, idst 
from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver

class MSForwardModelSolver(BaseForwardModelSolver):
    """Angular spectrum multislice forward/backward propagation."""

    def __init__(
        self,
        simulation_space,
        ptycho_probes,
        results_dir="",
        use_logging=False,
        verbose=False,
        log=None,
        method="ASM",
        transform="FFT",
        pruning_fraction=0.0,
        symmetric_strang_splitting=False,
    ):
        super().__init__(
            simulation_space,
            ptycho_probes,
            results_dir=results_dir,
            use_logging=use_logging,
            verbose=verbose,
            log=log,
        )
        if self.simulation_space.dimension == 3:
            raise NotImplementedError(
                "3D simulation space is not yet implemented for MSForwardModelSolver."
            )

        self.k = self.simulation_space.k  # Wave number
        self.nx_eff = self.simulation_space.effective_shape[0]  # Effective refractive index Including empty space padding
        self.dx = self.simulation_space.dx  # Pixel size in x
        self.nx = self.simulation_space.nx  # Number of pixels in x
        self.nz = self.simulation_space.nz  # Number of slices in z
        self.dz = self.simulation_space.dz  # Distance between slices
        self.n_medium = (
            self.simulation_space.n_medium
        )  # Refractive index of surrounding medium
        self.method = method.upper()  # 'ASM' or 'PARAXIAL'
        self.transform = transform # 'FFT', 'DCT', 'DST', 'absorbing_boundary'
        if self.transform not in ['FFT', 'DCT', 'DST', 'absorbing_boundary']:
            raise ValueError(f"Unknown transform type: {self.transform}")
        if self.method not in ['ASM', 'PARAXIAL']:
            raise ValueError(f"Unknown method: {self.method}")
        self.pruning_fraction = pruning_fraction  # Fraction of high frequencies to prune (0.0 to 1.0)
        self.symmetric_strang_splitting = symmetric_strang_splitting  # Use Strang splitting for propagation

        # Precompute angular spectrum kernels for forward/backward propagation
        self._kernel_cache = {}

    def reset_cache(self):
        self._kernel_cache = {}

    def _get_propagation_kernels(self, dz: float, transform_type: str) -> np.ndarray:
        dz = self.dz if dz is None else dz
        key = (transform_type, dz)
        
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        L = self.nx_eff * self.dx 

        # --- 1. GENERATE WAVENUMBERS (kx) ---
        if transform_type == "FFT":
            fx = np.fft.fftfreq(self.nx_eff, d=self.dx)
            kx = 2 * np.pi * fx
        elif transform_type == "DCT":
            modes = np.arange(self.nx_eff)
            kx = (np.pi * modes) / L
        elif transform_type == "DST":
            modes = (np.arange(self.nx_eff) + 1)
            kx = np.pi * modes / L
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        # --- 2. COMPUTE PROPAGATOR ---
        
        if self.method == "ASM":
            # --- EXACT HELMHOLTZ (with Carrier Removal) ---
            # Full operator: i * sqrt(k^2 - kx^2)
            # Envelope operator: i * (sqrt(k^2 - kx^2) - k)
            
            inside = self.k**2 - kx**2
            # Clip negative values (evanescent) to 0
            sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
            
            # Add 1j. Wave propagation requires imaginary eigenvalues.
            # Subtract self.k. Removes fast oscillations (envelope).
            Lambda = 1j * (sqrt_term - self.k)
            
            # Full propagator
            H = np.exp(Lambda * dz)

        elif self.method == "PARAXIAL":
            # --- PARAXIAL APPROXIMATION ---
            # The standard paraxial operator is derived assuming the carrier is removed.
            # Lambda = -i * kx^2 / (2k)
            # Note: This is numerically consistent with the ASM envelope approximation:
            # i*(k(1 - kx^2/2k^2) - k) approx -i*kx^2/2k
            Lambda = (-1j * kx**2) / (2 * self.k)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        H = np.exp(Lambda * dz)
        

        # --- 3. APPLY PRUNING---
        if self.pruning_fraction > 0.0:
            # Determine max physical frequency (Nyquist limit)
            k_max = np.pi / self.dx
            
            # Calculate cutoff frequency
            # e.g., if fraction=0.2, we keep bottom 80% (cutoff = 0.8 * k_max)
            k_cutoff = k_max * (1.0 - self.pruning_fraction)
            
            # Create binary mask
            # For FFT: handles centered low freq + negative low freq
            # For DCT/DST: handles monotonic low freq
            mask = np.abs(kx) <= k_cutoff
            
            # Apply mask to propagator
            H *= mask

        self._kernel_cache[key] = H.astype(np.complex128)
        return self._kernel_cache[key]
    
    
    def transform_absorbing_boundary(self, psi, dz, inverse=False):
        """
        Absorbing Boundary Batched 4-Processor Solver.
        """
        
        # 1. BATCH: Stack Real and Imag parts
        # shape: (2, N) -> [Real_Part, Imag_Part]
        psi_stack = np.stack([psi.real, psi.imag], axis=0)
        
        # 2. FORWARD TRANSFORMS (Standard Library, Batched)
        # DCT coefficients for [Real, Imag]
        C_stack = dct(psi_stack, type=2, norm='ortho', axis=-1)
        
        # DST coefficients for [Real, Imag]
        S_stack = dst(psi_stack, type=2, norm='ortho', axis=-1)
        
        # 3. GET KERNELS
        H_dct = self._get_propagation_kernels(dz, "DCT")
        H_dst = self._get_propagation_kernels(dz, "DST")
        
        if inverse:
            H_dct = np.conj(H_dct)
            H_dst = np.conj(H_dst)

        # 4. PROPAGATE
        # C_stack[0] is Cr, C_stack[1] is Ci
        Cr, Ci = C_stack[0], C_stack[1]
        Sr, Si = S_stack[0], S_stack[1]
        
        # Propagate DCT Branch (Mirror Universe)
        C_next_r = Cr * H_dct.real - Ci * H_dct.imag
        C_next_i = Cr * H_dct.imag + Ci * H_dct.real
        
        # Propagate DST Branch (Zero Universe)
        S_next_r = Sr * H_dst.real - Si * H_dst.imag
        S_next_i = Sr * H_dst.imag + Si * H_dst.real
        
        # Restack for inverse
        C_next_stack = np.stack([C_next_r, C_next_i], axis=0)
        S_next_stack = np.stack([S_next_r, S_next_i], axis=0)
        
        # 5. INVERSE TRANSFORMS (Standard Library, Batched)
        u_dct_stack = idct(C_next_stack, type=2, norm='ortho', axis=-1)
        u_dst_stack = idst(S_next_stack, type=2, norm='ortho', axis=-1)
        
        # 6. AVERAGE
        u_final_stack = 0.5 *(u_dct_stack + u_dst_stack)
        
        return u_final_stack[0] + 1j * u_final_stack[1]

    
    def transform_fft(self, psi, dz, inverse=False):
        """Standard Periodic Propagation"""
        # Note: If inverse=True (backprop), we conjugate the kernel
        H = self._get_propagation_kernels(dz, "FFT")
        if inverse: H = np.conj(H)
        
        Psi = np.fft.fft(psi)
        Psi *= H
        return np.fft.ifft(Psi)
    
    def transform_dct(self, psi, dz, inverse=False):
        """Neumann (Mirror) Propagation"""
        H = self._get_propagation_kernels(dz, "DCT")
        if inverse: H = np.conj(H)

        # 1. Forward Transforms (Real and Imag separately)
        # We must cast to complex because the rotation H is complex
        Psi_real = dct(np.real(psi), type=2, norm="ortho")
        Psi_imag = dct(np.imag(psi), type=2, norm="ortho")
        
        # 2. Mix into complex domain
        Psi_complex = Psi_real + 1j * Psi_imag
        
        # 3. Apply Propagator (Rotation)
        Psi_complex *= H
        
        # 4. Inverse Transforms
        # Note: idct handles real arrays. We must split again.
        psi_out_real = idct(np.real(Psi_complex), type=2, norm="ortho")
        psi_out_imag = idct(np.imag(Psi_complex), type=2, norm="ortho")
        
        return psi_out_real + 1j * psi_out_imag

    def transform_dst(self, psi, dz, inverse=False):
        """Dirichlet (Zero-Value) Propagation"""
        H = self._get_propagation_kernels(dz, "DST")
        if inverse: H = np.conj(H)

        # Use actual DST functions!
        Psi_real = dst(np.real(psi), type=2, norm="ortho")
        Psi_imag = dst(np.imag(psi), type=2, norm="ortho")
        
        Psi_complex = Psi_real + 1j * Psi_imag
        Psi_complex *= H
        
        psi_out_real = idst(np.real(Psi_complex), type=2, norm="ortho")
        psi_out_imag = idst(np.imag(Psi_complex), type=2, norm="ortho")
        
        return psi_out_real + 1j * psi_out_imag


    def _propagate(self, psi, dz):
        # Dispatcher
        if self.transform == "FFT":
            return self.transform_fft(psi, dz, inverse=False)
        elif self.transform == "DCT":
            return self.transform_dct(psi, dz, inverse=False)
        elif self.transform == "DST":
            return self.transform_dst(psi, dz, inverse=False)
        elif self.transform == "absorbing_boundary":
            return self.transform_absorbing_boundary(psi, dz, inverse=False)

    def _backpropagate(self, psi, dz):
        """
        The backward pass must use the ADJOINT of the forward transform.
        If forward used DCT, backward MUST use DCT with conjugated kernel.
        """
        if self.transform == "FFT":
            return self.transform_fft(psi, dz, inverse=True)
        elif self.transform == "DCT":
            return self.transform_dct(psi, dz, inverse=True)
        elif self.transform == "DST":
            return self.transform_dst(psi, dz, inverse=True)
        elif self.transform == "absorbing_boundary":
            return self.transform_absorbing_boundary(psi, dz, inverse=True)

    def _object_transmission_function(self, n_slice):
        """Compute the object transmission function for a given slice."""
        return np.exp(1j * self.k * (n_slice - self.n_medium) * self.dz)
    
    def object_to_refractive_index(self, obj_slice):
        return self.n_medium - 1j * np.log(obj_slice) / (self.k * self.dz)


    def _solve_single_probe(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
        n: Optional[np.ndarray] = None,
        obj: Optional[np.ndarray] = None,
        mode="forward",
        probe: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform forward (or backward) propagation through multiple slices
        for a single probe position, following Eq. (2) in Maiden et al. (2012).

        Parameters
        ----------
        angle_idx : int
            Index of the illumination angle.
        scan_idx : int
            Index of the probe position.
        n : ndarray, optional
            Refractive index distribution. If None, uses self.simulation_space.n_true.
        mode : {'forward', 'reverse', 'forward_rotated', 'reverse_rotated'}, optional
            If 'reverse', performs backward propagation (adjoint operation).
            If 'forward', performs forward propagation.
        **kwargs : dict, optional
            Additional arguments (not used here).

        Returns
        -------
        wavefield_through_slices : ndarray
            Complex-valued wavefield at each slice along z. Shape (nx, nz).
        """
        assert mode in {
            "forward",
            "backward",
        }, f"Invalid mode: {mode}"

        # Object refractive index distribution
        if obj is None:
            if n is None:
                n = self.simulation_space.refractive_index_empty

            n_proj = self.get_projected_obj(n, mode=mode, proj_idx=proj_idx)

            n_proj = self.simulation_space.create_object_contribution(n=n_proj,mid_n=True, scan_index=scan_idx)
            n = n_proj.reshape(n.shape[0],)  # Shape (nx, nz-1)
            obj = np.array([self._object_transmission_function(n_slice=n[:, z]) for z in range(self.nz-1)]).T

        # Select initial probe condition
        if probe is None:
            probe = self.probes[proj_idx, scan_idx, :]

        # Initial probe field at the entrance plane
        psi_current = probe
        u = np.empty((obj.shape[0], self.nz), dtype=complex)

        # --- 1. ENTRANCE HALF-STEP (Strang Only) ---``
        if self.symmetric_strang_splitting:
            if mode == "forward":
                psi_current = self._propagate(psi_current, self.dz / 2.0)
            else:
                # Backward pass adjoints the forward operation order
                psi_current = self._backpropagate(psi_current, self.dz / 2.0)

        # Propagation through slices
        # For backward pass, we iterate slices in REVERSE order: self.nz-1 down to 0
        if mode == "backward":
            z_range = range(self.nz - 1, -1, -1)
        else:
            z_range = range(self.nz)

        # Forward propagation through slices
        for i, z in enumerate(z_range):
            # Transmission through current slice
            # Reference:
            #   F. Wittwer, J. Hagemann, D. Brückner, S. Flenner, and C. G. Schroer,
            #   "Phase retrieval framework for direct reconstruction of the projected refractive index
            #   applied to ptychography and holography," *Optica*, vol. 9, no. 3, pp. 288–297, 2022.
            #   DOI: https://doi.org/10.1364/OPTICA.447021
            u[:, z] = psi_current
            if mode == "forward":
                psi_exit  =  psi_current * obj[:, z]
            else:
                # Conjugate transmission for backprop
                psi_exit  =  psi_current * np.conj(obj[:, z])

            # B. PROPAGATE (Full Step between slices)
            if mode == "forward":
                psi_current = self._propagate(psi_exit, self.dz)
            else:
                psi_current = self._backpropagate(psi_exit, self.dz)

        
        # --- 3. EXIT HALF-STEP (Strang Only) ---
        # After the last transmission, we still have half a propagation to leave the sample
        if self.symmetric_strang_splitting:
            if mode == "forward":
                psi_current = self._propagate(psi_current, self.dz / 2.0)
            else:
                psi_current = self._backpropagate(psi_current, self.dz / 2.0)

        return u, psi_current
