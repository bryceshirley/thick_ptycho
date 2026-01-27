import numpy as np
from typing import Optional
from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver


class SpectralCNForwardModelSolver(BaseForwardModelSolver):
    """
    Paraxial Wave Equation Solver using Crank-Nicolson scheme with 
    Shifted Preconditioned Richardson Iteration.
    
    Dimensions: 2D Propagation (1D Transverse x, 1D Axial z).
    """

    def __init__(
        self,
        simulation_space,
        ptycho_probes,
        results_dir="",
        use_logging=False,
        verbose=False,
        log=None,
        richardson_iters=3,
        use_extrapolation=False,
        method="PARAXIAL",
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
                "3D simulation space is not yet implemented for CNForwardModelSolver."
            )

        self.k = self.simulation_space.k  # Wave number
        self.nx_eff = self.simulation_space.effective_shape[0]  # Effective pixels (padded)
        self.dx = self.simulation_space.dx  # Pixel size in x
        self.nx = self.simulation_space.nx  # Physical pixels in x
        self.nz = self.simulation_space.nz  # Number of slices in z
        self.dz = self.simulation_space.dz  # Distance between slices
        self.n_medium = self.simulation_space.n_medium # Refractive index of medium

        self.richardson_iters = richardson_iters
        self.use_extrapolation = use_extrapolation
        self.method = method.upper()
        if method not in ["PARAXIAL","ASM"]:
            raise ValueError(f"Unknown method '{method}' for SpectralCNForwardModelSolver.")

        # --- 1. GENERATE WAVENUMBERS (kx) ---
        fx = np.fft.fftfreq(self.nx_eff, d=self.dx)
        kx = 2 * np.pi * fx
        
        # --- 2. PRECOMPUTE LAMBDA OPERATOR ---
        # We store Lambda as a 1D array corresponding to the padded space
        if self.method == "ASM":
            # --- EXACT HELMHOLTZ (with Carrier Removal) ---
            # Full operator: i * sqrt(k^2 - kx^2)
            # Envelope operator: i * (sqrt(k^2 - kx^2) - k)
            
            inside = self.k**2 - kx**2
            # Clip negative values (evanescent) to 0
            sqrt_term = np.sqrt(np.clip(inside, 0.0, None))
            
            # Add 1j. Wave propagation requires imaginary eigenvalues.
            # Subtract self.k. Removes fast oscillations (envelope).
            self.Lambda = 1j * (sqrt_term - self.k)
        else:
            # --- PARAXIAL APPROXIMATION ---
            # The standard paraxial operator is derived assuming the carrier is removed.
            # Lambda = -i * kx^2 / (2k)
            # Note: This is numerically consistent with the ASM envelope approximation:
            # i*(k(1 - kx^2/2k^2) - k) approx -i*kx^2/2k
            self.Lambda = (-1j * kx**2) / (2 * self.k)
    
    def reset_cache(self):
        pass

    def _get_operators(self, C_slice, dz):
        """
        Constructs operator eigenvalues and mean shift for a specific slice.
        Returns: (delta_C_dz, M_hat_inv)
        """
        # 1. Shifted Preconditioner Setup
        # C_bar = np.mean(C_slice)
        # delta_C = C_slice - C_bar
        
        # Scale by dz for the step
        # The equation is (I - 0.5(L+C)dz)
        # L_dz = self.Lambda * dz
        # C_bar_dz = C_bar * dz
        
        # M_hat = 1 - 0.5*L*dz - 0.5*C_bar*dz
        M_hat = 1.0 - 0.5 * (self.Lambda * dz) # - 0.5 * (C_bar * dz)
        M_inv_hat = 1.0 / M_hat
        
        # delta_C_dz = delta_C * dz
        delta_C_dz = C_slice * dz  # Using full C_slice directly
        
        return delta_C_dz, M_inv_hat

    def _step_crank_nicolson(self, u_in, C_slice, dz):
        """
        Performs a single O(dz^2) Crank-Nicolson step using Preconditioned Richardson.
        """
        delta_C_dz, M_inv_hat = self._get_operators(C_slice, dz)
        
        # Helper for M^{-1} (Diagonal in Fourier Space)
        def apply_M_inv(v):
            return np.divide(np.fft.ifft(np.fft.fft(v) * M_inv_hat), (1+0.5*delta_C_dz))

        # Helper for P = M - 0.5 * delta_C * dz
        # Applying P v = M v - 0.5 * delta_C_dz * v
        def apply_P(v):
            V = np.fft.fft(v)
            # M_hat = 1 / M_inv_hat
            # M v = IFFT( V / M_inv_hat ) ... wait, M_hat is just 1/M_inv_hat
            Mv = np.fft.ifft(V / M_inv_hat)
            return Mv - 0.5 * (delta_C_dz * v)

        # --- INNER LOOP (Preconditioned Richardson) ---
        # Solve Px = u_in
        b = u_in
        
        # Initial Guess: 0th order Neumann (x0 = M^{-1} b)
        x = apply_M_inv(b) # Free space propagation only guess
        
        for _ in range(self.richardson_iters):
            # Residual: r = b - Px
            # Px = apply_P(x)
            # Optimization: We can compute Px efficiently inside the loop or re-use logic
            # For clarity:
            Px = apply_P(x)
            r = b - Px
            
            # Correction: dx = M^{-1} r
            dx = apply_M_inv(r)
            x = x + dx

        # --- OUTER UPDATE (Cayley Transform) ---
        # u^{n+1} = 2x - u^n
        u_out = 2 * x - u_in
        
        return u_out

    def propagate_slice(self, u_in, C_slice, dz):
        """Propagates through a single slice, optionally using Richardson Extrapolation."""
        if not self.use_extrapolation:
            return self._step_crank_nicolson(u_in, C_slice, dz)
        else:
            # Richardson Extrapolation for O(dz^4)
            # 1. One Coarse Full Step
            u_full = self._step_crank_nicolson(u_in, C_slice, dz)
            
            # 2. Two Fine Half Steps
            u_half = self._step_crank_nicolson(u_in, C_slice, dz / 2.0)
            u_half = self._step_crank_nicolson(u_half, C_slice, dz / 2.0)
            
            # 3. Combine: (4*S_n/2 - S_n) / 3
            return (4.0 * u_half - u_full) / 3.0

    def _solve_single_probe(
        self,
        scan_idx: int = 0,
        proj_idx: int = 0,
        n: Optional[np.ndarray] = None,
        mode="forward",
        probe: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform forward propagation through multiple slices for a single probe position.
        Note: Adjoint/Backward is not implemented for CN yet.
        """
        if mode != "forward":
             raise NotImplementedError("Only forward mode is implemented for Crank-Nicolson solver.")

        # 1. Get Refractive Index Distribution (n)
        if n is None:
             n = self.simulation_space.refractive_index_empty
        
        # n typically has shape (nx, nz). We need to project/rotate if needed.
        n = self.get_projected_obj(n, mode=mode, proj_idx=proj_idx)

        # 2. Create Refractive Index Contribution Operator C with Second Order Accuracy
        # We compute C for all slices at once for efficiency
        C = self.simulation_space.create_object_contribution(
            n=n, scan_index=scan_idx
        ).reshape(-1, self.nz - 1) # Handles padding internally

        # 3. Get Padded Probe
        if probe is None:
            probe = self.probes[proj_idx, scan_idx, :] # Padding handled internally

        # 4. Initialize History
        # u will store the wavefield at each slice (padded size)
        u = np.empty((self.nx_eff, self.nz), dtype=complex)
        u[:, 0] = probe

        # 5. Propagation Loop
        # Unlike Multislice which splits Transmit/Propagate, 
        # CN handles the full slab (Laplacian + Potential) in one step per slice.
        for z in range(self.nz-1):
            # Propagate through slice thickness dz
            u[:, z+1] = self.propagate_slice(u[:, z], C[:, z], self.dz)

        # psi_current is now the exit wavefield
        return u, u[:, -1]