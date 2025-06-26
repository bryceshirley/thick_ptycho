# ThickPtyPy: Thick Sample Ptychography Simulations

**ThickPtyPy** is a Python project for simulating and reconstructing thick-sample 
ptychography using paraxial approximations. It includes solvers for the forward 
model in 2D, and 3D using a finite difference scheme to solve the paraxial wave
equation, as well as reconstruction algorithms such as least squares
minimization of simplified thick-sample ptychography model.

---

## Jupyter Notebooks

* `1_forward_model_3d.ipynb` — This tutorial demonstrates how to use `thickptypy` to set up and solve a 3D forward problem with multiple probes, both iteratively and as a full system.
* `2_ptychography_simple_example.ipynb` — This tutorial uses EPie reconstructions assuming the simulated data is of a thin sample.
* `3_forward_model_2d.ipynb` — This tutorial demonstrates how to use `thickptypy` to set up and solve a 2D forward problem with multiple probes, both iteratively and as a full system.

* `4_least_squares_1d.ipynb` — This tutorial demonstrates how to use `thickptypy` to set up and solve a least_squares problem to reconstuct a thick sample.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bryceshirley/thickptypy.git
   cd thickptypy
   ```

2. **Install dependencies (requires poetry to be installed)**:

   ```bash
   poetry install
   ```

3. **Activate the virtual environment. The following command prints the activate command to run**:

   ```bash
   poetry env activate
   ```

4. **Activate Kernel for Jupyter Notebooks:**

   ```bash
   poetry run ipython kernel install --name "thickptypy" --user
   ```
---

## Project Structure

```
thickptypy/
├── README.md                     # Project documentation
├── conftest.py                   # Test configuration
├── notebooks/                    # Jupyter notebooks for demonstrations
│   ├── 1_forward_model_3d.ipynb
│   ├── 2_ptychography_simple_example.ipynb
│   ├── 3_forward_model_2d.ipynb
│   ├── 4_least_squares_1d.ipynb
│   ├── results/                  # Output folder for notebook results
│   └── utils.py                  # Helper Functions for 2_ptychography_simple_example
├── pyproject.toml                # Build system and dependency configuration
├── tests/                        # Tests for convergence of forward model
│   ├── test_solver_backward_*.py
│   ├── test_solver_dirichlet_*.py
│   ├── test_solver_impedance_*.py
│   └── test_solver_neumann_*.py
└── thickptypy/                   # Core Python package
    ├── forward_model/            # Forward modeling for wave propagation
    │   ├── boundary_conditions.py
    │   ├── initial_conditions.py
    │   ├── linear_system.py
    │   └── solver.py
    ├── reconstruction/           # Reconstruction algorithms
    │   └── least_squares.py
    ├── sample_space/             # Sample, object and scan path constructions
    │   ├── optical_objects.py
    │   └── sample_space.py
    └── utils/
        └── visualisations.py     # Visualisation of results
```

---

## Testing

Tests are located in the `tests/` directory and cover a wide range of solver configurations.

Run all tests:

```bash
poetry run pytest
```

To check specific modules (e.g., 2D Neumann BCs):

```bash
poetry run pytest tests/test_solver_neumann_2d.py
```

For convergence plots
```bash
poetry run pytest tests/test_solver_neumann_2d.py --plot
```

For coverage reports:

```bash
poetry run pytest --cov=thickptypy tests/
```

---

## Authors

* **Bryce Shirley** — [bryce.shirley@stfc.ac.uk](mailto:bryce.shirley@stfc.ac.uk)
* **Niall Bootland** — [niall.bootland@stfc.ac.uk](mailto:niall.bootland@stfc.ac.uk)

---
### DISCLAIMER
>  Phase Focus Limited of Sheffield, UK has an international portfolio of patents and pending applications which relate to ptychography. A current list is available at www.phasefocus.com/patents
Phase Focus grants non-commercial, royalty free licences of its patent rights for academic research use, for reconstruction of simulated data and for reconstruction of data obtained at synchrotrons at X-ray wavelengths. These licences can be applied for online by clicking on this link: www.phasefocus.com/licence


