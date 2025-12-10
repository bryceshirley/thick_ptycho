# thick_ptycho: Thick Sample Ptychography Simulations

**thick_ptycho** is a Python project for simulating and reconstructing thick-sample 
ptychography. It supports various forward models including multislice and Paraxial
approximations, and provides tools for generating synthetic data and performing
reconstructions.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bryceshirley/thick_ptycho.git
   cd thick_ptycho
   ```

2. **Install dependencies (requires [uv to be installed](https://docs.astral.sh/uv/getting-started/installation/))**:

   ```bash
   uv sync --extra dev
   ```

## Installation with Petsc

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bryceshirley/thick_ptycho.git
   cd thick_ptycho
   ```

2. **Install Petsc and Petsc4py (Complex-versions) In Conda Env**

   ```bash
   conda create -n petsc_env -c conda-forge python=3.11 "petsc=*=complex*" "petsc4py=*=complex*" mpi4py compilers
   conda activate petsc_env
   ```

3. **Install thick_ptycho dependencies (requires [uv to be installed](https://docs.astral.sh/uv/getting-started/installation/))**:

   ```bash
   uv pip install -e ".[dev]"
   ```

---

## Generating Simulation Data

Use the `generate_data.py` script to create simulation data based on a configuration file.

```bash
uv run python generate_data.py gen_data_config.yaml
```

This will generate simulated ptychographic data and save it to `./results/sim/{timestamp}/`.

---

## Running Reconstructions

Use the `run_reconstruction.py` script to perform reconstructions using a configuration file.

```bash
uv run python run_reconstruction.py recon_config.yaml
```

This will perform the reconstruction and save results to `./results/recon/{timestamp}/`.

---

## Notebooks

Several Jupyter notebooks are provided in the `notebooks/` directory to demonstrate
various features and use cases of the package, including 2D and 3D simulations
and reconstructions using different solvers such as multislice and Paraxial.

---

## Testing

Tests are located in the `tests/` directory and cover a wide range of solver configurations.

```bash
uv run pytest
```

For coverage reports:

```bash
uv run pytest --cov=thick_ptycho
```

---

## Authors

* **Bryce Shirley** — [bryce.shirley@stfc.ac.uk](mailto:bryce.shirley@stfc.ac.uk)
* **Niall Bootland** — [niall.bootland@stfc.ac.uk](mailto:niall.bootland@stfc.ac.uk)

---
### DISCLAIMER
>  Phase Focus Limited of Sheffield, UK has an international portfolio of patents and pending applications which relate to ptychography. A current list is available at www.phasefocus.com/patents
Phase Focus grants non-commercial, royalty free licences of its patent rights for academic research use, for reconstruction of simulated data and for reconstruction of data obtained at synchrotrons at X-ray wavelengths. These licences can be applied for online by clicking on this link: www.phasefocus.com/licence


