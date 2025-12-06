# thick_ptycho: Thick Sample Ptychography Simulations

**thick_ptycho** is a Python project for simulating and reconstructing thick-sample 
ptychography. It includes solvers for the forward 
model in 2D, and 3D using a finite difference scheme to solve the paraxial wave
equation, as well as reconstruction algorithms such as least squares
minimization of simplified thick-sample ptychography model.

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


