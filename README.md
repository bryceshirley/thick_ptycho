# thick_ptycho: Thick Sample Ptychography Simulations and Reconstructions

**thick_ptycho** is a Python project for simulating and reconstructing thick-sample 
ptychography. It supports various forward models including multislice and Paraxial
approximations, and provides tools for generating synthetic data and performing
reconstructions.


---
[![Documentation](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://stfc.github.io/thick_ptycho/) ![Test Status](https://github.com/bryceshirley/thick_ptycho/actions/workflows/tests.yml/badge.svg) ![Linting Status](https://github.com/bryceshirley/thick_ptycho/actions/workflows/linter.yml/badge.svg)

## Notebooks Tutorials

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


