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