# tests/conftest.py
import multiprocessing as mp

def pytest_configure():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
