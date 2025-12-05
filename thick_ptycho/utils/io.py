import logging
import os
import subprocess
from datetime import datetime

import yaml


def setup_log(
    results_dir: str = "results",
    log_file_name: str = "log.txt",
    use_logging: bool = True,
    verbose: bool = False,
):
    """
    Return log(msg, level) that writes to console and results_dir/log_file_name.
    Independent of any class.
    """
    # unique logger name per log file so multiple calls don’t share handlers
    safe_name = os.path.join(results_dir or "results", log_file_name)
    logger = logging.getLogger(f"log.{safe_name}")

    if use_logging and not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False  # prevent duplicate logs

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File handler
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            logfile = os.path.join(results_dir, log_file_name)
            file_handler = logging.FileHandler(logfile, mode="w")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    def log(msg, level=logging.INFO, flush=False):
        if use_logging:
            logger.log(level, msg)
        if verbose:
            print(msg, flush=flush)

    return log


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str:
    """Return current git commit hash, or 'unknown' if not available."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def results_dir_name(dir_name: str = "") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", dir_name, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
