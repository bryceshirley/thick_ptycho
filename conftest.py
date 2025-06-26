def pytest_addoption(parser):
    """Add a command-line option to control plotting."""
    parser.addoption(
        "--plot", action="store_true", default=False, help="Enable plotting of convergence plots"
    )
    parser.addoption(
        "--plot_error", action="store_true", default=False, help="Enable plotting of error"
    )