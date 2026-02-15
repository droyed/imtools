"""Demo configuration module for loading config.yaml settings."""

import os
import yaml


def get_output_dir(demo_name=None):
    """
    Get the output directory for demo files.

    Parameters
    ----------
    demo_name : str, optional
        If provided, creates a subdirectory with this name within the base output dir

    Returns
    -------
    str
        The output directory path
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_dir = config.get('output_dir', 'output_demos')
    if demo_name:
        return os.path.join(base_dir, demo_name)
    return base_dir
