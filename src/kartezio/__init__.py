"""
Kartezio - Cartesian Genetic Programming framework for computer vision.

A Python framework for evolutionary computation applied to biomedical image segmentation
and analysis, featuring component-based architecture and extensible primitive libraries.
"""

import logging

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1a3"
__author__ = "Kevin Cortacero"
__email__ = "kevin.cortacero@gmail.com"

# Set up basic logging configuration if no handlers are configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
