"""The atlas-densities command line launcher"""

import logging

import click

from atlas_densities.app import cell_densities, combination, mtype_densities
from atlas_densities.version import VERSION

L = logging.getLogger(__name__)


def cli():
    """The main CLI entry point"""
    logging.basicConfig(level=logging.INFO)
    group = {
        "cell-densities": cell_densities.app,
        "mtype-densities": mtype_densities.app,
        "combination": combination.app,
    }
    help_str = "The main CLI entry point."
    app = click.Group("atlas_densities", group, help=help_str)
    app = click.version_option(VERSION)(app)
    app()
