"""cli.py"""

import os
import logging
import click
from fake_faces.processing import cropface


@click.group()
def cli():
    """The fake-faces command line interface."""


def main():
    """Program entry point"""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    cli.add_command(cropface)
    cli()


if __name__ == "__main__":
    main()
