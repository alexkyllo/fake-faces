"""cli.py"""

import os
import logging
import click


@click.group()
def cli():
    """The fake-faces command line interface."""


def main():
    """Program entry point"""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    cli()


if __name__ == "__main__":
    main()
