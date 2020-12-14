"""cli.py"""

import os
import logging
import click
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
from fake_faces.processing import cropface, align_all, falsify
from fake_faces.training import train, exp
from fake_faces.labeling import label
from fake_faces.testing import make_metrics, make_fairness_metrics, learning_curves


@click.group()
def cli():
    """The fake-faces command line interface."""


def main():
    """Program entry point"""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    cli.add_command(cropface)
    cli.add_command(align_all)
    cli.add_command(falsify)
    cli.add_command(train)
    cli.add_command(label)
    cli.add_command(exp)
    cli.add_command(make_metrics)
    cli.add_command(make_fairness_metrics)
    cli.add_command(learning_curves)
    cli()


if __name__ == "__main__":
    main()
