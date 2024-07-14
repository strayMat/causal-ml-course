#! /usr/bin/env python
import logging

import click
from dotenv import load_dotenv

from mleco.constants import LOG_LEVEL
from mleco.utils import hello

# see `.env` for requisite environment variables
load_dotenv()


logging.basicConfig(
    level=logging.getLevelName(LOG_LEVEL),
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)


@click.group()
@click.version_option(package_name="mleco")
def cli():
    pass


@cli.command()
def main() -> None:
    """mleco Main entrypoint"""
    click.secho(hello(), fg="green")


if __name__ == "__main__":
    cli()
