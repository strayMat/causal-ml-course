# Usage

## Requirements

- TODO



# Installation

You can install Mleco via [pip](https://pip.pypa.io/):

```shell script
pip install mleco
```

# Using the project

- TODO

## Running the project

> 📝 **Note**
> All following commands are relative to the project root directory and assume
> `make` is installed.

You can run the project as follows:

### Locally via Poetry

Run:

```shell script
make provision-environment # Note: installs ALL dependencies!
poetry shell # Activate the project's virtual environment
jupyter notebook # Launch the Jupyter server
cli main # Run the project main entrypoint
```

> 📝 **Note**
> If you want to launch the jupyter notebooks directly, simply use `make jupyter-notebook`.



# Development

> 📝 **Note**
> For convenience, many of the below processes are abstracted away
> and encapsulated in single [Make](https://www.gnu.org/software/make/) targets.

> 🔥 **Tip**
> Invoking `make` without any arguments will display
> auto-generated documentation on available commands.

## Package and Dependencies Installation

Make sure you have Python 3.8+ and [poetry](https://python-poetry.org/)
installed and configured.

To install the package and all dev dependencies, run:

```shell script
make provision-environment
```

> 🔥 **Tip**
> Invoking the above without `poetry` installed will emit a
> helpful error message letting you know how you can install poetry.









## Documentation

```shell script
make docs-clean docs-html
```

> 📝 **Note**
> This command will generate html files in `docs/_build/html`.
> The home page is the `docs/_build/html/index.html` file.
