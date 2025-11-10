Installation
============

This guide explains how to install the FoamAdapter Python package and set up the development environment using standard Python tools and Poe.

Requirements
------------
- Python 3.9 or newer


Installing the Package
---------------------

To install the package and all development and documentation dependencies, run:

.. code-block:: bash

    pip install -e .[all]

- Use ``.[dev]`` for development dependencies only.
- Use ``.[docs]`` for documentation dependencies only.

Using uv (Optional)
------------------

For faster and more isolated Python workflows, you can use [uv](https://github.com/astral-sh/uv) as a drop-in replacement for pip. To get started with uv:

1. Install uv:

   .. code-block:: bash

       pip install uv

2. Install dependencies with uv:

   .. code-block:: bash

       uv pip install -e .[all]

3. Run commands in the uv environment:

   .. code-block:: bash

       uv run poe test

You can use uv in place of pip and python for most commands. All other instructions in this documentation work with or without uv.

Usage with Poe the Poet
-----------------------

[Poe the Poet](https://github.com/nat-n/poethepoet) is included in the development dependencies and provides convenient task automation.

To see available tasks, run:

.. code-block:: bash

        poe --help

Available Poe Tasks
-------------------

- **test**: Run the test suite using pytest.

    .. code-block:: bash

            poe test

- **lint**: Run ruff to check code style and linting issues in the source and test directories (excluding src/NeoN).

    .. code-block:: bash

            poe lint

- **format**: Format the code using ruff (for formatting) in the source and test directories (excluding src/NeoN).

    .. code-block:: bash

            poe format

- **build_docs**: Build the Sphinx documentation (HTML) into the doc/_build directory.

    .. code-block:: bash

            poe build_docs

- **view_docs**: Open the built HTML documentation in your default web browser.

    .. code-block:: bash

            poe view_docs
