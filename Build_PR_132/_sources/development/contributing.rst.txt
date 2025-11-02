Contributing Guide
=================

Welcome to the FoamAdapter project! This guide will help you contribute effectively to both the C++ core and Python interface components.

Getting Started
---------------

Prerequisites for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Required Tools**

* **Git**: Version control
* **CMake**: 3.22+ for building C++ components
* **Compiler**: GCC 12+ or Clang 18+
* **Python**: 3.9+ for Python development
* **OpenFOAM**: 2406+ for testing and integration

**Development Dependencies**

.. code-block:: bash

    # Python development tools
    pip install black isort mypy pytest pytest-cov
    pip install pre-commit sphinx sphinx-autobuild
    
    # C++ development tools (Ubuntu/Debian)
    sudo apt install clang-format clang-tidy cppcheck doxygen

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Fork and Clone**

.. code-block:: bash

    # Fork the repository on GitHub, then:
    git clone https://github.com/YOUR_USERNAME/FoamAdapter.git
    cd FoamAdapter
    
    # Add upstream remote
    git remote add upstream https://github.com/exasim-project/FoamAdapter.git

**2. Create Development Branch**

.. code-block:: bash

    git checkout -b feat/your-feature-name
    # or
    git checkout -b fix/issue-description

**3. Set Up Pre-commit Hooks**

.. code-block:: bash

    pre-commit install
    pre-commit run --all-files  # Test the setup

Development Workflow
--------------------

Code Style Guidelines
~~~~~~~~~~~~~~~~~~~~~
