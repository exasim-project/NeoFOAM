Installation Guide
==================

This guide covers the installation and setup of the FoamAdapter Python module, which provides a Python interface to OpenFOAM solvers and utilities through the pybFoam backend.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

Before installing FoamAdapter Python module, ensure your system meets the following requirements:

* **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
* **Python**: 3.9 or higher
* **OpenFOAM**: Version 2406 or later
* **CMake**: 3.22 or higher
* **Compiler**: GCC 12+ or Clang 18+
* **CUDA**: 12.1+ (optional, for GPU acceleration)

Python Dependencies
~~~~~~~~~~~~~~~~~~~

The FoamAdapter Python module requires the following Python packages:

.. code-block:: text

    pybFoam>=1.0.0
    pydantic>=2.0.0
    typer>=0.9.0
    matplotlib>=3.6.0
    networkx>=2.8.0

Installation Methods
--------------------

Development Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or if you want the latest features:

1. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/exasim-project/FoamAdapter.git
       cd FoamAdapter

2. **Set up Python environment**:

   .. code-block:: bash

       # Using conda (recommended)
       conda create -n foamadapter python=3.11
       conda activate foamadapter
       
       # Or using venv
       python -m venv foamadapter-env
       source foamadapter-env/bin/activate

3. **Install dependencies**:

   .. code-block:: bash

       pip install -r requirements.txt

4. **Build the C++ backend**:

   .. code-block:: bash

       cmake --preset production
       cmake --build --preset production

5. **Install the Python module**:

   .. code-block:: bash

       pip install -e ./src

Environment Setup
-----------------

OpenFOAM Environment
~~~~~~~~~~~~~~~~~~~~

Ensure OpenFOAM is properly sourced in your environment:

.. code-block:: bash

    # Add to your ~/.bashrc or source before using FoamAdapter
    source /opt/openfoam2406/etc/bashrc

You can verify your OpenFOAM installation:

.. code-block:: bash

    which blockMesh
    which icoFoam

pybFoam Configuration
~~~~~~~~~~~~~~~~~~~~~

FoamAdapter depends on pybFoam for the Python-OpenFOAM interface. Ensure pybFoam is properly configured:

.. code-block:: bash

    python -c "import pybFoam; print(pybFoam.__version__)"

If pybFoam is not available, follow the pybFoam installation guide.

GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration with CUDA:

1. **Install CUDA 12.1+**:
   
   Follow the NVIDIA CUDA installation guide for your distribution.

2. **Verify CUDA installation**:

   .. code-block:: bash

       nvcc --version
       nvidia-smi

3. **Rebuild with CUDA support**:

   .. code-block:: bash

       cmake --preset production-cuda
       cmake --build --preset production-cuda

Verification
------------

Test Basic Installation
~~~~~~~~~~~~~~~~~~~~~~~

Verify the installation by running a simple test:

.. code-block:: bash

    python -c "import foamadapter; print('FoamAdapter imported successfully')"

Test CLI Interface
~~~~~~~~~~~~~~~~~~

Check if the command-line interface is working:

.. code-block:: bash

    python -m foamadapter --help

You should see the FoamAdapter CLI help message.

Test Solver Access
~~~~~~~~~~~~~~~~~~

Verify that solvers are accessible:

.. code-block:: bash

    python -m foamadapter solver --help

Run Example Case
~~~~~~~~~~~~~~~~

Test with a simple case (assuming you have the tutorials):

.. code-block:: bash

    cd tutorials/cavity
    python -m foamadapter solver icofoam --help

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'pybFoam'**

Solution: Install pybFoam or ensure it's in your Python path.

.. code-block:: bash

    pip install pybFoam

**OpenFOAM not found**

Solution: Ensure OpenFOAM is properly sourced:

.. code-block:: bash

    source /path/to/openfoam/etc/bashrc

**CUDA compilation errors**

Solution: Ensure CUDA is properly installed and compatible with your compiler:

.. code-block:: bash

    # Check CUDA compatibility
    nvcc --version
    gcc --version

**Permission errors during installation**

Solution: Use virtual environments or install with user flag:

.. code-block:: bash

    pip install --user -e ./src

Performance Issues
~~~~~~~~~~~~~~~~~~

If you experience slow performance:

1. **Check OpenMP settings**:

   .. code-block:: bash

       export OMP_NUM_THREADS=4  # Adjust based on your CPU cores

2. **Verify mesh quality** in your test cases

3. **Monitor memory usage** during large simulations

Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

1. **Check the GitHub Issues**: `FoamAdapter Issues <https://github.com/exasim-project/FoamAdapter/issues>`_
2. **Review the documentation**: :doc:`../index`
3. **Check  logs** for underlying solver issues
4. **Verify pybFoam installation** and compatibility

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Useful environment variables for debugging:

.. code-block:: bash

    # Enable verbose output
    export FOAM_ADAPTER_VERBOSE=1
    
    # Set custom case directory
    export FOAM_CASE_DIR=/path/to/case
    
    # OpenFOAM debugging
    export FOAM_ABORT=1

Next Steps
----------

After successful installation:

1. Read the :doc:`quickstart` guide
2. Explore the :doc:`cli` documentation  
3. Try the example cases in the tutorials directory
4. Review the solver-specific documentation

For advanced usage and development, see the full API documentation.