.. _first_kernel:

Implementing a your first kernel
================================

This section explains how to implement a first kernel.
Here the ``gaussGreenDiv`` class serves as an example.
The class is implemented in the following files:
- ``include/NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``src/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``test/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``

The ``gaussGreenDiv`` class represents the following term
.. math::

   \int \nabla \phi dV
