.. _first_kernel:

Implementing your first kernel
================================

This section explains how to implement a first kernel.
Here the ``gaussGreenDiv`` class serves as an example.
The class is implemented in the following files:

- ``include/NeoFOAM/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``src/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``test/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``

The ``gaussGreenDiv`` class represents the following term :math:`\int \nabla \cdot \phi dV` and is a particular implementation of the ``dsl::explicit::div`` operator.
Hence, in order to make the implementation selectable at runtime we let the ``GaussGreenDiv`` class derive from ``DivOperatorFactory::Register<GaussGreenDiv>`` and implement the static name function, (see also registerClass).

The actual implementation of the operator can be found in the ``gaussGreenDiv.cpp`` file.
The ``GaussGreenDiv::div`` member calls a free standing function ``computeDiv`` with the correct arguments.
In NeoFOAM this is a common pattern to use free standing functions since they are easier to test and communicate all dependencies via the function arguments.


.. math::

   \int \nabla \phi dV
