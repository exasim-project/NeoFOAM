.. _fvcc_BC:

Boundary Conditions
===================

In contrast to OpenFOAM the boundary conditions do not store the data teh underlaying data but instead modify the data provided by ````DomainField```` in ````fvccVolField``` and ``fvccSurfaceField``.
The interface for the face centered and volField centered data is similar but not identical. The 'correctBoundaryConditions' update the data of the domainField.

.. doxygenclass:: NeoFOAM::fvccBoundaryField
    :members:
        correctBoundaryConditions



.. doxygenclass:: NeoFOAM::fvccSurfaceBoundaryField
    :members:
        correctBoundaryConditions

The above class are the baseClass of the specific implementation that provide the actual boundary conditions.

BC for volField
^^^^^^^^^^^^^^^

Currently the following boundary conditions are implemented for volField for scalar:

- ``fvccScalarCalculatedBoundaryField``
- ``fvccScalarEmptyBoundaryField``
- ``fvccScalarFixedValueBoundaryField``
- ``fvccScalarZeroGradientBoundaryField``

Currently the following boundary conditions are implemented for volField for Vector:

- ``fvccVectorCalculatedBoundaryField``
- ``fvccVectorEmptyBoundaryField``
- ``fvccVectorFixedValueBoundaryField``
- ``fvccVectorZeroGradientBoundaryField``

BC for surfaceField
^^^^^^^^^^^^^^^^^^^

Currently the following boundary conditions are implemented for volField for scalar:

- ``fvccSurfaceScalarCalculatedBoundaryField``
- ``fvccSurfaceScalarEmptyBoundaryField``
