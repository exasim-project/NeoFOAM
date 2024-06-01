.. _fvcc_fields:

Fields (FVCC)
=============

.. warning::
    The API of the classes probably will change in the future as currently parallelization is not supported.

Cell centered fields
^^^^^^^^^^^^^^^^^^^^

The ``fvccVolField`` stores the field values at cell centers and along boundaries, providing essential data for constructing the DSL (Domain Specific Language). This functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and to the ``internalField`` and ``boundaryField`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates field's boundary conditions, which are specified at construction. The boundaryConditions do not hold the data but modify the ``DomainField`` or ``BoundaryField`` container.

Functionally, fvccVolField parallels several OpenFOAM classes such as ``volScalarField``, ``volVectorField``, and ``volTensorField``.

.. doxygenclass:: NeoFOAM::fvccVolField
    :members:
        field_,
        fvccVolField,
        internalField,
        boundaryField,
        correctBoundaryConditions

Face centered fields
^^^^^^^^^^^^^^^^^^^^

The ``fvccSurfaceField`` stores the field values at face centers and along boundaries, providing essential data for constructing the DSL (Domain Specific Language). This functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and to the ``internalField`` and ``boundaryField`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates field's boundary conditions, which are specified at construction. The boundaryConditions do not hold the data but modify the ``DomainField`` or ``BoundaryField`` container.

Functionally, fvccVolField parallels several OpenFOAM classes such as ``surfaceScalarField``, ``surfaceVectorField``, and ``surfaceTensorField``.
However, the internalField also contains the boundary values, so no branches (if) are required when loop over all cell faces. So the size of the internalField in the NeoFOAM and OpenFOAM is different.

.. doxygenclass:: NeoFOAM::fvccSurfaceField
    :members:
        field_,
        fvccSurfaceField,
        internalField,
        boundaryField,
        correctBoundaryConditions
