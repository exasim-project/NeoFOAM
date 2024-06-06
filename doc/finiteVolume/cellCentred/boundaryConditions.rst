.. _fvcc_BC:

.. warning::
    The API of the classes probably will change in the future. It is a first draft implementation to iterate over the design.


Boundary Conditions
===================

In contrast to OpenFOAM the boundary conditions do not store the underlying  data but instead modify the data provided by ``DomainField`` in ``fvccVolField`` and ``fvccSurfaceField``.
The interface for the face centered and volField centered data is similar but not identical. The 'correctBoundaryConditions' update the data of the domainField.

.. doxygenclass:: NeoFOAM::fvccBoundaryField
    :members:
        correctBoundaryConditions



.. doxygenclass:: NeoFOAM::fvccSurfaceBoundaryField
    :members:
        correctBoundaryConditions

The above class are the baseClass of the specific implementation that provide the actual boundary conditions.

Boundary Conditions for VolumeField's
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The boundary condition modify the data of the ``boundaryField`` with the visitor pattern

.. code-block:: c++

    void fvccScalarFixedValueBoundaryField::correctBoundaryConditions(
        BoundaryFields<scalar>& bfield, const Field<scalar>& internalField
    )
    {
        fixedValueBCKernel kernel_(mesh_, patchID_, start_, end_, uniformValue_);
        std::visit([&](const auto& exec) { kernel_(exec, bfield, internalField); }, bfield.exec());
    }

The logic is implemented in the kernel classes:

.. code-block:: c++

    void fixedValueBCKernel::operator()(
        const GPUExecutor& exec, BoundaryFields<scalar>& bField, const Field<scalar>& internalField
    )
    {
        using executor = typename GPUExecutor::exec;
        auto s_value = bField.value().field();
        auto s_refValue = bField.refValue().field();
        scalar uniformValue = uniformValue_;
        Kokkos::parallel_for(
            "fvccScalarFixedValueBoundaryField",
            Kokkos::RangePolicy<executor>(start_, end_),
            KOKKOS_LAMBDA(const int i) {
                s_value[i] = uniformValue;
                s_refValue[i] = uniformValue;
            }
        );
    }

As the BoundaryFields class stores all data in a continuous array the boundary condition must only update the data in the range of the boundary specified by the `start_` and `end_` index. In the above simple boundary condition, the kernel only sets the values to a uniform/fixed value. The ``value`` field stores the current value of the boundary condition that is used by the explicit operators and the ``refValue`` stores the value of the boundary condition that is used by the implicit operators.

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
