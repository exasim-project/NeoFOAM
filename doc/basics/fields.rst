.. _fvcc_fields:
Fields
======

Overview
^^^^^^^^

.. warning::
    The API of the classes probably will change in the future as currently parallelization is not supported.

NeoFOAM implements several field classes:

- ``Field<ValueType>`` the basic GPU capable container class supporting algebraic operations
- ``BoundaryFields<ValueType>`` A GPU friendly datastructure storing boundary data.
- ``DomainField<ValueType>`` The combination of an internal field and its corresponding boundary data.

Besides these container like field classes several finite volume specific field classes are implemented. The corresponding classes are:

- ``GeometricFieldMixin<ValueType>`` Mixin class combining a ``DomainField`` and the corresponding mesh.
- ``VolumeField<ValueType>`` Uses the GeometricFieldMixin and implements finite volume specific members, including the notion of concrete boundary condiditons
- ``SurfaceField<ValueType>`` The surface field equivalent to ``VolumeField``

The Field<ValueType> class
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Field class is the basic container class and is the central elements for implementing a platform portable CFD framework.
Fields should allow to perform basic algebraic operations such as binary operations like the addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar.

In the following, some implementation details of the field operations are detailed using the additions operator as an example.
The block of code below shows an example implementation of the addition operator.

.. code-block:: cpp

    [[nodiscard]] Field<T> operator+(const Field<T>& rhs)
    {
        Field<T> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }


Besides creating a temporary for the result it mainly calls the free standing ``add`` function which is implemented in ``include/NeoFOAM/field/fieldFreeFunctions.hpp``.
This, in turn, dispatches to the ``fieldBinaryOp`` function, passing the actual kernel as lambda.
The ``fieldBinaryOp``  is implemented using our parallelFor implementations which ultimately dispatch to the ``Kokkos::parallel_for`` function, see `Kokkos documentation  <https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_ for more details.

.. code-block:: cpp

    template<typename ValueType>
    void add(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
    {
      detail::fieldBinaryOp(
          a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
      );
    }

A simplified version of the ``parallelFor`` function is shown below.

.. code-block:: cpp
    template<typename Executor, parallelForKernel Kernel>
    void parallelFor(
        [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel
    )
    {
        auto [start, end] = range;
        if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
        {
        ...
        }
        else
        {
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelFor",
                Kokkos::RangePolicy<runOn>(start, end),
                KOKKOS_LAMBDA(const size_t i) { kernel(i); }
            );
        }
    }

The code snippet highlights another important aspect, the executor.
The executor defines the ``Kokkos::RangePolicy``, see  `Kokkos Programming Model  <https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/ProgrammingModel.md>`_.
Besides defining the RangePolicy, the executor also holds functions for allocating and deallocationg memory.
See our `documentation  <https://exasim-project.com/NeoFOAM/latest/basics/executor.html>`_ for more details on the executor model.

Further `Details  <https://exasim-project.com/NeoFOAM/latest/doxygen/html/classNeoFOAM_1_1Field.html>`_.

Cell Centered Fields
^^^^^^^^^^^^^^^^^^^^

The ``VolumeField`` stores the field values at cell centers and along boundaries, providing essential data for constructing the Domain Specific Language (DSL). This functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and write to the ``internalField`` and ``boundaryFields`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates the field's boundary conditions, which are specified at construction. It does not hold the data but rather modifies the ``DomainField`` or ``BoundaryField`` container.

Functionally, ``fvccVolField`` parallels several OpenFOAM classes such as ``volScalarField``, ``volVectorField``, and ``volTensorField``. Note: "fvcc" represents "Finite Volume Cell Centered".

.. doxygenclass:: NeoFOAM::finiteVolume::cellCentred::VolumeField
    :members:
        field_,
        fvccVolField,
        internalField,
        boundaryField,
        correctBoundaryConditions

Face Centered fields
^^^^^^^^^^^^^^^^^^^^

The ``SurfaceField`` class stores the field values interpreted as face centers values.  Additionally, it stores boundaries for the corresponding boundary conditions. This provides essential data for constructing the DSL. The functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and to the ``internalField`` and ``boundaryField`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates field's boundary conditions, which are specified at construction. It does not hold the data, but modify the ``DomainField`` or ``BoundaryField`` container.

Functionally, fvccVolField parallels several OpenFOAM classes such as ``surfaceScalarField``, ``surfaceVectorField``, and ``surfaceTensorField``.
However, the ``internalField`` also contains the boundary values, so no branches (if) are required when iterating over all cell faces. Thus the size of the ``internalField`` in NeoFOAM differs from that of OpenFOAM.

.. doxygenclass:: NeoFOAM::finiteVolume::cellCentred::SurfaceField
    :members:
        field_,
        fvccSurfaceField,
        internalField,
        boundaryField,
        correctBoundaryConditions


.. _api_fields:
