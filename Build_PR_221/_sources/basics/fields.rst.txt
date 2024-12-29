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
The Field classes are the central elements for implementing a platform portable CFD framework. Fields should allow to perform basic algebraic operations such as binary operations like the addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar.

In the following, we will explain the implementation details of the field operations using the additions operator as an example. The block of code below shows an example implementation of the addition operator.

.. code-block:: cpp

    [[nodiscard]] Field<T> operator+(const Field<T>& rhs)
    {
        Field<T> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }


Besides creating a temporary for the result it mainly calls the free standing ``add`` function which is implemented in ``FieldOperations.hpp``. This, in turn, dispatches to the ``addOp`` functor, that holds the actual addition kernels. In the case of addition this is implemented as a  ``Kokkos::parallel_for`` function, see `Kokkos documentation  <https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_ for more details.

.. code-block:: cpp

   using executor = typename Executor::exec;
   auto a_f = a.field();
   auto b_f = b.field();
   Kokkos::parallel_for(
      Kokkos::RangePolicy<executor>(0, a_f.size()),
      KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = a_f[i] + b_f[i]; }
   );

The code snippet also highlights another important aspect, the executor. The executor, here defines the ``Kokkos::RangePolicy``, see  `Kokkos Programming Model  <https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/ProgrammingModel.md>`_. Besides defining the RangePolicy, the executor also holds functions for allocating and deallocationg memory. A full example of using NeoFOAMs fields with a GPU executor could be implemented as

.. code-block:: cpp

    NeoFOAM::GPUExecutor GPUExec {};
    NeoFOAM::Field<NeoFOAM::scalar> GPUa(GPUExec, N);
    NeoFOAM::fill(GPUa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUb(GPUExec, N);
    NeoFOAM::fill(GPUb, 2.0);
    auto GPUc = GPUa + GPUb;

Interface
^^^^^^^^^

.. doxygenclass:: NeoFOAM::Field
    :members:

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
