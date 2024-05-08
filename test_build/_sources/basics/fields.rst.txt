.. _api_fields:

Fields
======

The Field classes are the central elements for implementing a platform portable CFD framework. Fields are able to perform basic algebraic operations such as addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar. The Field classes store the data in a platform-independent way and the executor, which is used to dispatch the operations to the correct device. The Field classes are implemented in the ``Field.hpp`` header file and mainly store a pointer to the data, the size of the data, and the executor.

.. doxygenclass:: NeoFOAM::Field
    :members: size_, data_, exec_

To run a function on the GPU, the data and function need to be trivially copyable. This is not the case with the existing Field class, and it can be viewed as a wrapper around the data. To solve this issue, the field class has a public member function called ``field()`` that returns a span that can be used to access the data on the CPU and GPU.

.. doxygenclass:: NeoFOAM::Field
    :members: field

The following example shows how to use the field function to access the data of a field and set all the values to 1.0. However, the for loop is only executed on the single CPU core and not on the GPU.

.. code-block:: cpp

     NeoFOAM::CPUExecutor cpuExec {};
     NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, size);
     std::span<double> sA = a.field();
     // for loop
     for (int i = 0; i < sA.size(); i++)
     {
          sA[i] = 1.0;
     }

To run the for loop on the GPU is a bit more complicated and is based on the Kokkos library that simplifies the process and support different parallelization strategies such as OpenMP and different GPU vendors. The following example shows how to set all the values of a field to 1.0 on the GPU.

.. code-block:: cpp

     NeoFOAM::GPUExecutor gpuExec {};
     NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, size);
     std::span<double> sB = b.field();
     Kokkos::parallel_for(
          Kokkos::RangePolicy<gpuExec::exec>(0, sB.size()),
          KOKKOS_LAMBDA(const int i) { sB[i] = 1.0; }
     );

Kokkos requires the knowledge of where to run the code and the range of the loop. The range is defined by the size of the data and the executor. The KOKKOS_LAMBDA is required to mark the function so it is also compiled for the GPU. This approach however is not very user-friendly and requires the knowledge of the Kokkos library. To simplify the process, the Field class stores the executor and the field independent of the device can be set to 1.0 with the following code.

.. code-block:: cpp

     NeoFOAM::Field<NeoFOAM::scalar> c(gpuExec, size);
     NeoFOAM::fill(b, 10.0);

The fill function uses the std::visit function to call the correct function based on the executor as described in the previous section.

.. code-block:: cpp

    Operation op{};
    std::visit([&](const auto& exec)
               { op(exec); },
               exec);

Following other operations such as summation, min, max etc are listed below

.. code-block:: cpp

    // organize the FieldOperation so the can be easily shown here

The Field can now be used to compose more complex types such as the BoundaryFields and domainFields

.. doxygenclass:: NeoFOAM::BoundaryFields
    :members:
        value_
        refValue_
        valueFraction_
        refGrad_
        boundaryTypes_
        offset_
        nBoundaries_
        nBoundaryFaces_
