.. _algorithms:

Parallel Algorithms
===================

Parallel algorithms are a basic building block for implementing advanced kernels.
To simplify the implementation of the advanced kernels we provide a set of standard algorithms.
These can be found in the following files:

- ``include/NeoFOAM/core/parallelAlgorithms.hpp``
- ``test/core/parallelAlgorithms.cpp``

Currently, the following algorithms are provided:

- ``parallelFor``
- ``parallelReduce``

The following code block shows the implementation of a parallelFor for fields

.. code-block:: cpp

    template<typename Executor, typename ValueType, parallelForFieldKernel<ValueType> Kernel>
    void parallelFor([[maybe_unused]] const Executor& exec, Field<ValueType>& field, Kernel kernel)
    {
        auto span = field.span();
        if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
        {
            for (size_t i = 0; i < field.size(); i++)
            {
                span[i] = kernel(i);
            }
        }
        else
        {
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelFor",
                Kokkos::RangePolicy<runOn>(0, field.size()),
                KOKKOS_LAMBDA(const size_t i) { span[i] = kernel(i); }
            );
        }
    }


based on the Executor type a kernel function is either run directly within a for loop or dispatched to ``Kokkos::parallel_for`` for all non ``SerialExecutors``.
The executor type determines the ``Kokkos::RangePolicy<runOn>``, additionally we name the kernel as ``parallelFor`` to improve identifiability in profiling tools like nsys.
Finally, a ``KOKKOS_LAMBDA`` is dispatched assigning the result of the given kernel function to the span of the field.
Here the span holds data pointers to the device data and defines the begin and end pointer of the data.



To learn more on how to use the algorithms it is recommended to check the corresponding `unit test <https://github.com/exasim-project/NeoFOAM/blob/main/test/core/parallelAlgorithms.cpp>`_.
