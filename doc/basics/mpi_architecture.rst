.. _basic_mpi_architecture:

MPI Architecture
================

Background
----------

Virtually all large scientific and engineering codes require are too large to be practically usable with a single 'computing element'. Most problems are broken down into smaller parts and distributed accross several 'computing elements'. In many cases these 'computing elements' do not share the same memory addressing space, or distributed memory architecture. This introduces the need to communicate data and synchronise the solution proceedure accross these 'computing elements'. ``NeoFOAM``, link many others uses `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>` to achieve this, with each 'computing element' referred to as an ``MPI rank`` (``rank``). Note: for shared memory architecture (including GPU computing) see `executor.rst <executor.rst>`. Fundementally 2 problems need to be solved:

1. How should the global computation be partitioned and distributed accross the ``Ranks``.
2. What data needs be communicated between which ``ranks``.

Finally, for a scalable solution it is desired that the comminication is 'masked', meaning as much as possible the overhead and cost of the communication is done in parallel to the main computation and therefore not the bottleneck. Broadly this is achieved by non-blocking communication between ``ranks`` in conjunction with the minimisation of the frequency and size of the communication. Since the comminication architecture informs the partitioning, we will discuss this first.

Communication
-------------

MPI Wrapping
^^^^^^^^^^^^

``MPI`` is brought into ``NeoFOAM`` in the ``operators.hpp`` file. The purpose of this file is to wrap ``MPI`` functions such that they work more seemlessly with ``NeoFOAM`` data types, and also supplying typical defaults. For example the ``MPI_Allreduce`` function is wrapped:

.. code-block:: c++

    template<typename valueType>
    void reduceAllScalar(valueType* value, const ReduceOp op, MPI_Comm comm)
    {
        MPI_Allreduce(
            MPI_IN_PLACE, reinterpret_cast<void*>(value), 1, getType<valueType>(), getOp(op), comm
        );
    }

such that scalar reduction size is handled automatically. Ontop of the wrapped operators is the ``MPI`` enviroment. Here the ``environment.hpp`` files is included, which contains two classes ``MPIInit`` and ``MPIEnvironment``.  The former is a simple RAII class that initialises and finalises (in the destructor) the ``MPI`` environment, thus a typicaly program using ``NeoFOAM`` would start by  alling the MPIInit constructor.

.. code-block:: c++
    #include "NeoFOAM/core/mpi/environment.hpp"

    int main(int argc, char** argv)
    {
        NeoFOAM::mpi::MPIInit mpi(argc, argv);

        // main solver

        // MPI_finalize is called in the destructor
    }

.. note::
    Since we support ``OpenMP`` (through ``Kokkos``), we need to enusre threading support is available when we start.

Once started the ``MPIEnvironment`` class is used to manage the ``MPI_communicator``, ``MPI_rank`` and ``MPI_size``. Since the class will populate the communicator with ``MPI_COMM_WORLD`` and initialise the rank and size the class can be constructed from anywhere in the code.

.. note::
    In the future it is intended to use this class to manage the ``MPI Communicators`` (allow for splitting of the communicator), at which point the above will no longer apply and the instance will need to be parsed.

Global Communication
^^^^^^^^^^^^^^^^^^^^

With the above in place global communication (i.e. communication between all ``ranks`` on a ``MPI_Communicator``) can be achieved by using the enviroment and operators.

.. code-block:: c++
    #include "NeoFOAM/core/mpi/environment.hpp"
    #include "NeoFOAM/core/mpi/operators.hpp"

    int main(int argc, char** argv)
    {
        NeoFOAM::mpi::MPIInit mpi(argc, argv);
        NeoFOAM::mpi::MPIEnvironment mpiEnv;

        double value = 1.0;
        NeoFOAM::mpi::reduceAllScalar(&value, NeoFOAM::mpi::ReduceOp::SUM, mpiEnv.com());

        if(mpiEnv.rank() == 0)
            std::cout<<"Value "<<value<<std::endl; // result is number of ranks.
    }

Point-to-Point Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section focuses on the approach for two ranks to communicate with each other, specifically using non-blocking communication for field data synchronization. To begin, the reader is reminded of 'communication terminology': simplex, half-duplex, and full-duplex. Where simplex communication is one-way, from sender to receiver. Half-duplex allows two-way communication but only one direction at a time. Full-duplex enables two-way communication simultaneously in both directions.

To facilitate communication between two ranks, a half-duplex buffer is introduced, namely the ``HalfDuplexCommBuffer``, which is responsible for non-blocking sending/receiving data to/from different ranks and into member data buffers. To generalise the buffer for different data types ``type-punning`` is used, as such the actual data which is transferred is always of type ``char``. Further since memory allocation is relatively expensive the buffer is never sized down. Finally, the buffers lay memory out continously than therefore it is required to have some map between the buffer position and the original data container (being communicated) position. This is part of the partitioning problem, and not the responsibility of the buffer.

.. note::
    The ``HalfDuplexCommBuffer`` duplex buffer has some guard rails in to esnure once communication has started various operations are no-longer possible until it is finished.

To acheive full-duplex communication two half-duplex buffers are combined, to form the ``FullDuplexCommBuffer``. The process for two way communication is then broken down into the following steps:

1. Initalise the communication, using a name and data type. This obtains the buffer as a used resource.
2. Load the buffer with data to send.
3. Start the communication.
4. Do other work to mask the communication.
5. Wait for the communication to finish.
6. Unload the buffer with the received data.
7. Finalise the communication, releasing the buffer.

The full communication between two ranks is thus given below:

.. code-block:: c++
    #include <unordered_map>
    #include <vector>
    #include "NeoFOAM/core/mpi/environment.hpp"
    #include "NeoFOAM/core/mpi/operators.hpp"
    #include "NeoFOAM/core/mpi/comm_buffer.hpp"

    int main(int argc, char** argv)
    {
        NeoFOAM::mpi::MPIInit mpi(argc, argv);
        NeoFOAM::mpi::MPIEnvironment mpiEnv;

        // create the buffers
        std::vector<std::size_t> sendSize;
        std::vector<std::size_t> receiveSize;
        std::vector<double> allData = {1.0, 2.0, 3.0}; // the local data (could be a field or similar)
        std::unordered_map<std::size_t, std::size_t> sendMap;
        std::unordered_map<std::size_t, std::size_t> receiveMap;

        // ...
        // populate above data
        // ...

        NeoFOAM::mpi::FullDuplexCommBuffer buffer(mpiEnv, sendSize, receiveSize);

        // Obtain the buffer.
        buffer.initComm<double>("test_communication");

        // load the send buffer
        auto sendBuffer = buffer.getSendBuffer<double>(); // span returned.
        sendBuffer[0] = allData[sendMap[0]];

        // start the non-blocking communication
        buffer.startComm();

        // ...
        // do other work
        // ...

        // wait for the communication to finish
        buffer.waitComplete();

        // unload the recieve buffer
        auto receiveBuffer = buffer.getReceiveBuffer<double>(); // span returned.
        allData[receiveMap[0]] = receiveBuffer[0];

        // finalise the communication, releasing the buffer
        buffer.finaliseComm();
    }


.. note::
    The copying to and from the buffers does introduce an overhead, which could later be removed by using 'inplace' communication. This remains an open point.

TODO: where to send -> number of buffers is different to comm path ways

Partitioning
------------

The purpose of partitioning is to divide the global computation into smaller parts that can be solved in parallel, and essentially to distribute the computation accross the ``ranks``. One the boundary

Currently there is no formal partitioning system in ``NeoFOAM``, however it is assumed that all communication is done on the ``MPI World`` communicator. This is to be updated in the future, together with dynamic load balancing.



Future Work
-----------

1. Allow ``MPI Communicators`` to be split, allowing for more complex partitioning of the computation.
2. Mesh partitioning
3. Implement dynamic load balancing.
