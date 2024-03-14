.. _api_compatibility:

Compatibility With OpenFOAM
===========================

Overview
^^^^^^^^

NeoFOAM aims to achieve a restricted level of compatibility with existing OpenFOAM source code (API: 2312).
Users of OpenFOAM should be able to replace OpenFOAM headers and libraries with NeoFOAM counterparts and then be able to compile and run their applications.
Additional code changes on the user side should be managed through provided tools.

ABI Compatibility
^^^^^^^^^^^^^^^^^

NeoFOAM will **not** provide ABI compatible libraries to any OpenFOAM version.
ABI compatible libraries severely restrict our NeoFOAM design.
It would require NeoFOAM to provide the same types as OpenFOAM which are required to have the same size, layout, and alignment as the OpenFOAM versions (see e.g. https://en.wikipedia.org/wiki/Application_binary_interface#Description ).
This size requirement alone is not possible due to the new `Executor` object that some classes will have to contain.

API Compatibility
^^^^^^^^^^^^^^^^^

NeoFOAM will provide **partial** API compatibility to specific OpenFOAM versions.
Only selected parts of the OpenFOAM library are targeted by NeoFOAM to be replaced.
Other parts are not directly targeted, but might still require changes that are propagated from the main NeoFOAM interface.
These changes should be automated as much as possible.

.. _compatible_libraries:
The NeoFOAM API replacement targets the libraries:

* `src/OpenFOAM`
* `src/finiteVolume`

For these libraries, NeoFOAM will provide most of the types and functions that are defined in OpenFOAM.
A clear exception are all types, functions, etc. defined in the detail namespace `Foam::Detail`.
The NeoFOAM types will not have the exact same API as in OpenFOAM.
This would require especially the same type hierarchy as in OpenFOAM, which would limit the NeoFOAM design.
Instead, the NeoFOAM types will provide a compatibility similar to duck typing.
For example, the NeoFOAM `Field` type will provide the same functionality as in OpenFOAM, but it will not have the same base types (`Foam::FieldBase` and `Foam::List<T>`).

.. _compatibility_realization:
Realization
^^^^^^^^^^^

The compatibility to OpenFOAM will be measured by the following two examples:

#. The incompressible solver `simpleFoam`.
#. The RAS turbolence model `kEpsilon`.

To make these example use NeoFOAM the following changes will be required:

* Replace OpenFOAM headers from the libraries mentioned :ref:`here _compatible_libraries`  with NeoFOAM headers
* Link against NeoFOAM libraries instead of OpenFOAM's
* Apply automatic code changes to fix any incompatibility.

With that, the examples should compile and user should be able to run their code with NeoFOAM as the backend.

Minimal Compatibility
^^^^^^^^^^^^^^^^^^^^^

To accelerate the development of NeoFOAM the libraries mentioned :ref:`here _compatible_libraries` are not fully replaced by NeoFOAM all at once.
Instead, first the API of NeoFOAM will cover the parts that are necessary for the `simpleFOAM` solver.
Then it will be extended to the `kEpsilon` model.
Afterwards NeoFOAM should be in a position to handle simple applications and further OpenFOAM parts can be replaced as necessary.
