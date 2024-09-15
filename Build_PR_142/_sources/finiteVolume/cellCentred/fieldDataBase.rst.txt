.. _fvcc_fieldDataBase:

fieldDataBase
=============

OpenFOAM manages its objects through an object registry. The object registry can be viewed as a database that registers types derived from regIOobject. This database is accessible and used within Time and Mesh to register objects. This approach aims to decouple the mesh from the implementation, adhering to the single responsibility principle.

The same principle applies to geometricField, which also provides access to oldTime and, therefore, needs to manage the time state.

However, the design of the DSL for convenient use requires accessibility to oldTime from the current objects, as well as access to additional fields to compute boundary conditions.

As a result, a new FieldDatabase must be introduced to store all registered fields via the SolutionFields. The SolutionFields handles the storage of oldTime(s), caches gradients, and manages iterations, while also determining which fields need to be written to disk.

The SolutionFields is created by the FieldDatabase and maintains a reference to the database. When a field is created with the SolutionFields, it is stored in the database. Boundary conditions have access to the main field and, therefore, can access the database to retrieve any necessary fields.

One example of where this is required is the heatFlux in boundary conditions. The face normal gradient requires knowledge of the heat conductivity.

This design simplifies the usage of fields and makes it easier to access the required fields. However, the global nature of the database introduces a specific initialization order. In our case, the thermodynamic model must be initialized before the boundary conditions are created.

This dependency on initialization order is a drawback of the design. Nonetheless, the design is more flexible and allows for more convenient access to fields. From personal experience, the initialization order has not been a significant issue.

However, global variables introduce hidden states and should generally be avoided. As a result, global access to the database should be read-only.

The general design is shown in the following class diagram:


.. mermaid::

    classDiagram
        class operations{
            <<namespace>>
            +oldTime()
            +cacheGradient()
            +storeIteration()
            +getFieldDataBase()
        }
        class FieldDataBase{
            std::vector~SolutionFields~ fem
            +CRUD()
            +createSolutionFields() SolutionFields

        }
        class SolutionFields{
            -const FieldDataBase& fdb
            -std::strind name
            -GeometricField~Type~ field
            -map~intFieldComponent~ fieldDB
            +createField(): GeometricField
            +CRUD()
        }
        class GeometricField~Type~{
            DomainField~Type~ field
            correctBoundaryCondition()
            std::optional~SolutionFields&~ fem
        }
        class FieldCategory{
            <<enumeration>>
            Iteration
            oldTime
            cacheField
        }
        class FieldComponent~Type~{
            +std::shared_ptr~Type~ field
            +std::size_t timeIndex
            +std::size_t iterationIndex
            +bool write
            +FieldCategory category
        }

        FieldDataBase o-- operations
        SolutionFields o-- operations
        FieldDataBase o-- SolutionFields
        GeometricField o-- SolutionFields
        SolutionFields o-- FieldComponent
        FieldComponent *-- FieldCategory
