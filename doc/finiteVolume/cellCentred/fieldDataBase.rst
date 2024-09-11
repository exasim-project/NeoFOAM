.. _fvcc_fieldDataBase:

fieldDataBase
=============

OpenFOAM mangages its opjects over a object registry. The object registry can be viewed as a database the regsiter types that are derived from regIOobject. The database is accessible used inside Time and mesh to register the objects. This approach tries to decouple the mesh from the implemnetation so it applies to the single reponsible principle. 

The same applies to the geometricField that also provides the access to the oldTime and therefore also needs to be mangage the time state.

However, the design of the DSL for convient use requires the acessbility of the oldTime from the current objects and the access to additional fields to compute boundary conditions.

Consequently, a new fieldDataBase needs to be introduced that stores all register field via the FieldEntityManager. The FieldEntityManager to store the oldTime(s), cacheGradient and storeIteration and also stores with fields need to be written to disk.

The FieldEntityManager is created by the FieldDataBase and stores a reference to the database. If a field is created with the FieldEntityManager it is stored in the database. 
BoundaryConditions have access to the main field and can therefore can access the database and retrieve arbitrary fields.

An example where this is required is the heatFlux of the boundary conitions, the face normal gradient requires the knowledge of the heat conductivity.

This design has simplifies the usage of the fields and makes it simpler to access the requires fields. However, the global nature of the database requires a specific initialization order. In our case the thermodynamic model needs to be initialized before the creation of the boundary conditions.

This dependency of the initialization order is a drawback of the design. However, the design is more flexible and allows to access the fields in a more convient way and from personal experience the initialization order is not a big issue.

However, global variables introduce hidden state and be generally be avoided. Consequently, the access to the database should be read only from in case of global access.

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
            std::vector~FieldEntityManager~ fem
            +CRUD()
            +createFieldEntityManager() FieldEntityManager

        }
        class FieldEntityManager{
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
            std::optional~FieldEntityManager&~ fem
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
        FieldEntityManager o-- operations
        FieldDataBase o-- FieldEntityManager
        GeometricField o-- FieldEntityManager
        FieldEntityManager o-- FieldComponent
        FieldComponent *-- FieldCategory
