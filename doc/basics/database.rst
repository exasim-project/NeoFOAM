.. _basics_Database:

Database
========

OpenFOAM stores objects in the ``objectRegistry`` and they can be accessed by their name and type.
In modern software development, best practices emphasize modularity, testability, and loosely coupled components.
While the ``objectRegistry`` of OpenFOAM offers convenience and simple approach for managing objects, its inherent design conflicts with these principles by:

- Encouraging reliance on a central container.
- Making the codebase less flexible for testing, refactoring, or adopting new architectural patterns.
- Introducing potential performance bottlenecks and debugging challenges.

In contrast, NeoFOAM adopts a document-based database approach, where data is stored as a collection of documents.
Each document consists of a pair of strings and values, validated using a custom validator function to ensure data integrity.
The database is not tightly coupled to the ``fvMesh`` or ``Time`` classes, making it more flexible and easier to test.

The following class diagram illustrates the relationships between the ``Database``, ``Collection``, and ``Document`` components:

.. mermaid::

    classDiagram

        class Database{
            -std::unordermap&lt;std::string, Collection> data
        }

        class CollectionA{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class CollectionB{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class CollectionC["..."]{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class DocumentA{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        class DocumentB{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        class DocumentC{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        class DocumentD{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        class DocumentE["..."]{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        class DocumentF["..."]{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }


        Database <-- CollectionA
        Database <-- CollectionB
        Database <-- CollectionC

        CollectionA <-- DocumentA
        CollectionA <-- DocumentB
        CollectionA <-- DocumentE
        CollectionB <-- DocumentC
        CollectionB <-- DocumentD
        CollectionB <-- DocumentF

A database can have 0 to N collection and each collection can have 0 to N documents. 
At the lowest level is the ``Document`` class, which is a Dictionary with an ID and a validator function to ensure data integrity.
This container similar to a python dictionary enables the storage of key-value pairs and therefore can be used to store any type of data.
The following code snippet shows how to create a document and access its values:

.. sourcecode:: cpp

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}});
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.id().substr(0, 4) == "doc_");
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);
    };       

``NeoFOAM::Document`` mainly extendes the ``Dictionary`` and offers the possibility to validate the data stored in the document. The following code snippet shows how to create a document with a custom validator function:

.. sourcecode:: cpp

    auto validator = [](const NeoFOAM::Dictionary& dict)
    { return dict.contains("key1") && dict.contains("key2"); };

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}}, validator);
    REQUIRE_NOTHROW(doc.validate());

As stated earlier, the Document are than stored in the collection that are stored in the database as shown in the class diagram above.
This design allows the storage of the desired data but the usage would be inconvient as only document wouldn't have any additional functionality.
Therefore, the developer can create a custom collection that provides additional functionality to the document that will be discussed in the last section.
This is done in the FieldCollection class that is used to store the field data in the database.



FieldCollection
---------------

The ``FieldCollection`` stores all fields and provides additional functionality which is stored in a document.
The ``FieldDocument`` stores the field itself in ``std::any`` and additional information like the time index, iteration index, and subcycle index.
The following code snippet shows how to create a ``FieldDocument``:

.. sourcecode:: cpp

    template<class geoField>
    FieldDocument(
        const geoField& field,
        std::size_t timeIndex,
        std::size_t iterationIndex,
        std::int64_t subCycleIndex
    )
        : doc_(
              Document(
                  {{"name", field.name},
                   {"timeIndex", timeIndex},
                   {"iterationIndex", iterationIndex},
                   {"subCycleIndex", subCycleIndex},
                   {"field", field}}
              ),
              validateFieldDoc
          )
    {}

The ``FieldDocument`` stores its data in a ``Document`` and provide the approiate getters and setters for the data.
The user will most likely not directly create FieldDocument but use the ``FieldCollection`` to register Field.

.. sourcecode:: cpp

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "newTestFieldCollection");

    fvcc::VolumeField<NeoFOAM::scalar>& T =
        fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(CreateField {
            .name = "T", .mesh = mesh, .timeIndex = 1, .iterationIndex = 1, .subCycleIndex = 1
        });

The ``FieldCollection`` has a static instance methods that returns the instance of the ``FieldCollection`` or creates a new one if it doesn't exist.
registerField method is used to register fields and expects a CreateFunction that returns a FieldDocument and expects a database as an argument.
The createFunction could look as followed:

.. sourcecode:: cpp

    struct CreateField
    {
        std::string name;
        NeoFOAM::UnstructuredMesh mesh;
        std::size_t timeIndex = 0;
        std::size_t iterationIndex = 0;
        std::int64_t subCycleIndex = 0;
        NeoFOAM::Document operator()(NeoFOAM::Database& db)
        {
            std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
            for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
            {
                NeoFOAM::Dictionary dict;
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", 2.0);
                bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
            }
            NeoFOAM::Field internalField =
                NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0);
            fvcc::VolumeField<NeoFOAM::scalar> vf(
                mesh.exec(), name, mesh, internalField, bcs, db, "", ""
            );
            return NeoFOAM::Document(
                {{"name", vf.name},
                {"timeIndex", timeIndex},
                {"iterationIndex", iterationIndex},
                {"subCycleIndex", subCycleIndex},
                {"field", vf}},
                fvcc::validateFieldDoc
            );
        }
    };

This design can be easilty extended to create and read fields.
The ``FieldCollection`` allows us to access and find fields by their name, time index, iteration index, or subcycle index.

.. sourcecode:: cpp

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "newTestFieldCollection");
    
    auto resName = fieldCollection.find([](const NeoFOAM::Document& doc)
                                    { return doc.get<std::string>("name") == "T"; });

    REQUIRE(resName.size() == 1);
    const auto& fieldDoc = fieldCollection.fieldDoc(resName[0]);
    const auto& constVolField = fieldDoc.field<fvcc::VolumeField<NeoFOAM::scalar>>();

Adding a new collection and documents to the database
-----------------------------------------------------

The collection is a type erased interface that allows the simple extension by types and only requires that the correct function are implemented in the class.
The collection than can return access to the custom documents that are stored in the collection.
The developer can create a custom document that extends the ``Document`` and provides additional functionality as shown below:


.. sourcecode:: cpp

    class CustomDocument
    {
    public:

        CustomDocument(
            const std::string& name,
            const double& testValue
        )
            : doc_(
                NeoFOAM::Document(
                    {
                        {"name", name},
                        {"testValue", testValue}
                    }
                    , validateCustomDoc
                )
            )
        {}

        std::string& name() { return doc_.get<std::string>("name"); }

        const std::string& name() const { return doc_.get<std::string>("name"); }

        double testValue() const { return doc_.get<double>("testValue"); }

        double& testValue() { return doc_.get<double>("testValue"); }
        

        NeoFOAM::Document& doc() { return doc_; }

        const NeoFOAM::Document& doc() const { return doc_; }

        std::string id() const { return doc_.id(); }

        static std::string typeName() { return "CustomDocument"; }

    private:

        NeoFOAM::Document doc_;
    };

The own datastructure can be stored in the documents and the member functions can be defined to access the data. 
Enabling a simple extension of the database and the collection. The newly created Document can be stored in a custom collection as shown below:


.. sourcecode:: cpp

    class CustomCollection : public NeoFOAM::CollectionMixin<CustomDocument>
    {
    public:

        CustomCollection(NeoFOAM::Database& db, std::string name)
            : NeoFOAM::CollectionMixin<CustomDocument>(db, name)
        {}

        bool contains(const std::string& id) const { return docs_.contains(id); }

        bool insert(const CustomDocument& cc)
        {
            std::string id = cc.id();
            if (contains(id))
            {
                return false;
            }
            docs_.emplace(id, cc);
            return true;
        }

        static CustomCollection& instance(NeoFOAM::Database& db, std::string name)
        {
            NeoFOAM::Collection& col = db.insert(name, CustomCollection(db, name));
            return col.as<CustomCollection>();
        }
    };

The CollectionMixin is a template class that simplifies the implementation of the collection and offer the possibility to store user-defined documents as shown below.


