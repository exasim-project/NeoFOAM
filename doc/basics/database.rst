.. _fvcc_Database:

Database
========

NeoFOAM uses a document-based database to store the data. This general purpose database stores data in a collection of documents. Each document is pair of strings and values that are validate with a custom validator function as shown in the the following class diagram:

.. mermaid::

    classDiagram

        class Database{
            -std::unordermap&lt;std::string, Collection> data
        }

        class Collection{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class Document{
            +valdiate()
            -std::string id_;
            -std::function<bool(Dictionary)> validator;
            -Dictionary dict
        }

        Database "1" -- "n" Collection
        Collection "1" -- "n" Document

The general abstraction is that each object can be represented as Key-Value pairs similar to the ``Dictionary`` class. The following code snippet shows how to create a document and access its values:

.. sourcecode:: cpp

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}});
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.id().substr(0, 4) == "doc_");
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);
    };       

``NeoFOAM::Document`` mainly extendes the ``Dictionary`` class by and id and adds a validator function that can be used to validate the document. The following code snippet shows how to create a document with a custom validator function:

.. sourcecode:: cpp

    auto validator = [](const NeoFOAM::Dictionary& dict)
    { return dict.contains("key1") && dict.contains("key2"); };

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}}, validator);
    REQUIRE_NOTHROW(doc.validate());

These documents are stored in a collection that can be accessed by the database. The following code snippet shows how to create a collection and add a document to it:

.. sourcecode:: cpp

    NeoFOAM::Document doc;
    doc.insert("key1", std::string("value1"));
    auto doc1Id = collection1.insert(doc);


The collection can be accessed by the database. The following code snippet shows add and get a collection:

.. sourcecode:: cpp

    NeoFOAM::Database db;
    db.createCollection("collection1", "testCollection");
    auto& collection1 = db.getCollection("collection1");


The job of the developer is to create constraints on the data that is stored in the collection e.g. by providing free functions, validators to simplify the data access and ensure the data consistency:

.. sourcecode:: cpp

    class FieldDocument
    {
    public:

        std::string name;
        std::size_t timeIndex;
        std::size_t iterationIndex;
        std::int64_t subCycleIndex;
        std::any field; // stores any VolumeField or SurfaceField

        static Document create(FieldDocument fDoc); // also creates a document with a custom validator function

        Document doc();
    };

    // access field members
    std::size_t timeIndex(const Document& doc); // const access
    std::size_t& timeIndex(Document& doc); // non-const access


    // usage
    auto doc = fvcc::FieldDocument::create(
        {.name = "T",
            .timeIndex = 1,
            .iterationIndex = 2,
            .subCycleIndex = 3,
            .field = createVolumeField(mesh, "T")}
    );
    // insert
    std::string keyDoc1 = fieldCollection.insert(doc1);

    // modify the timeIndex
    fvcc::timeIndex(doc2) = 3;

The simplest way to additional functionality for a collection is by providing free functions that operate on the document as shown above. However, there are other approaches how to wrap a collection with additional functionality.