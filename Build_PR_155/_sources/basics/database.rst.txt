.. _fvcc_Database:

Database
========


.. mermaid::

    classDiagram

        class Database{
            +CRUD()
            -std::unordermap&lt;std::string, Collection> data
        }

        class Collection{
            +CRUD()
            +query()
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


        
