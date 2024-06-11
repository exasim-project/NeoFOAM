.. _basics_registerclass:

Derived class discovery at compile time
=======================================

The ``RegisterClass`` class is a template class that allows to register derived class into a map to the can be instaniated from the base class similar to the OpenFOAM runtime mechanism. The classes at registered at compile time via static member initialization. Additional explanation can be found in the following links: `Ref1 <https://stackoverflow.com/questions/52354538/derived-class-discovery-at-compile-time>`_ `Ref2 <https://stackoverflow.com/questions/10332725/how-to-automatically-register-a-class-on-creation>`_ .

This approach allows to create a plugin architecture where now derived class can be loaded at runtime and simplifies the runtime selection of a specific class. Now the derived classes can be created based on the name provided:

.. code-block:: cpp

    std::unique_ptr<baseClass> derivedClass =
        baseClass::create("Name of derivedClass", "Additional arguments",...);


.. doxygenclass:: NeoFOAM::RegisterClass
   :members:

The ``RegisterClassManager`` class is a template class that manages the registration of the derived classes and stores the map of the registered classes. The map associates a function with a string that is used to create the derived class.

.. doxygenclass:: NeoFOAM::RegisterClassManager
   :members:

Usage
^^^^^

The following example shows how to use the ``RegisterClass`` and ``RegisterClassManager`` to automatically register derived classes. (details see ``test_RegisterClass.cpp``).

To implement our design, we must first define a createFunction. This function will return a ``std::unique_ptr`` to a base class, taking an argument list that specifies the class interface. Following this, we'll establish a ``RegisterClassManager``. This manager will handle the registration and storage of all classes that have been registered.

.. code-block:: cpp

    // forward declaration so we can use it to define the create function and the class manager
    class TestBaseClass;

    // define the create function use to instantiate the derived classes
    using createFunc = std::function<std::unique_ptr<TestBaseClass>(std::string, double)>;

    // define the class manager to register the classes
    using TestBaseClassManager = NeoFOAM::RegisterClassManager<TestBaseClass, createFunc>;

We derive the base class from the ``TestBaseClassManager`` and define the interface that the derived classes must implement. We also define a template alias ``TestBaseClassReg`` that will be used to register the derived classes. The ``create`` function will be used to instantiate the derived classes based on the name provided. The ``registerClass`` function will be used to register the derived classes. The derived classes will implement the interface functions ``testString`` and ``testValue``.

.. code-block:: cpp

    // interface that needs to be implemented by the derived classes
    class TestBaseClass : public TestBaseClassManager
    {
    public:


        template<typename derivedClass>
        using TestBaseClassReg = NeoFOAM::RegisterClass<derivedClass, TestBaseClass, createFunc>;

        // create function to instantiate the derived classes
        static std::unique_ptr<TestBaseClass>
        create(const std::string& name, std::string testString, double testValue)
        {
            try
            {
                auto func = classMap.at(name);
                return func(testString, testValue);
            }
            catch (const std::out_of_range& e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
                return nullptr;
            }
        }


        template<typename derivedClass>
        bool registerClass()
        {
            return TestBaseClassReg<derivedClass>::reg;
        }

        virtual ~TestBaseClass() = default;

        // interface that needs to be implemented by the derived classes
        virtual std::string testString() = 0;

        virtual double testValue() = 0;

        // ...

    };

The derived classes will be registered using the ``registerClass`` function. The derived classes will implement the interface functions ``testString`` and ``testValue``. The ``create`` function will be used to instantiate the derived classes based on the name provided in the ``name`` function.

.. code-block:: cpp

    class TestDerivedClass : public TestBaseClass
    {

    public:

        // the constructor is used to register the class
        TestDerivedClass(std::string name, double test)
            : TestBaseClass(), testString_(name), testValue_(test)
        {
            registerClass<TestDerivedClass>(); // register the class
        }

        // must be implemented by the derived classes to register the class
        static std::unique_ptr<TestBaseClass> create(std::string name, double test)
        {
            return std::make_unique<TestDerivedClass>(name, test);
        }

        // must be implemented by the derived classes to register the class
        static std::string name() { return "TestDerivedClass"; }

        virtual std::string testString() override { return testString_; };

        virtual double testValue() override { return testValue_; };

    private:

        std::string testString_;
        double testValue_;
    };


After the classes have been defined, we can use the ``create`` function to instantiate the derived classes based on the name provided.

.. code-block:: cpp

    std::unique_ptr<TestBaseClass> testDerived =
        TestBaseClass::create("TestDerivedClass", "FirstDerived", 1.0);
