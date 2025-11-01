Automated Test Integration
==========================

This page shows how to include actual test files in your documentation.

Including Test Files
--------------------

You can include entire test files in your documentation:

.. admonition:: Framework NBI Dataclasses Test (Click to expand)
   :class: toggle

    .. literalinclude:: ../test/framework/test_nbi_dataclasses.py
        :language: python
        :caption: Framework NBI Dataclasses Test
        :linenos:

Include Specific Tests
----------------------

.. literalinclude:: ../test/framework/test_nbi_dataclasses.py
   :language: python
   :pyobject: test_injection
   :caption: Just the test_injection function

Running Specific Tests
----------------------

You can also include and run specific test functions:

.. testcode::

    # Import the test module
    import sys
    import os
    sys.path.insert(0, os.path.abspath('../test'))
    
    from framework.test_nbi_dataclasses import test_injection
    
    # Run the test
    print("Running test_injection...")
    test_injection()
    print("Test passed successfully!")

.. testoutput::

    Running test_injection...
      Inside function: a = 'a', b = 1.0
    Test passed successfully!

Test Documentation with Results
-------------------------------

Here's how to show test results in documentation:

.. testcode::

    from framework.test_nbi_dataclasses import test_runner
    
    print("Running complete solver test...")
    test_runner()
    print("Solver test completed successfully!")

.. testoutput::

    Running complete solver test...
    Solver test completed successfully!

Test Coverage Integration
-------------------------

To include test coverage in your documentation, you can use the ``coverage`` extension
and create reports that are always up-to-date with your actual test suite.

Continuous Documentation Testing
---------------------------------

To ensure your documentation stays in sync with your tests:

1. **Auto-run tests during docs build**: The ``doctest`` extension will automatically
   run any ``.. testcode::`` blocks when you build the documentation.

2. **Include actual test files**: Using ``.. literalinclude::`` ensures the documentation
   shows the actual current test code.

3. **CI Integration**: Set up your CI to fail documentation builds if tests fail.

Example CI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # In your .github/workflows or similar
    - name: Test Documentation
      run: |
        cd doc
        make doctest
        
    - name: Build Documentation  
      run: |
        cd doc
        make html