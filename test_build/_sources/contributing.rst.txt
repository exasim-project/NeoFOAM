Contributing
^^^^^^^^^^^^

Building the Documentation
""""""""""""""""""""""""""

The documentation is built using sphinx and doxygen that are required to be installed on your system. The documentation can be built using the following command:

   .. code-block:: bash 

    cmake -B build -DNEOFOAM_BUILD_DOC=ON # configure the build
    cmake --build build --target sphinx # build the documentation

The documentation will be built in the `build/doc/sphinx` directory and can be viewed by opening the `index.html` file in a web browser.

   .. code-block:: bash 

    firefox build/docs/sphinx/index.html

Alternatively, the documentation can be built by just adding the `-DNEOFOAM_BUILD_DOC=ON` to the configutation step of the build process and then building the documentation using the `sphinx` target.