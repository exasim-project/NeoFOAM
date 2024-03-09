Contributing
^^^^^^^^^^^^

Building the Documentation
""""""""""""""""""""""""""

NeoFOAMs documentation can be found [online](https://exasim-project.com/NeoFOAM/). However, if you wan't to build the documentation locally you can do so, by executing the following steps. 
First make sure that  sphinx and doxygen are installed on your system. Second execute the following commands:

   .. code-block:: bash 

    cmake -B build -DNEOFOAM_BUILD_DOC=ON # configure the build
    cmake --build build --target sphinx # build the documentation

The documentation will be built in the `build/doc/sphinx` directory and can be viewed by opening the `index.html` file in a web browser.

   .. code-block:: bash 

    firefox build/docs/sphinx/index.html

Alternatively, the documentation can be built by just adding the `-DNEOFOAM_BUILD_DOC=ON` to the configutation step of the build process and then building the documentation using the `sphinx` target.