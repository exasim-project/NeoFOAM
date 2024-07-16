Contributing
^^^^^^^^^^^^

Contributions are highly welcomed. Here is some information getting you started.

NeoFOAM Code Style Guide
""""""""""""""""""""""""

To simplify coding and code reviews you should adhere to the following code style. However, most
of NeoFOAMs code style guide is checked via automated tools.
For example, formatting and non-format related code style is enforced via clang-format and clang-tidy.
Corresponding configuration files are `.clang-format <https://exasim-project.com/NeoFOAM/.clang-format>`_
`.clang-tidy <https://exasim-project.com/NeoFOAM/.clang-tidy>`_.
Furthermore, adequate licensing of the source code is checked via the reuse linter and typos checks for obvious spelling issues.
Thus, this style guide doesn't list all stylistic rules explicitly but rather gives advise for ambiguous situations and mentions the rational for some decissions.

 * Use `camelCase` for functions and members and capitalized `CamelCase` for classes
 * Use descriptive template parameter names.
   For example prefer `template <class ValueType>` over `template <class T>` if the type can be a float or double.
 * Use `[[nodiscard]]` except for getters.
   To indicate that a function simply returns a value instead of performing expensive computations `[[nodiscard]]` can be omitted.
 * Use `final` except for abstract classes.
   NeoFOAMs aims at having a rather flat inheritance hierarchy and composition is generally preferred.
   To avoid unintended inheritance the using `final` is advised.
 * Avoid storing references as data members in classes.
   An exception might be a `const Unstructured& mesh` as it will most likely outlive any other objects.
 * Place in-out parameter at the end of function argument lists.
   E.g. `foo(const Type& in, Type& out)` instead of `foo(Type& out,const Type& in)`.
 * use US spelling

Folder structurce and file names:

 * use `camelCase` for files and folder names
 * the location of implementation and header files should be consistent.
   Ie. a file `src/model/a.cpp` should have a header file in `include/NeoFOAM/model/a.hpp` and corresponding test implementation in `test/model/a.cpp`
 * File and folder names should not be redundant.


Building the Documentation
""""""""""""""""""""""""""

NeoFOAMs documentation can be found `main <https://exasim-project.com/NeoFOAM/>`_  and `doxygen <https://exasim-project.com/NeoFOAM/doxygen/html/>`_ documentation can be found online. However, if you want to build the documentation locally you can do so, by executing the following steps.
First, make sure that Sphinx and Doxygen are installed on your system. Second, execute the following commands:

   .. code-block:: bash

    cmake -B build -DNEOFOAM_BUILD_DOC=ON # configure the build
    cmake --build build --target sphinx # build the documentation
    # or
    sphinx-build -b html ./doc ./docs_build


The documentation will be built in the `docs_build` directory and can be viewed by opening the `index.html` file in a web browser.

   .. code-block:: bash

    firefox docs_build/index.html

Alternatively, the documentation can be built by just adding the `-DNEOFOAM_BUILD_DOC=ON` to the configuration step of the build process and then building the documentation using the `sphinx` target.
