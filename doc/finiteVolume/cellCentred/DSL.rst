Domain Specific Language
========================

Domain Specific Language (DSL) dramatically simplifies the process of writing and solving equations. Engineers can express equations in a concise and readable form, closely resembling their mathematical representation and no or little knowledge of the numerical schemes and implementation is required. This approach allows engineers to focus on the physics of the problem rather than the numerical implementation and helps in reducing the time and effort required to develop and maintain complex simulations.

The Navier-Stokes equations can be expressed in the DSL as follows in OpenFOAM as follows:

.. code-block:: cpp

    fvVectorMatrix UEqn
    (
        fvm::ddt(U)
        + fvm::div(phi, U)
        - fvm::laplacian(nu, U)
    );

    solve(UEqn == -fvc::grad(p));

To solve the continuity equation in OpenFOAM with the PISO or SIMPLE algorithm, the VectorMatrix, UEqn, needs to provide the diagonal and off-diagonal terms of the matrix.

.. code-block:: cpp

    volScalarField rAU(1.0/UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

This approach is readable and easy to understand for engineers familiar with OpenFOAM. However, it has several limitations due to its implementation in OpenFOAM:

    - the solution system is always a sparse matrix
    - the sparse matrix is always a LDU matrix that is not supported by external linear solvers
    - LDU matrix only supports finite volume cell-centred discretisation

NeoFOAM DSL tries to address these issues by providing:

    - lazy evaluation evaluation of the equations
    - a more modular design
    - a standard matrix format for support of external linear solvers

The use of standard matrix format combined with lazy evaluation allows for the use of external libraries to integrate PDEs in time and space. The equation system can be passed to **sundials** and be integrated by **RK methods** and **BDF methods** on heterogeneous architectures.

The NeoFOAM DSL is designed as drop in replacement for OpenFOAM DSL and the adoption should be possible with minimal effort and the same equation from above should read:

.. code-block:: cpp

    dsl::EqnSystem<NeoFOAM::scalar> UEqn
    (
        fvcc::impOp::ddt(U)
        + fvcc::impOp::div(phi, U)
        - fvcc::impOp::laplacian(nu, U)
    )

    solve(UEqn == -fvcc::expOp::grad(p));


The majority of the work is done in the solve step: assemble the system and solve the system. After the system is solved or assembled, it allows needs to be provide access to the linear system for the SIMPLE and PISO algorithms.


EqnSystem
---------


The `EqnSystem` template class in NeoFOAM represents holds, manage, builds and solves the DSL and is the core part that orchestrates the following questions:

    - How to discretize the eqnterms?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to integrate the system in time?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to solve the system?
        - In OpenFOAM this information is provided in **fvSolution**

The main difference between OpenFOAM and NeoFOAM is that the DSL in NeoFOAM is evaluated lazily. Therefore, the evaluation is no longer tied to a sparse matrix enabling other numerical integration strategies (RK method) or even substepping can be integrated inside the equation.

To evaluate the terms lazily, `EqnSystem` stores 3 vectors:

.. mermaid::

    classDiagram
        class EqnTerm {

            +explicitOperation(...)
            +implicitOperation(...)
        }
        class DivEqnTerm {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class TemporalEqnTerm {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class Others["..."] {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class EqnSystem {
            +temporalTerms_: vector~EqnTerm~
            +implicitTerms_: vector~EqnTerm~
            +explicitTerms_: vector~EqnTerm~
        }
        EqnTerm <|-- DivEqnTerm
        EqnTerm <|-- TemporalEqnTerm
        EqnTerm <|-- Others
        EqnSystem <|-- EqnTerm

So, the general assumption is that a EqnSystem consists of multiple EqnTerms that are either explicit, implicit or temporal. Consequently, plus, minus or scaling with a field needs to be handled by the EqnTerm.


EqnTerm
-------


`EqnTerm` represents a term in an equation. It is a template class that can be instantiated with different value types. An `EqnTerm` can be explicit, implicit or temporal and needs be scalable by a scalar value or a field. `EqnTerm` is implemented as Type Erasure (more details `[1] <https://medium.com/@gealleh/type-erasure-idiom-in-c-0d1cb4f61cf0>`_ `[2] <https://www.youtube.com/watch?v=4eeESJQk-mw>`_ `[3] <https://www.youtube.com/watch?v=qn6OqefuH08>`_). So, the `EqnTerm` class provides a common interface for classes without inheritance. Consequently, the classes only needs to implement the interface and can be used in the DSL as shown in the example:


Example:
    .. code-block:: cpp

        NeoFOAM::DSL::EqnTerm<NeoFOAM::scalar> divTerm =
            Divergence(NeoFOAM::DSL::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, ...);

        NeoFOAM::DSL::EqnTerm<NeoFOAM::scalar> ddtTerm =
            TimeTerm(NeoFOAM::DSL::EqnTerm<NeoFOAM::scalar>::Type::Temporal, exec, ..);


To fit the specification of the EqnSystem (storage in a vector), the EqnTerm needs to be able to be scaled:

.. code-block:: cpp

        NeoFOAM::Field<NeoFOAM::scalar> scalingField(exec, nCells, 2.0);
        auto sF = scalingField.span();

        dsl::EqnTerm<NeoFOAM::scalar> customTerm =
            CustomTerm(dsl::EqnTerm<NeoFOAM::scalar>::Type::Explicit, exec, nCells, 1.0);

        auto scaledTerm = 2.0 * customTerm; // 2.0 is the scaling factor
        scaledTerm = -1.0 * customTerm; // -1.0 is the scaling factor
        scaledTerm = scalingField * customTerm; // scalingField is the scaling field

        // EqnTerm also supports a similar syntax as OpenFOAM
        auto scaledTerm = (scale + scale + scale + scale) * customTerm;

        // EqnTerm also supports the use of a lambda as scaling function to reduce the number of temporaries generated
        scaledTerm =
            (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return sF[i] + sF[i] + sF[i]  + sF[i]; }) * customTerm;

The simplest approach to add a custom EqnTerm, is to derive a class from the EqnTermMixin and implement the required functions. A class that want to use the EqnTerm interface needs to provide the following functions:

    - build: build the term
    - explicitOperation: perform the explicit operation
    - implicitOperation: perform the implicit operation
    - display: display the term
    - getType: get the type of the term
    - exec: get the executor
    - nCells: get the number of cells
    - volumeField: get the volume field

An example is given below:

.. code-block:: cpp

    class CustomEqnTerm : public dsl::EqnTermMixin<NeoFOAM::scalar>
    {

    public:

        // constructors ..
        NeoFOAM::scalar read(const NeoFOAM::Input& input)
        {
            // ..
        }

        void build(const NeoFOAM::Input& input)
        {
            value = read(input);
            termEvaluated = true;
        }

        std::string display() const { return "Laplacian"; }

        void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source)
        {
            NeoFOAM::scalar setValue = value;
            // scaleField is defined in EqnTermMixin
            // and accounts for the scaling of the terms
            // and considers scaling by fields and scalars
            auto scale = scaleField();
            auto sourceField = source.span();
            NeoFOAM::parallelFor(
                source.exec(),
                {0, source.size()},
                KOKKOS_LAMBDA(const size_t i) { sourceField[i] += scale[i] * setValue; }
            );
        }

        // other helper functions
        dsl::EqnTerm<NeoFOAM::scalar>::Type getType() const { return termType_; }

        const NeoFOAM::Executor& exec() const { return exec_; }

        std::size_t nCells() const { return nCells_; }

        fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

        dsl::EqnTerm<NeoFOAM::scalar>::Type termType_;


        const NeoFOAM::Executor exec_;
        std::size_t nCells_;
        NeoFOAM::scalar value = 1.0;
    };

The required scaling of the term is handle by the `scaleField` function that is provided by the EqnTermMixin. The `scaleField` function returns 'ScalingField' class that considers scaling by fields and scalars.

.. code-block:: cpp

    template <typename ValueType>
    class ScalingField
    {

        // the span is only used if it is defined
        KOKKOS_INLINE_FUNCTION
        ValueType operator[](const size_t i) const { return useSpan ? values[i] * value : value; }

    }
