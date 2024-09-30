EqnTerm
=======


The template `EqnTerm` represents a term in an equation and can be instantiated with different value types.
An `EqnTerm` is either explicit, implicit or temporal, and needs to be scalable by a scalar value or a further field.
The `EqnTerm` implementation uses Type Erasure (more details `[1] <https://medium.com/@gealleh/type-erasure-idiom-in-c-0d1cb4f61cf0>`_ `[2] <https://www.youtube.com/watch?v=4eeESJQk-mw>`_ `[3] <https://www.youtube.com/watch?v=qn6OqefuH08>`_) to achieve polymorphism without inheritance. Consequently, the class needs only to implement the interface which is used in the DSL and which is shown in the below example:


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

        auto constantScaledTerm = 2.0 * customTerm; // A constant scaling factor of 2 for the term.
        auto fieldScaledTerm = scalingField * customTerm; // scalingField is used to scale the term.

        // EqnTerm also supports a similar syntax as OpenFOAM
        auto multiScaledTerm = (scale + scale + scale + scale) * customTerm;

        // EqnTerm also supports the use of a lambda as scaling function to reduce the number of temporaries generated
        auto lambdaScaledTerm =
            (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return sF[i] + sF[i] + sF[i]  + sF[i]; }) * customTerm;

To add a user-defined `EqnTerm`, a new derived class must be created, inheriting from `EqnTermMixin`,
 and provide the definitions of the below virtual functions that are required for the `EqnTerm` interface:

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

The required scaling of the term is handle by the `scaleField` function, provided by `EqnTermMixin`. The `scaleField` function returns the 'ScalingField' class that is used to scale by fields and scalars.

.. code-block:: cpp

    template <typename ValueType>
    class ScalingField
    {

        // the span is only used if it is defined
        KOKKOS_INLINE_FUNCTION
        ValueType operator[](const size_t i) const { return useSpan ? values[i] * value : value; }

    }
