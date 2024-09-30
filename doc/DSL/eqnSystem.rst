EqnSystem
---------


The `EqnSystem` template class in NeoFOAM holds, manages, builds and solves the expressed/programmed equation and its core responsibilities lie in the answering of the following questions:

    - How to discretize the spatial terms?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to integrate the system in time?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to solve the system?
        - In OpenFOAM this information is provided in **fvSolution**

The main difference between NeoFOAM and OpenFOAM is that the DSL is evaluated lazily, i.e. evaluation is not performed on construction by default.
Since, the evaluation is not tied to the construction but rather delayde, other numerical integration strategies (e.g. RK methods or even substepping within an the equation) are possible.

To implement lazy evaluation, the `EqnSystem` stores 3 vectors:

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

Thus, an `EqnSystem` consists of multiple `EqnTerms` which are either explicit, implicit, or temporal.
Consequently, addition, subtraction, and scaling with a field needs to be handled by the `EqnTerm`.
