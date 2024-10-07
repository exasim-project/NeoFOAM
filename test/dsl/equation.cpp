// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "common.hpp"
#include "NeoFOAM/DSL/equation.hpp"

// class Laplacian
// {

// public:

//     std::string display() const { return "Laplacian"; }

//     void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
//     {
//         auto sourceField = source.span();
//         NeoFOAM::parallelFor(
//             source.exec(),
//             {0, source.size()},
//             KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1.0 * scale; }
//         );
//     }

//     dsl::EqnTerm::Type getType() const { return termType_; }

//     const NeoFOAM::Executor& exec() const { return exec_; }

//     const std::size_t nCells() const { return nCells_; }

//     fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

//     dsl::EqnTerm::Type termType_;

//     const NeoFOAM::Executor exec_;
//     const std::size_t nCells_;
// };

// class Divergence
// {

// public:

//     std::string display() const { return "Divergence"; }

//     void explicitOperation(NeoFOAM::Field<NeoFOAM::scalar>& source, NeoFOAM::scalar scale)
//     {
//         auto sourceField = source.span();
//         NeoFOAM::parallelFor(
//             source.exec(),
//             {0, source.size()},
//             KOKKOS_LAMBDA(const size_t i) { sourceField[i] += 1.0 * scale; }
//         );
//     }

//     dsl::EqnTerm::Type getType() const { return termType_; }

//     const NeoFOAM::Executor& exec() const { return exec_; }

//     const std::size_t nCells() const { return nCells_; }

//     fvcc::VolumeField<NeoFOAM::scalar>* volumeField() { return nullptr; }

//     dsl::EqnTerm::Type termType_;

//     const NeoFOAM::Executor exec_;
//     const std::size_t nCells_;
// };

using Equation = NeoFOAM::DSL::Equation;

TEST_CASE("Equation")
{
    NeoFOAM::Executor exec = GENERATE(
        NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
        NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
        NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    );

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);
    auto mesh = NeoFOAM::createSingleCellMesh(exec);

    Field fA(exec, 1, 2.0);
    BoundaryFields bf(exec, mesh.nBoundaryFaces(), mesh.nBoundaries());

    std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
    auto vf = VolumeField(exec, mesh, fA, bf, bcs);
    auto fB = Field(exec, 1, 4.0);

    auto a = Dummy(exec, vf);
    auto b = Dummy(exec, vf);

    SECTION("Create from operators " + execName)
    {
        auto eqnA = a + b;
        auto eqnB = fB * Dummy(exec, vf) + 2 * Dummy(exec, vf);

        REQUIRE(eqnA.size() == 2);
        REQUIRE(eqnB.size() == 2);

        REQUIRE(getField(eqnA.explicitOperation()) == 4);
        REQUIRE(getField(eqnB.explicitOperation()) == 12);
    }

    // BENCHMARK("Creation from EqnTerm")
    // {
    //     dsl::EqnSystem eqnSys = lapTerm + divTerm;
    //     return eqnSys;
    // };

    // {
    //     dsl::EqnSystem eqnSys(lapTerm + lapTerm + divTerm + divTerm);
    //     REQUIRE(eqnSys.size() == 4);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 4.0);
    // }
    // BENCHMARK("Creation from multiple terms")
    // {
    //     dsl::EqnSystem eqnSys2(lapTerm + lapTerm + divTerm + divTerm);
    //     return eqnSys2;
    // };

    // {
    //     dsl::EqnSystem eqnSys(lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1));
    //     REQUIRE(eqnSys.size() == 2);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 2.0);
    // }
    // BENCHMARK("Creation from term and temporary term")
    // {
    //     dsl::EqnSystem eqnSys(lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1));
    //     return eqnSys;
    // };

    // {
    //     dsl::EqnSystem eqnSys = lapTerm - divTerm;
    //     REQUIRE(eqnSys.size() == 2);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 0.0);
    // }

    // {
    //     dsl::EqnSystem eqnSys(lapTerm - Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1));
    //     REQUIRE(eqnSys.size() == 2);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 0.0);
    // }

    // {
    //     dsl::EqnSystem eqnSys(lapTerm - lapTerm - divTerm - divTerm);
    //     REQUIRE(eqnSys.size() == 4);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == -2.0);
    // }

    // {
    //     dsl::EqnSystem eqnSys(
    //         lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) + divTerm
    //         + Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     dsl::EqnSystem eqnSys2(
    //         lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) + divTerm
    //         + Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     REQUIRE(eqnSys.size() == 4);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 4.0);
    //     dsl::EqnSystem combinedEqnSys = eqnSys + eqnSys2;
    //     REQUIRE(combinedEqnSys.size() == 8);
    //     REQUIRE(getField(combinedEqnSys.explicitOperation()) == 8.0);
    // }
    // BENCHMARK("Creation from term and temporary term")
    // {
    //     dsl::EqnSystem eqnSys(
    //         lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) + divTerm
    //         + Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     dsl::EqnSystem eqnSys2(
    //         lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) + divTerm
    //         + Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     dsl::EqnSystem combinedEqnSys = eqnSys + eqnSys2;
    //     return combinedEqnSys;
    // };

    // {
    //     dsl::EqnSystem eqnSys(
    //         lapTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) - divTerm
    //         - Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     dsl::EqnSystem eqnSys2(
    //         lapTerm - Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1) - divTerm
    //         - Divergence(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );
    //     REQUIRE(eqnSys.size() == 4);
    //     REQUIRE(getField(eqnSys.explicitOperation()) == 0.0);
    //     REQUIRE(eqnSys2.size() == 4);
    //     REQUIRE(getField(eqnSys2.explicitOperation()) == -2.0);

    //     SECTION("multiplying eqnSys by 2")
    //     {
    //         dsl::EqnSystem multiplyEqnSys = 2.0 * eqnSys2;
    //         REQUIRE(multiplyEqnSys.size() == 4);
    //         REQUIRE(getField(multiplyEqnSys.explicitOperation()) == -4.0);
    //     }

    //     SECTION("adding eqnSys to eqnSys2")
    //     {
    //         dsl::EqnSystem addEqnSys = eqnSys2 + eqnSys;
    //         REQUIRE(addEqnSys.size() == 8);
    //         REQUIRE(getField(addEqnSys.explicitOperation()) == -2.0);
    //     }
    //     SECTION("subtracting eqnSys from eqnSys2")
    //     {
    //         std::cout << "subtracting eqnSys from eqnSys2" << std::endl;
    //         dsl::EqnSystem subtractEqnSys = eqnSys - eqnSys2;
    //         REQUIRE(subtractEqnSys.size() == 8);
    //         REQUIRE(getField(subtractEqnSys.explicitOperation()) == 2.0);
    //     }
    // }
    // // profiling
    // // with different number of terms
    // BENCHMARK("Creation from 2 terms")
    // {
    //     dsl::EqnSystem eqnSys(divTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1));

    //     return eqnSys;
    // };

    // BENCHMARK("Creation from 4 terms")
    // {
    //     dsl::EqnSystem eqnSys(
    //         divTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );

    //     return eqnSys;
    // };

    // BENCHMARK("Creation from 8 terms")
    // {
    //     dsl::EqnSystem eqnSys(
    //         divTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );

    //     return eqnSys;
    // };

    // BENCHMARK("Creation from 16 terms")
    // {
    //     dsl::EqnSystem eqnSys(
    //         divTerm + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //         + Laplacian(dsl::EqnTerm::Type::Explicit, exec, 1)
    //     );

    //     return eqnSys;
    // };
}
