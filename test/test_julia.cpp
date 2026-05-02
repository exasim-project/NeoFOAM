// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "common.hpp"
#include "constrainHbyA.H"
#include <julia.h>
JULIA_DEFINE_FAST_TLS; // only define this once, in an executable (not in a
// shared library) if you want fast code.
namespace fvc = Foam::fvc;
namespace fvm = Foam::fvm;

namespace dsl = NeoN::dsl;
namespace nnfvcc = NeoN::finiteVolume::cellCentred;
namespace nf = NeoFOAM;

extern Foam::Time* timePtr; // A single time object

static bool check_julia_exception(const char* where)
{
    jl_value_t* exc = jl_exception_occurred();
    if (exc) {
        std::cerr << where << ": Julia exception: "
                  << jl_typeof_str(exc) << std::endl;
        return true;
    }
    return false;
}

TEST_CASE("Julia Momentum")
{
    jl_init();

    jl_eval_string(R"(
        function get_operator(tokens::Vector{String})
            if tokens[1] == "DivOperator"
            else # only Laplacian for now
                return Laplace{Float64}(1.0)
            end
            operator = ifelse(tokens[1] == "DivOperator", CentralDiffScheme{Float64},UpwindScheme{Float64}) 
            divscheme = ifelse(tokens[2] == "linear", CentralDiffScheme{Float64},UpwindScheme{Float64}) 
            div = Div{Float64,divscheme}(divscheme(), 1.0) 
            return String(Symbol(div))
        end
        function SOAFusedFaceBasedAssembly(numInteriorFaces::Int32, owner::Ptr{Cvoid}, neighbor::Ptr{Cvoid},  ) where {P<:AbstractFloat}
        #function SOAFusedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
            nu = input.nu
            faces = input.faces
            U_b = input.U_boundary
            U = input.U_internal
            nCells = length(input.cells.index)
            @inbounds for iFace in 1:input.numInteriorFaces
                @inbounds iOwner = faces.iOwner[iFace]
                @inbounds iNeighbor = faces.iNeighbor[iFace]
                @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

                @inbounds vals[faces.ownerIdx[iFace]] += valueUpper
                @inbounds vals[faces.neighborIdx[iFace]] += valueLower
                @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
                @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
            end
            @inbounds for bFace in numInteriorFaces:length(owner)
            @inbounds for iBoundary in eachindex(input.boundaries)
                if U_b[iBoundary].type != "fixedValue"
                    continue
                end
                @inbounds theBoundary = input.boundaries[iBoundary]
                startFace = theBoundary.startFace + 1
                endFace = startFace + theBoundary.nFaces
                for iFace in startFace:endFace-1
                    @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
                    diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

                    @inbounds vals[faces.ownerIdx[iFace]] += diag
                    # RHS/Source
                    @inbounds RHS[faces.iOwner[iFace]] += rhsx
                    @inbounds RHS[faces.iOwner[iFace]+nCells] += rhsy
                    @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
                end
            end
            return vals, RHS
        end # function batchedFaceBasedAssembly
    )");

    float epsilon = 1e-32;
    Foam::Time& runTime = *timePtr;

    // auto [execName, exec] = GENERATE(allAvailableExecutor());
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});

    auto rt = nf::createAdapterRunTime(runTime, exec);
    auto& mesh = rt.mesh;
    auto& schemesDict = rt.fvSchemesDict;
    schemesDict = nf::mapFvSchemes(schemesDict);
    auto divs = schemesDict.subDict("divSchemes");
    auto phiu = divs.get<NeoN::TokenList>("div(phi,U)");

    auto ofU = randomVectorField(runTime, mesh, "U");
    auto ofp = randomScalarField(runTime, mesh, "p");
    ofp.correctBoundaryConditions();
    ofU.correctBoundaryConditions();
    auto& oldOfU = ofU.oldTime();
    oldOfU.primitiveFieldRef() = Foam::vector(0.0, 0.0, 0.0);
    oldOfU.correctBoundaryConditions();

    auto& vectorCollection = nnfvcc::VectorCollection::instance(rt.db, "VectorCollection");
    auto& nfP = NeoFOAM::constructAndRegister(vectorCollection, rt, ofp);

    Foam::surfaceScalarField ofPhi(
        Foam::IOobject(
            "phi",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        fvc::flux(ofU)
    );

    Foam::surfaceScalarField ofNu(
        Foam::IOobject(
            "nu",
            runTime.timeName(),
            mesh,
            Foam::IOobject::NO_READ,
            Foam::IOobject::NO_WRITE
        ),
        mesh,
        Foam::dimensionedScalar("nu", Foam::dimensionSet(0, 2, -1, 0, 0), 0.01)
    );

    auto nfPhi = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofPhi);
    auto nfNu = NeoFOAM::constructFrom(rt.exec, rt.nfMesh, ofNu);

    auto& nfU = NeoFOAM::constructAndRegister(vectorCollection, rt, ofU);
    auto& nfOldU = fvcc::oldTime(nfU);
    NeoN::fill(nfOldU.internalVector(), NeoN::Vec3(0.0, 0.0, 0.0));
    nfOldU.correctBoundaryConditions();

    SECTION("Solve transient momentum without grad(p) on " )
    {

        Foam::fvVectorMatrix ofUEqn(
            fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
        );

        nf::PDESolver<NeoN::Vec3> nfUEqn(
            // dsl::imp::ddt(nfU) + 
            dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
            nfU,
            rt
        );

        NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

        auto& solverDict = rt.fvSolutionDict.subDict("solvers");
        solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

        // Foam::solve(ofUEqn);
        nfUEqn.assembleWithJulia();
        auto operators = "DivOperator:Gauss,Linear;LaplacianOperator";
        jl_eval_string(R"(
            function use_pointer(p::Vector{Float64})
                # arr = unsafe_wrap(Array, Ptr{Float64}(p), 135)
                p[1] = 42.0
                return nothing
            end
        )");

        jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
        jl_array_t* ptr = nfUEqn.coeffMatrixJuliaPtr();
        jl_value_t* julia_ptr = (jl_value_t*)ptr;

        double *xData = jl_array_data(ptr, double);
        jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");
        std::cout << "size : " << nfUEqn.linearSystem().matrix().nNonZeros() << std::endl;

        jl_value_t* args[2];
        args[0] = julia_ptr;
        args[1] = jl_box_int32(nfUEqn.linearSystem().matrix().nNonZeros());
        jl_call1(func, julia_ptr);
        check_julia_exception("use_ptr");
        REQUIRE(nfUEqn.linearSystem().view().matrix.values[0][0] == 42.0);
    }

    // SECTION("Solve transient momentum with grad(p) on " + execName)
    // {
    //     Foam::fvVectorMatrix ofUEqn(
    //         fvm::ddt(ofU) + fvm::div(ofPhi, ofU) - fvm::laplacian(ofNu, ofU)
    //     );

    //     nf::PDESolver<NeoN::Vec3> nfUEqn(
    //         dsl::imp::ddt(nfU) + dsl::imp::div(nfPhi, nfU) - dsl::imp::laplacian(nfNu, nfU),
    //         nfU,
    //         rt
    //     );

    //     NeoN::fill(nfUEqn.linearSystem().rhs(), NeoN::Vec3(0.0, 0.0, 0.0));

    //     // require fields to be initially the same
    //     nf::compare(nfU, ofU, ApproxVector(epsilon));
    //     nf::compare(nfP, ofp, ApproxScalar(epsilon));

    //     auto& solverDict = rt.fvSolutionDict.subDict("solvers");
    //     solverDict.subDict("U") = nf::mapFvSolution(solverDict.subDict("U"));

    //     Foam::solve(ofUEqn == -fvc::grad(ofp));

    //     nfUEqn.solve(-1.0 * dsl::exp::grad(nfP));
    //     nfU.correctBoundaryConditions();
    //     nf::compare(nfU, ofU, ApproxVector({1e-08, 1e-08, 1e-08}));
    // }
    jl_atexit_hook(0);
}
