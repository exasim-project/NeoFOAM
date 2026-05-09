// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"
#include "NeoFOAM/compatibility/fvSolution.hpp"

namespace dsl = NeoN::dsl;
namespace nfvcc = NeoN::finiteVolume::cellCentred;

namespace NeoFOAM
{

/*@brief extends expression by giving access to assembled matrix
 * @note used in neoIcoFOAM directly instead of dsl::expression
 * TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
 * for dependent operations like discrete momentum fields
 * needs storage for assembled matrix? and whether update is needed like for rAU and HbyA
 */
template<typename ValueType, typename IndexType = NeoN::localIdx>
class PDESolver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    using LinearSystem =
        NeoN::la::LinearSystem<ValueType, NeoN::la::CSRMatrix<ValueType, NeoN::localIdx>>;

public:

    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, RunTime& runTime)
        : psi_(psi)
        , expr_(expr)
        , runTime_(runTime)
        , ls_(readOrCreate<LinearSystem>(
              runTime,
              "linearSystem" + psi.name,
              [&psi]() { return NeoN::la::createEmptyLinearSystem<ValueType>(psi.mesh()); }
          ))
    {
        expr_.read(runTime_.fvSchemesDict);
    };

    PDESolver(const PDESolver& expr)
        : psi_(expr.psi_)
        , expr_(expr.expr_)
        , runTime_(expr.runTime_)
        , ls_(expr.ls_) {};

    ~PDESolver() = default;

    VolumeField& getField() { return this->psi_; }

    const VolumeField& getField() const { return this->psi_; }

    [[nodiscard]] LinearSystem& linearSystem() { return ls_; }

    [[nodiscard]] const LinearSystem& linearSystem() const { return ls_; }

    NeoN::dsl::Expression<ValueType>& expression() { return expr_; }

    const NeoN::Executor& exec() const { return ls_.exec(); }

    template<typename FunctorValueType>
    struct SetReference : public NeoN::dsl::PostAssemblyBase<ValueType, IndexType>
    {

        NeoN::localIdx pRefCell_;
        NeoN::scalar pRefValue_;

        SetReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
            : pRefCell_(pRefCell)
            , pRefValue_(pRefValue)
        {}

        virtual void operator()(NeoN::la::LinearSystem<
                                FunctorValueType,
                                NeoN::la::CSRMatrix<FunctorValueType, IndexType>>& ls)
        {
            const auto rowOffs = ls.matrix().sparsity()->rowOffs().view();
            const auto diagOffset = ls.faceToMatrixAddress()->diagOffset().view();
            auto rhs = ls.rhs().view();
            auto values = ls.matrix().values().view();
            // make an explicit copy to avoid capture this warning in kokkos lambda
            auto pRefValue = pRefValue_;

            NeoN::parallelFor(
                ls.exec(),
                {pRefCell_, pRefCell_ + 1},
                NEON_LAMBDA(const std::size_t refCelli) {
                    auto diagIdx = rowOffs[refCelli] + diagOffset[refCelli];
                    auto diagValue = values[diagIdx];
                    rhs[refCelli] += diagValue * pRefValue;
                    values[diagIdx] += diagValue;
                }
            );
        }
    };

    NeoN::finiteVolume::cellCentred::DdtScheme ddtScheme() const
    {
        for (const auto& op : expr_.temporalOperators())
        {
            const auto s = op.ddtScheme();
            if (s != NeoN::finiteVolume::cellCentred::DdtScheme::None)
            {
                return s;
            }
        }
        return NeoN::finiteVolume::cellCentred::DdtScheme::None;
    }

    void setReference(NeoN::localIdx pRefCell, NeoN::scalar pRefValue)
    {
        needReference_ = true;
        pRefCell_ = pRefCell;
        pRefValue_ = pRefValue;
    }

    /** @brief assemble the linear system owned by the solver based on the current expression */
    LinearSystem& assemble()
    {
        expr_.assemble(runTime_.t, runTime_.dt, ls_);
        return ls_;
    }


    /** @brief assemble the linear system with an additional rhs term
     *
     * the following assembly logic is applied
     * 1. the "owned" linear system gets assembled
     * 2. a copy of the assembled linear system is made and the rhs is assembled
     * 3. the new linear system with rhs is returned
     */
    NeoN::la::LinearSystem<ValueType> assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(runTime_.fvSchemesDict);
        auto ls = NeoN::la::LinearSystem<ValueType>(assemble());

        auto expTmp = rhsExpr.explicitOperation(psi_.mesh().nCells());

        auto [vol, expSource, rhsV] = NeoN::views(psi_.mesh().cellVolumes(), expTmp, ls.rhs());
        NeoN::parallelFor(
            psi_.exec(),
            {0, rhsV.size()},
            NEON_LAMBDA(const NeoN::localIdx i) { rhsV[i] -= expSource[i] * vol[i]; }
        );

        return ls;
    }

    NeoN::la::SolverStats solve() { return solveImpl(expr_, ls_); }

    /** @brief solve expression with additional rhs
     *
     * This function will create two versions of the linear system corresponding to the expression
     * 1. the owned linear system without rhs is assembled and stored
     * 2. a temporary linear system with rhs assembled and solved
     */
    NeoN::la::SolverStats solve(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        // assemble wo rhs first
        auto ls = assemble(std::move(rhs));

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fvSolution = solverDict.subDict(psi_.name);
        auto solver = NeoN::la::Solver(psi_.exec(), fvSolution);
        // Do some sanity checks before trying to solve
        // NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
        auto stats = solver.solve(ls, psi_.internalVector());

        for (auto& stat : stats.entries)
        {
            NeoN::Logging::info(
                "Solving for {} Initial residual: {} Final residual: {} No Iterations: {}",
                psi_.name,
                stat.initResNorm,
                stat.finalResNorm,
                stat.numIter
            );
        }
        return stats;
    }

    std::string juliaOP() { return expr_.juliaOP(); }


    void juliaAssemble(
        const nfvcc::SurfaceField<double>& faceFlux,
        const nfvcc::VolumeField<ValueType>& phi,
        const nfvcc::SurfaceField<double>& gamma
    )
    {
        const auto matIt = ls_.faceToMatrixAddress();
        const NeoN::UnstructuredMesh& mesh = phi.mesh();
        const auto nInternalFaces = mesh.nInternalFaces();
        const auto nCells = mesh.nCells();
        auto fusedOPString = juliaOP();
        jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");
        auto deltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);
        std::cout << "internal\n";
        jl_function_t* func = jl_get_function(mod, "faceBased");

        // double
        const auto JUfaceFluxV = faceFlux.internalVector().juliaPtr();
        // int32
        const auto JUowner = mesh.faceOwner().juliaPtr();
        // int32
        const auto JUneighbour = mesh.faceNeighbour().juliaPtr();
        // int32
        const auto JUsurfFaceCells = mesh.boundaryMesh().faceCells().juliaPtr();
        // uint8
        const auto JUdiagOffs = matIt->diagOffset().juliaPtr();
        // uint8
        const auto JUownOffs = matIt->ownerOffset().juliaPtr();
        // uint8
        const auto JUneiOffs = matIt->neighbourOffset().juliaPtr();
        // int32
        const auto JUrowOffs = matIt->sparsityPattern()->rowOffs().juliaPtr();
        // vec3<double>
        const auto phiJuliaPtr = phi.internalVector().juliaPtr();

        // double
        const auto JUGamma = gamma.internalVector().juliaPtr();
        // double
        const auto JUdeltaCoeffs = deltaCoeffs.juliaPtr();
        // double
        const auto JUmagFaceAreas = mesh.magFaceAreas().juliaPtr();

        auto jval = NeoN::Vector<double>(ls_.exec(), nInternalFaces * 2 + nCells, 0.0);
        auto jRHS = NeoN::Vector<double>(ls_.exec(), nCells * 3, 0.0);
        // double
        auto valPtr = jval.juliaPtr();
        // double
        auto rhsPtr = jRHS.juliaPtr();

        /*
            numInteriorFaces::Int32,
            numCells::Int32,
            owner::Vector{Int32},
            neighbour::Vector{Int32},
            diagOffs::Vector{UInt8},
            ownOffs::Vector{UInt8},
            neiOffs::Vector{UInt8},
            rowOffs::Vector{Int32},
            vals::Vector{Float64},
            phi_::Matrix{Float64},
            opString::String,
            faceFlux::Vector{Float64},
            gamma::Vector{Float64},
            deltaCoeffs::Vector{Float64},
            magFaceArea::Vector{Float64}
            )
        */

        size_t nInputs = 13;
        jl_value_t* args[nInputs];
        args[0] = jl_box_int32(nInternalFaces);
        args[1] = (jl_value_t*)JUowner;
        args[2] = (jl_value_t*)JUneighbour;
        args[3] = (jl_value_t*)JUdiagOffs;
        args[4] = (jl_value_t*)JUownOffs;
        args[5] = (jl_value_t*)JUneiOffs;
        args[6] = (jl_value_t*)JUrowOffs;
        args[7] = (jl_value_t*)valPtr;
        args[8] = jl_cstr_to_string(fusedOPString.c_str());
        args[9] = (jl_value_t*)JUfaceFluxV;
        args[10] = (jl_value_t*)JUGamma;
        args[11] = (jl_value_t*)JUdeltaCoeffs;
        args[12] = (jl_value_t*)JUmagFaceAreas;

        jl_call(func, args, nInputs);
        jl_value_t* exc = jl_exception_occurred();
        if (exc)
        {
            std::cerr << ": Julia exception: " << jl_typeof_str(exc) << std::endl;
        }
        std::cout << "boundary\n";

        jl_function_t* bfunc = jl_get_function(mod, "faceBasedBoundary");
        // auto [refGradient, valueFraction, refValue] = juliaPtrs(
        // vec3<double>
        auto refGrad = phi.boundaryData().refGrad().juliaPtr();
        // double
        auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
        // vec3<double>
        auto refValue = phi.boundaryData().refValue().juliaPtr();
        auto bdeltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);
        // double
        const auto bJUdeltaCoeffs = deltaCoeffs.juliaPtr(); // scalar

        size_t nbinputs = 14;
        jl_value_t* bargs[nbinputs];
        bargs[0] = jl_box_int32(nInternalFaces);
        bargs[1] = (jl_value_t*)JUsurfFaceCells;
        bargs[2] = (jl_value_t*)JUdiagOffs;
        bargs[3] = (jl_value_t*)JUrowOffs;
        bargs[4] = (jl_value_t*)valPtr;
        bargs[5] = jl_cstr_to_string(fusedOPString.c_str());
        bargs[6] = (jl_value_t*)JUfaceFluxV;
        bargs[7] = (jl_value_t*)JUGamma;
        bargs[8] = (jl_value_t*)bJUdeltaCoeffs;
        bargs[9] = (jl_value_t*)JUmagFaceAreas;
        bargs[10] = (jl_value_t*)valueFraction;
        bargs[11] = (jl_value_t*)refValue;
        bargs[12] = (jl_value_t*)refGrad;
        bargs[13] = (jl_value_t*)rhsPtr;
        std::chrono::steady_clock::time_point begin3j = std::chrono::steady_clock::now();
        jl_call(bfunc, bargs, nbinputs);
        std::chrono::steady_clock::time_point endj3 = std::chrono::steady_clock::now();
        std::cout << "Assembly time (JULIA) boundary: = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(endj3 - begin3j).count()
                  << "[ns]" << std::endl;
        jl_value_t* bexc = jl_exception_occurred();
        if (bexc)
        {
            std::cerr << ": Julia exception: " << jl_typeof_str(bexc) << std::endl;
        }
    }

    // void warmupB()
    // {
    //     jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");
    //     jl_function_t* bfunc = jl_get_function(mod, "faceBasedBoundary");
    //     size_t nbinputs = 14;
    //     jl_value_t* bargs[nbinputs];
    //     int32_t nInt = 1;
    //     auto tmpdouble = NeoN::Vector<double>(ls_.exec(), 1, 0.0);
    //     auto tmpInt = NeoN::Vector<int32_t>(ls_.exec(), 1, 0.0);
    //     // auto tmpuInt = NeoN::Vector<uint8_t>(ls_.exec(), 1, 0.0);
    //     auto tmpuInt = NeoN::Array<uint8_t>(ls_.exec(), 1, 0.0);
    //     auto tmpumat = NeoN::Vector<NeoN::Vec3>(ls_.exec(), 1, NeoN::zero<NeoN::Vec3>());

    //     const auto JUtmpdouble = tmpdouble.juliaPtr();
    //     const auto JUtmpInt = tmpInt.juliaPtr();
    //     const auto JUtmpuInt = tmpuInt.juliaPtr();
    //     const auto JUtmpumat = tmpumat.juliaPtr();


    //     bargs[0] = jl_box_int32(nInt);
    //     bargs[1] = (jl_value_t*)JUtmpuInt;
    //     bargs[2] = (jl_value_t*)JUtmpuInt;
    //     bargs[3] = (jl_value_t*)JUtmpInt;
    //     bargs[4] = (jl_value_t*)JUtmpdouble;
    //     bargs[5] = jl_cstr_to_string("Div{Float64, upwind{Float64}}(upwind{Float64}(), 1.0)");
    //     bargs[6] = (jl_value_t*)JUtmpdouble;
    //     bargs[7] = (jl_value_t*)JUtmpdouble;
    //     bargs[8] = (jl_value_t*)JUtmpdouble;
    //     bargs[9] = (jl_value_t*)JUtmpdouble;
    //     bargs[10] = (jl_value_t*)JUtmpdouble;
    //     bargs[11] = (jl_value_t*)JUtmpumat;
    //     bargs[12] = (jl_value_t*)JUtmpumat;
    //     bargs[13] = (jl_value_t*)JUtmpdouble;
    //     std::chrono::steady_clock::time_point begin3j = std::chrono::steady_clock::now();
    //     jl_call(bfunc, bargs, nbinputs);
    //     std::chrono::steady_clock::time_point endj3 = std::chrono::steady_clock::now();
    //     std::cout << "Assembly time (JULIA) warumup = "
    //               << std::chrono::duration_cast<std::chrono::microseconds>(endj3 - begin3j).count()
    //               << "[ns]" << std::endl;
    // }

private:

    NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Only if ValueType is scalar
        auto functs = std::vector<NeoN::dsl::PostAssemblyBase<ValueType, IndexType>> {};

        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            functs =
                needReference_
                    ? std::vector<NeoN::dsl::PostAssemblyBase<ValueType, IndexType>> {SetReference<
                          ValueType>(pRefCell_, pRefValue_)}
                    : std::vector<NeoN::dsl::PostAssemblyBase<ValueType, IndexType>> {};
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fieldSolverDict = solverDict.subDict(psi_.name);

        auto stats = NeoN::la::SolverStats();
        // TODO NOTE: This is a temporary solution to avoid negative values on the diagonal
        // when IC is selected as preconditioner by scaling the system matrix with -1.0.
        // NOTE: This will produce -p as a result.
        if (psi_.name == "p" && fieldSolverDict.contains("preconditioner")
            && fieldSolverDict.subDict("preconditioner").template get<std::string>("type")
                   == "preconditioner::Ic")
        {
            auto exprIn = -1.0 * expr;
            stats = NeoN::dsl::detail::iterativeSolveImpl(
                exprIn,
                ls,
                psi_,
                runTime_.t,
                runTime_.dt,
                runTime_.fvSchemesDict,
                fieldSolverDict,
                functs
            );
        }
        else
        {
            stats = NeoN::dsl::detail::iterativeSolveImpl(
                expr,
                ls,
                psi_,
                runTime_.t,
                runTime_.dt,
                runTime_.fvSchemesDict,
                fieldSolverDict,
                functs
            );
        }

        for (auto stat : stats.entries)
        {
            NeoN::Logging::info(
                "Solving for {} Initial residual: {} Final residual: {} No Iterations: {}",
                psi_.name,
                stat.initResNorm,
                stat.finalResNorm,
                stat.numIter
            );
        }

        return stats;
    }

    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    LinearSystem ls_;
    bool needReference_;
    NeoN::localIdx pRefCell_;
    NeoN::scalar pRefValue_;
};


template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> applyOperator(
    const la::LinearSystem<ValueType, NeoN::la::CSRMatrix<ValueType, IndexType>>& ls,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    NeoN::finiteVolume::cellCentred::VolumeField<ValueType> res(
        psi.exec(),
        "ls_" + psi.name,
        psi.mesh(),
        psi.internalVector(),
        psi.boundaryData(),
        psi.boundaryConditions()
    );
    NeoN::la::computeResidual(ls.matrix(), ls.rhs(), psi.internalVector(), res.internalVector());
    return res;
}


template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const PDESolver<ValueType, IndexType> expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}

}
