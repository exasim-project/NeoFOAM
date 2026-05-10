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
        auto values = ls_.matrix().values().view();
        
        for (size_t i = 0; i < 4; i++)
        {
            std::cout << "value "<< i << ": " << values[i]<< std::endl;    
        }
        
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

    void juliaFaceBased(
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
        auto deltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);

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

        // vec3<double>
        auto refGrad = phi.boundaryData().refGrad().juliaPtr();
        // double
        auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
        // vec3<double>
        auto refValue = phi.boundaryData().refValue().juliaPtr();
        auto bdeltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);
        // double
        const auto bJUdeltaCoeffs = deltaCoeffs.juliaPtr();

        auto jval = NeoN::Vector<double>(ls_.exec(), nInternalFaces * 2 + nCells, 0.0);
        auto jRHS = NeoN::Vector<double>(ls_.exec(), nCells * 3, 0.0);
        // double
        auto valPtr = jval.juliaPtr();
        // double
        auto rhsPtr = jRHS.juliaPtr();
        size_t nbinputs = 18;
        jl_value_t* args[nbinputs];
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
        args[13] = (jl_value_t*)valueFraction;
        args[14] = (jl_value_t*)refValue;
        args[15] = (jl_value_t*)refGrad;
        args[16] = (jl_value_t*)rhsPtr;
        args[17] = (jl_value_t*)JUsurfFaceCells;
        jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");

        jl_function_t* func = jl_get_function(mod, "faceBasedAll");

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        jl_call(func, args, nbinputs);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Assembly time (JULIA, All): = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << "[ns]" << std::endl;
        jl_value_t* bexc = jl_exception_occurred();
        if (bexc)
        {
            std::cerr << ": Julia exception: " << jl_typeof_str(bexc) << std::endl;
        }
    }

    [[deprecated]] void cFunctionAssembly(
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
        auto deltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);

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

        // vec3<double>
        auto refGrad = phi.boundaryData().refGrad().juliaPtr();
        // double
        auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
        // vec3<double>
        auto refValue = phi.boundaryData().refValue().juliaPtr();
        auto bdeltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces, 3.0);
        // double
        const auto bJUdeltaCoeffs = deltaCoeffs.juliaPtr();

        auto jval = NeoN::Vector<double>(ls_.exec(), nInternalFaces * 2 + nCells, 0.0);
        auto jRHS = NeoN::Vector<double>(ls_.exec(), nCells * 3, 0.0);
        // double
        auto valPtr = jval.juliaPtr();
        // double
        auto rhsPtr = jRHS.juliaPtr();

        jl_value_t* cfunc = jl_eval_string(R"(
            @cfunction(
                test, 
                Cvoid, 
                (
                    Cint,
                    Array{Cint,1}, 
                    Array{Cint,1},
                    Array{Cuchar,1},
                    Array{Cuchar,1},
                    Array{Cuchar,1},
                    Array{Cint,1},
                    Array{Cdouble,1},
                    Cstring,
                    Array{Cdouble,1},
                    Array{Cdouble,1},
                    Array{Cdouble,1},
                    Array{Cdouble,1},
                    Array{Cdouble,1},
                    Array{Cdouble,2},
                    Array{Cdouble,2},
                    Array{Cdouble,1},
                )
            )
        )");
        void* fptr = jl_unbox_voidpointer(cfunc);
        using TestFn = void (*)(
            int32_t,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            const char*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*,
            jl_array_t*
        );
        TestFn fn = reinterpret_cast<TestFn>(fptr);
        fn(nInternalFaces,
           JUowner,
           JUneighbour,
           JUdiagOffs,
           JUownOffs,
           JUneiOffs,
           JUrowOffs,
           valPtr,
           fusedOPString.c_str(),
           JUfaceFluxV,
           JUGamma,
           JUdeltaCoeffs,
           JUmagFaceAreas,
           valueFraction,
           refValue,
           refGrad,
           rhsPtr);
    }
    // template<typename ValueType>
    // void computeDdtDivLapImplCell(
    //     // scalar dt,
    //     const VolumeField<ValueType>& U,
    //     const SurfaceField<scalar>& phi,
    //     const SurfaceField<scalar>& gamma,
    //     const SurfaceInterpolation<ValueType>& divSurfInterp,
    //     //    const SurfaceInterpolation<ValueType>& lapSurfInterp,
    //     const FaceNormalGradient<ValueType>& faceNormalGradient,
    //     const dsl::Coeff coeffA,
    //     const dsl::Coeff coeffB,
    // )
    // {
    //     auto iterator =
    //         std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());

    //     auto exec = ls.exec();
    //     const auto& mesh = phi.mesh();
    //     const auto vol = mesh.cellVolumes().view();
    //     auto matrix = ls.matrix().view();
    //     const auto sp = ls.faceToMatrixAddress();
    //     auto cellBasedData = iterator->getCellBasedData();
    //     auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.views();
    //     auto faceNeighbourV = cellBasedData->faceNeighbour.view();
    //     auto faceSignV = cellBasedData->faceSign.view();
    //     auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.view();

    //     // const auto operatorScaling = this->getCoefficient();
    //     const auto diagOffs = ls.faceToMatrixAddress()->diagOffset().view();
    //     const auto oldVector = oldTime(U).internalVector().view();
    //     auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
    //     auto [colIdx, rowOffs] = ls.matrix().sparsity()->view();

    //     // const scalar a0a1 = 1.0 / dt;

    //     const auto [diaOffV, ownOffV, neiOffV] =
    //         views(sp->diagOffset(), sp->ownerOffset(), sp->neighbourOffset());
    //     const auto [gammaV, deltaV] =
    //         views(gamma.internalVector(), faceNormalGradient.deltaCoeffs().internalVector());

    //     const auto [phiV, /* weightsV,*/ magFaceAreaV] = views(
    //         phi.internalVector(),
    //         // weights.internalVector(),
    //         mesh.magFaceAreas()
    //     );

    //     parallelFor(
    //         exec,
    //         {0, iterator->size()},
    //         NEON_LAMBDA(const localIdx celli) {
    //             // DDT contribution to diagonal
    //             const auto diagIdx = rowOffs[celli] + diaOffV[celli];
    //             // const auto coeff = a0 * vol[celli];
    //             // auto diagValue = coeff * one<ValueType>();
    //             // auto rhsValue = coeff * oldVector[celli];

    //             // Loop over faces of this cell
    //             auto diagValue = zero<ValueType>();
    //             const auto numFaces = cellFacesSegments[celli + 1] - cellFacesSegments[celli];
    //             const auto startIdx = cellFacesSegments[celli];

    //             for (localIdx i = 0; i < numFaces; ++i)
    //             {
    //                 const auto faceIdx = cellFacesValues[startIdx + i];
    //                 const auto neiCell = faceNeighbourV[startIdx + i];
    //                 const auto sign = faceSignV[startIdx + i];

    //                 // Compute flux on-the-fly
    //                 const auto fluxDiv = phiV[faceIdx]; // faceFluxV[faceIdx];
    //                 // FIXME;
    //                 const auto weight = (phiV[faceIdx] >= 0) ? 0.0 : 1.0; // weightsV[faceIdx];
    //                 const auto lapFlux = deltaV[faceIdx] * gammaV[faceIdx] *
    //                 magFaceAreaV[faceIdx]; const auto combinedFlux = (-weight * fluxDiv +
    //                 lapFlux) * one<ValueType>();

    //                 const auto offDiagValue = sign * combinedFlux;
    //                 matrix.values[matrixColumnIdxV[startIdx + i]] += offDiagValue;

    //                 // Contribution to diagonal (subtract off-diagonal)
    //                 diagValue -= offDiagValue;
    //             }

    //             // Write diagonal and RHS
    //             matrix.values[diagIdx] += diagValue;// + a0a1 * one<ValueType>();
    //             // FIXME
    //             // const auto commonCoef = operatorScaling[celli] * vol[celli];
    //             // rhs[celli] += a0a1 * oldVector[celli];
    //         },
    //         "fusedKernelCellBased::cellLoop"
    //     );
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
