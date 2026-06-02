// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include "NeoFOAM/datastructures/runTime.hpp"
#ifdef NeoN_WITH_JULIA
#include <julia.h>
#endif
#include "NeoFOAM/compatibility/fvSolution.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

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

    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, RunTime& runTime, LinearSystem& ls)
    : psi_(psi)
    , expr_(expr)
    , runTime_(runTime)
    , ls_(ls)
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
        // auto v = ls_.matrix().values().view();
        // auto rhs = ls_.rhs().view();
        // auto bRhs = ls_.boundaryRhs().view();
        // auto bValues = ls_.boundaryMatrix().values().view();
        // auto va = ls_.matrix().values();
        // auto cpu = NeoN::CPUExecutor {};
        // auto cpuarr = va.copyToExecutor(cpu);
        // auto v = cpuarr.view();
        // std::cout << "NEON values: " << std::endl;
        // std::cout << v[0] << std::endl;
        // std::cout << v[1] << std::endl;
        // std::cout << v[2] << std::endl;
        // std::cout << v[3] << std::endl;
        // std::cout << v[4] << std::endl;
        // std::cout << v[v.size()-1] << std::endl;

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
#ifdef NeoN_WITH_JULIA

   void juliaCellbased(
       const nfvcc::SurfaceField<double>& faceFlux,
       const nfvcc::VolumeField<ValueType>& phi,
       const nfvcc::SurfaceField<double>& gamma
   )
   {
        const auto& mesh = phi.mesh();
        const auto nInternalFaces = mesh.nInternalFaces();
        const auto nBoundaryFaces = mesh.nBoundaryFaces();
        const auto nTotalFaces = nInternalFaces + nBoundaryFaces;
        auto fusedOPString = expr_.juliaOP();

        auto deltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces);
        auto deltaCoeffsB = NeoN::Vector<double>(ls_.exec(), nBoundaryFaces);

        auto deltaCoeff = deltaCoeffs.view();
        auto deltaCoeffB = deltaCoeffsB.view();

        bool useGPU = std::holds_alternative<NeoN::GPUExecutor>(mesh.exec());
        const auto [faceCenters, cellCenters] = views(mesh.faceCenters(), mesh.cellCenters());
        const auto [owners, neighbors, surfFaceCells] = views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());
        
        parallelFor(
            ls_.exec(),
            {0, nInternalFaces},
            NEON_LAMBDA(const size_t facei) {
                NeoN::Vec3 cellToCellDist = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
                deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
            },
            "basicGeometricScheme::updateDeltaCoeffsInternal"
        );

        parallelFor(
            ls_.exec(),
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const size_t bfi) {
                auto own = surfFaceCells[bfi];
                // TODO Issue #515
                NeoN::Vec3 cellToCellDist = faceCenters[nInternalFaces + bfi] - cellCenters[own];
                deltaCoeffB[bfi] = 1.0 / mag(cellToCellDist);
            },
            "basicGeometricScheme::updateDeltaCoeffsBoundary"
        );

        const auto matIt = ls_.faceToMatrixAddress();
        auto deltas = nfvcc::SurfaceField<NeoN::scalar>(
            mesh.exec(),
            "deltaCoeffs",
            mesh,
            nfvcc::createCalculatedBCs<nfvcc::SurfaceBoundary<NeoN::scalar>>(mesh)
        );
        auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls_.getMeshIterator()->get());

        auto exec = ls_.exec();
        auto matrix = ls_.matrix().juliaPtr();
        const auto sp = ls_.faceToMatrixAddress();
        auto cellBasedData = iterator->getCellBasedData();
        auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.juliaPtrs();
        auto faceSignV = cellBasedData->faceSign.juliaPtr();
        auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.juliaPtr();

        const auto diagOffs = sp->diagOffset().juliaPtr();
        // double
        const auto iGamma = gamma.internalVector().juliaPtr();
        // double
        const auto bGamma = gamma.boundaryData().value().juliaPtr();
        // double
        const auto ideltaCoeffs = deltaCoeffs.juliaPtr();
        // // double
        const auto bdeltaCoeffs = deltaCoeffsB.juliaPtr();
        const auto rowOffs = ls_.matrix().sparsity()->rowOffs().juliaPtr();

        const auto magFaceAreas = mesh.faceAreas().juliaPtr();
        const auto jowners = mesh.faceOwners().juliaPtr();

        // double
        const auto ifaceFluxV = faceFlux.internalVector().juliaPtr();
        // double
        const auto bfaceFluxV = faceFlux.boundaryData().value().juliaPtr();
        const auto boundaryFaceOwners = mesh.boundaryMesh().faceOwners().juliaPtr();

        auto valPtr = ls_.matrix().juliaPtr();
        // double
        auto rhsPtr = ls_.rhs().juliaPtr();
        auto bRhs = ls_.boundaryRhs().juliaPtr();
        auto bValues = ls_.boundaryMatrix().juliaPtr();
        const auto vol = mesh.cellVolumes().juliaPtr();
        const auto oldVector = oldTime(phi).internalVector().juliaPtr();
        // vec3<double>
        auto refGrad = phi.boundaryData().refGrad().juliaPtr();
        // double
        auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
        // vec3<double>
        auto refValue = phi.boundaryData().refValue().juliaPtr();
        size_t nbinputs = 27;
        if (useGPU)
        {            
            nbinputs += 2;
        }
        jl_value_t* args[nbinputs];
        args[0] =  jl_box_int32(iterator->size());
        args[1] =  (jl_value_t*)jowners;
        args[2] =  (jl_value_t*)cellFacesSegments;
        args[3] =  (jl_value_t*)diagOffs;
        args[4] =  (jl_value_t*)rowOffs;
        args[5] =  (jl_value_t*)cellFacesValues;
        args[6] =  (jl_value_t*)faceSignV;
        args[7] =  (jl_value_t*)valPtr;
        args[8] =  jl_cstr_to_string(fusedOPString.c_str());
        args[9] =  (jl_value_t*)ifaceFluxV;
        args[10] = (jl_value_t*)bfaceFluxV;
        args[11] = (jl_value_t*)iGamma;
        args[12] = (jl_value_t*)bGamma;
        args[13] = (jl_value_t*)ideltaCoeffs;
        args[14] = (jl_value_t*)bdeltaCoeffs;
        args[15] = (jl_value_t*)matrixColumnIdxV;
        args[16] = (jl_value_t*)magFaceAreas;
        args[17] = (jl_value_t*)valueFraction;
        args[18] = (jl_value_t*)boundaryFaceOwners;
        args[19] = (jl_value_t*)bValues;
        args[20] = (jl_value_t*)bRhs;
        args[21] = (jl_value_t*)vol;
        args[22] = (jl_value_t*)oldVector;
        args[23] = (jl_value_t*)refValue;
        args[24] = (jl_value_t*)refGrad;
        args[25] = (jl_value_t*)rhsPtr;
        args[26] = jl_box_float64(runTime_.dt);
        if (useGPU){
            args[27] = jl_box_int32(nInternalFaces);
            args[28] = jl_box_int32(nTotalFaces);
        }         
        jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");
		auto funcName = useGPU ? "cellBased_gpu" : "cellBased_";

        jl_function_t* func = jl_get_function(mod, funcName);

        jl_call(func, args, nbinputs);
        auto va = ls_.matrix().values();
        auto cpu = NeoN::CPUExecutor {};
        auto cpuarr = va.copyToExecutor(cpu);
        auto v = cpuarr.view();
        // auto v = ls_.matrix().values().view();
        // std::cout << "JULIA CELLBASED values: " << std::endl;
        // std::cout << v[0] << std::endl;
        // std::cout << v[1] << std::endl;
        // std::cout << v[2] << std::endl;
        // std::cout << v[3] << std::endl;
        // std::cout << v[4] << std::endl;
        // std::cout << v[v.size()-1] << std::endl;
        if (jl_exception_occurred())
        {
            const char* p = jl_string_ptr(
                jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
            );

            fprintf(stderr, "%s%s\n", "error: ", p);
        }
   }

    void juliaFaceBased(
        const nfvcc::SurfaceField<double>& faceFlux,
        const nfvcc::VolumeField<ValueType>& phi,
        const nfvcc::SurfaceField<double>& gamma
    )
    {
        const NeoN::UnstructuredMesh& mesh = phi.mesh();
        const auto nInternalFaces = mesh.nInternalFaces();
        const auto nBoundaryFaces = mesh.nBoundaryFaces();
        
        const auto [owners, neighbors, surfFaceCells] = views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());


        const auto [faceCenters, cellCenters] = views(mesh.faceCenters(), mesh.cellCenters());
        auto deltaCoeffs = NeoN::Vector<double>(ls_.exec(), nInternalFaces);
        auto deltaCoeffsB = NeoN::Vector<double>(ls_.exec(), nBoundaryFaces);

        auto deltaCoeff = deltaCoeffs.view();
        auto deltaCoeffB = deltaCoeffsB.view();

        bool useGPU = std::holds_alternative<NeoN::GPUExecutor>(mesh.exec());
        
        parallelFor(
            ls_.exec(),
            {0, nInternalFaces},
            NEON_LAMBDA(const size_t facei) {
                NeoN::Vec3 cellToCellDist = cellCenters[neighbors[facei]] - cellCenters[owners[facei]];
                deltaCoeff[facei] = 1.0 / mag(cellToCellDist);
            },
            "basicGeometricScheme::updateDeltaCoeffsInternal"
        );

        parallelFor(
            ls_.exec(),
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const size_t bfi) {
                auto own = surfFaceCells[bfi];
                // TODO Issue #515
                NeoN::Vec3 cellToCellDist = faceCenters[nInternalFaces + bfi] - cellCenters[own];
                deltaCoeffB[bfi] = 1.0 / mag(cellToCellDist);
            },
            "basicGeometricScheme::updateDeltaCoeffsBoundary"
        );
        
        
        const auto nTotalFaces = mesh.nTotalFaces();
        const auto nCells = mesh.nCells();
        auto fusedOPString = expr_.juliaOP();

        // double
        const auto ifaceFluxV = faceFlux.internalVector().juliaPtr();
        // double
        const auto bfaceFluxV = faceFlux.boundaryData().value().juliaPtr();
        // int32
        const auto JUowner = mesh.faceOwners().juliaPtr();
        // int32
        const auto JUneighbour = mesh.faceNeighbors().juliaPtr();
        // int32
        const auto boundaryFaceOwners = mesh.boundaryMesh().faceOwners().juliaPtr();
        // uint8
        const auto JUdiagOffs = ls_.faceToMatrixAddress()->diagOffset().juliaPtr();
        // uint8
        const auto JUownOffs = ls_.faceToMatrixAddress()->ownerOffset().juliaPtr();
        // uint8
        const auto JUneiOffs = ls_.faceToMatrixAddress()->neighbourOffset().juliaPtr();
        // int32
        const auto JUrowOffs = ls_.matrix().sparsity()->rowOffs().juliaPtr();

        // double
        const auto iGamma = gamma.internalVector().juliaPtr();
        // double
        const auto bGamma = gamma.boundaryData().value().juliaPtr();
        // double
        const auto ideltaCoeffs = deltaCoeffs.juliaPtr();
        // // double
        const auto bdeltaCoeffs = deltaCoeffsB.juliaPtr();
        // double
        const auto magFaceAreas = mesh.faceAreas().juliaPtr();

        // vec3<double>
        auto refGrad = phi.boundaryData().refGrad().juliaPtr();
        // double
        auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
        // vec3<double>
        auto refValue = phi.boundaryData().refValue().juliaPtr();


        auto valPtr = ls_.matrix().values().juliaPtr();
        // double
        auto rhsPtr = ls_.rhs().juliaPtr();
        auto bRhs = ls_.boundaryRhs().juliaPtr();
        auto bValues = ls_.boundaryMatrix().values().juliaPtr();
        const auto vol = mesh.cellVolumes().juliaPtr();
        const auto oldVector = oldTime(phi).internalVector().juliaPtr();
        size_t nbinputs = 26;
        if (useGPU)
            nbinputs += 2;
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
        args[9] =  (jl_value_t*)ifaceFluxV;
        args[10] = (jl_value_t*)bfaceFluxV;
        args[11] = (jl_value_t*)iGamma;
        args[12] = (jl_value_t*)bGamma;
        args[13] = (jl_value_t*)ideltaCoeffs;
        args[14] = (jl_value_t*)bdeltaCoeffs;
        args[15] = (jl_value_t*)magFaceAreas;
        args[16] = (jl_value_t*)valueFraction;
        args[17] = (jl_value_t*)refValue;
        args[18] = (jl_value_t*)refGrad;
        args[19] = (jl_value_t*)rhsPtr;
        args[20] = (jl_value_t*)boundaryFaceOwners;
        args[21] = (jl_value_t*)bValues;
        args[22] = (jl_value_t*)bRhs;
        args[23] = (jl_value_t*)vol;
        args[24] = (jl_value_t*)oldVector;
        args[25] = jl_box_float64(runTime_.dt);
        if (useGPU){
			args[26] = jl_box_int32(nCells);
			args[27] = jl_box_int32(nTotalFaces);	
		}
        jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");
		auto funcName = useGPU ? "assemble_gpu" : "assemble";
        jl_function_t* func = jl_get_function(mod, funcName);
        jl_call(func, args, nbinputs);
        // auto v = ls_.matrix().values().view();
        // std::cout << "JULIA FACEBASED values: " << std::endl;
        // std::cout << v[0] << std::endl;
        // std::cout << v[1] << std::endl;
        // std::cout << v[2] << std::endl;
        // std::cout << v[3] << std::endl;
        // std::cout << v[4] << std::endl;
        // std::cout << v[v.size()-1] << std::endl;
        if (jl_exception_occurred())
        {
            const char* p = jl_string_ptr(
                jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
            );

            fprintf(stderr, "%s%s\n", "error: ", p);
        }
    }

    void warmupFaceBased()
    {
        jl_module_t* mod = (jl_module_t*)jl_eval_string("MinimalFVM");

        jl_function_t* func = jl_get_function(mod, "warmup");
        auto fusedOPString = expr_.juliaOP();
        std::cout << "warming up facebased with " << fusedOPString << std::endl;

        jl_call1(func, jl_cstr_to_string(fusedOPString.c_str()));
    }

#endif
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
