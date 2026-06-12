// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 NeoFOAM authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/NeoN.hpp"

#include <algorithm>
#include <cmath>
#include <string>

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
template<
    typename ValueType,
    typename MatrixValueType = NeoN::scalar,
    typename IndexType = NeoN::localIdx>
class PDESolver
{
    using VolumeField = NeoN::finiteVolume::cellCentred::VolumeField<ValueType>;
    // Matrix coefficients use MatrixValueType (scalar by default, giving the segregated
    // vector-solve form for Vec3 fields); the rhs/solution use the field's ValueType.
    // TODO: future work selects MatrixValueType == ValueType (the coupled Vec3 matrix)
    // based on the presence of boundary conditions that require the full block-coupled form.
    using LinearSystem = NeoN::la::LinearSystem<MatrixValueType, ValueType>;

public:

    PDESolver(dsl::Expression<ValueType> expr, VolumeField& psi, RunTime& runTime)
        : psi_(psi)
        , expr_(expr)
        , runTime_(runTime)
        , ls_(readOrCreate<LinearSystem>(
              runTime,
              "linearSystem" + psi.name,
              // FIXME find a proper place
              [&psi, &runTime]()
              { return NeoN::la::createEmptyLinearSystem<MatrixValueType, ValueType>(psi.mesh()); }
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
        // Accept the call in serial (MPI not initialised — mpiRank is the
        // sentinel -1, which would compare unequal to 0 if cast to size_t)
        // and on rank 0 in parallel. SetReference::operator() further guards
        // the matrix mutation against non-rank-0 in initialised MPI.
        const auto& mpiEnv = runTime_.mpiEnvironment;
        if (!mpiEnv.isInitialized() || mpiEnv.rank() == 0)
        {
            needReference_ = true;
            pRefCell_ = pRefCell;
            pRefValue_ = pRefValue;
        }
    }

    /** @brief assemble the linear system owned by the solver based on the current expression */
    LinearSystem& assemble()
    {
        ls_.reset();
        expr_.assemble(runTime_.t, runTime_.dt, ls_, psi_.mesh());
        // auto v = ls_.matrix().values().view();
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
    LinearSystem assemble(dsl::SpatialOperator<NeoN::Vec3>&& rhs)
    {
        auto rhsExpr = dsl::Expression<ValueType>(-1.0 * rhs);
        rhsExpr.read(runTime_.fvSchemesDict);
        auto ls = LinearSystem(assemble());
        rhsExpr.assembleExplicitSource(ls, psi_.mesh());

        return ls;
    }

    NeoN::la::SolverStats solve()
    {
        ls_.reset();
        return solveImpl(expr_, ls_);
    }

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

        reportSolverStats(stats, fvSolution);
        return stats;
    }

    // Public because NVCC forbids extended __host__ __device__ lambdas
    // (NEON_LAMBDA) inside private or protected member functions.
    NeoN::la::SolverStats solveImpl(dsl::Expression<ValueType>& expr, LinearSystem& ls)
    {
        // Re-read schemes (idempotent with the constructor read)
        expr.read(runTime_.fvSchemesDict);

        // Assemble without post-assembly functors; we apply SetReference separately below
        // to ensure correct polymorphic dispatch — storing PostAssemblyBase by value causes
        // object slicing that silently disables virtual overrides.
        expr.assemble(runTime_.t, runTime_.dt, ls, psi_.mesh());

        // Apply reference-cell pinning directly (avoids object-slicing issue)
        if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
        {
            if (needReference_)
            {
                NeoN::dsl::SetReference<ValueType> refFunct(pRefCell_, pRefValue_);
                refFunct(ls);
            }
        }

        auto solverDict = runTime_.fvSolutionDict.subDict("solvers");
        auto fieldSolverDict = solverDict.subDict(psi_.name);
        NeoN::fence(psi_.exec());
        NF_ASSERT(ls.exec() == psi_.exec(), "Executors are not the same");

        auto solver = NeoN::la::Solver(psi_.exec(), fieldSolverDict);
        auto stats = solver.solve(ls, psi_.internalVector());

        reportSolverStats(stats, fieldSolverDict);

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
        if (!iterator->getCellBasedData())
        {
            iterator->setComputeCellBasedData(
                phi.mesh(), ls_.matrix().sparsity(), matIt
            );
        }
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
        jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");
		auto funcName = useGPU ? "cellBased_gpu" : "cellBased_";

        jl_function_t* func = jl_get_function(mod, funcName);

        jl_call(func, args, nbinputs);
        // auto va = ls_.matrix().values();
        // auto cpu = NeoN::CPUExecutor {};
        // auto cpuarr = va.copyToExecutor(cpu);
        // auto v = cpuarr.view();
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
        const nfvcc::SurfaceField<double>& gamma,
        const bool global = false
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
        jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");
		auto funcName = "";
        if (useGPU) {
            if(global){
                funcName = "globalassemble_gpu";
            }
            else {
                funcName = "assemble_gpu";
            }
        }
        else {
            if(global){
                funcName = "globalassemble";
            }
            else {
                funcName = "assemble";
            }
        }
        
        jl_function_t* func = jl_get_function(mod, funcName);
        jl_call(func, args, nbinputs);
        // auto va = ls_.matrix().values();
        // auto cpu = NeoN::CPUExecutor {};
        // auto cpuarr = va.copyToExecutor(cpu);
        // auto v = cpuarr.view();   
        // if (global){     
        //     std::cout << "JULIA GLOBALFACEBASED values: " << std::endl;
        // }
        // else {
        //     std::cout << "JULIA FACEBASED values: " << std::endl;
        // }
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
        jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");

        jl_function_t* func = jl_get_function(mod, "warmup");
        auto fusedOPString = expr_.juliaOP();
        std::cout << "warming up facebased with " << fusedOPString << std::endl;

        jl_call1(func, jl_cstr_to_string(fusedOPString.c_str()));
    }

#endif
private:

    // Per-component name (Ux/Uy/Uz) when a vector field is solved as separate
    // component systems; the plain field name otherwise (scalar or coupled solve).
    static std::string componentName(const std::string& base, std::size_t i, std::size_t n)
    {
        if (n <= 1) return base;
        constexpr const char* suffix[3] = {"x", "y", "z"};
        return i < 3 ? base + suffix[i] : base + "[" + std::to_string(i) + "]";
    }

    // OpenFOAM-style per-solve residual report: "<precond><solver>:  Solving for
    // <field>, Initial residual = ..., Final residual = ..., No Iterations N".
    // The "<precond><solver>" label (e.g. DICPCG) is read from the solver dict's
    // reportName meta key stashed by mapFvSolution -> no recomputation per solve.
    // Logging is rank-aware and skips formatting on muted ranks (NeoN shouldLog).
    void reportSolverStats(
        const NeoN::la::SolverStats& stats,
        const NeoN::Dictionary& fieldSolverDict
    ) const
    {
        const std::string label = fieldSolverDict.contains("reportName")
                                    ? fieldSolverDict.get<std::string>("reportName")
                                    : std::string("Ginkgo");
        const std::size_t n = stats.entries.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& stat = stats.entries[i];
            NeoN::Logging::info(
                "{}:  Solving for {}, Initial residual = {}, Final residual = {}, No Iterations {}",
                label,
                componentName(psi_.name, i, n),
                stat.initResNorm,
                stat.finalResNorm,
                stat.numIter
            );
        }
    }

    VolumeField& psi_;
    dsl::Expression<ValueType> expr_;
    const RunTime& runTime_;
    LinearSystem ls_;
    bool needReference_ = false;
    NeoN::localIdx pRefCell_ = 0;
    NeoN::scalar pRefValue_ = 0.0;
};


template<typename ValueType, typename IndexType = NeoN::localIdx>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> applyOperator(
    const la::LinearSystem<ValueType>& ls,
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


template<typename ValueType>
NeoN::finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const PDESolver<ValueType>& expr,
    const NeoN::finiteVolume::cellCentred::VolumeField<ValueType>& psi
)
{
    return applyOperator(expr.linearSystem(), psi);
}
}
