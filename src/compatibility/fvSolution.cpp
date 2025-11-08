// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: 2023 FoamAdapter authors
/* This file implements comparison operator to compare OpenFOAM and corresponding FoamAdapter fields
 * TODO the comparison operator only make sense for testing purposes
 * so this should be part of the tests
 */


#include "FoamAdapter/compatibility/fvSolution.hpp"
#include <map>
#include <NeoN/core/primitives/scalar.hpp>
#include <NeoN/core/primitives/label.hpp>


namespace FoamAdapter
{

void updateSolver(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM solver names to NeoN/Ginkgo solver names and types
    static const std::map<std::string, std::pair<std::string, std::string>> solverMap = {
        {"PCG", {"Ginkgo", "solver::Cg"}},
        {"PBiCG", {"Ginkgo", "solver::Bicg"}},
        {"PBiCGStab", {"Ginkgo", "solver::Bicgstab"}},
        {"smoothSolver", {"Ginkgo", "solver::Bicgstab"}},
        {"GAMG", {"Ginkgo", "solver::Multigrid"}},
    };

    std::string& solverName = solverDict.get<std::string>("solver");
    auto it = solverMap.find(solverName);
    if (it != solverMap.end())
    {
        std::cout << __FILE__ << ":\n\treplacing solver " << solverName << " by "
                  << it->second.second << "\n";
        solverName = it->second.first;
        // if (solverName == "GAMG")
        // {
        //     throw std::runtime_error(
        //         "GAMG is not supported in FoamAdapter, please use a different solver."
        //     );
        // }
        solverDict.insert("type", it->second.second);
    }
}

void updatePreconditioner(NeoN::Dictionary& solverDict)
{
    // Map OpenFOAM preconditioner types to NeoN/Ginkgo preconditioner types
    static std::map<std::string, NeoN::Dictionary> preconditionerMap = {
        {"DIC",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Jacobi")},
              {std::string("max_block_size"), 1}}
         )},
        {"DILU",
         NeoN::Dictionary(
             {{std::string("type"), std::string("preconditioner::Ilu")},
              {std::string("reverse_apply"), false},
              {std::string("factorization"),
               NeoN::Dictionary({{std::string("type"), std::string("factorization::ParIlu")}})}}
         )},
    };

    // if no preconditioner is set but smoother switch to BiCGStab with BJ
    if (!solverDict.contains("preconditioner"))
    {
        solverDict.insert("preconditioner", preconditionerMap["DIC"]);
    }
    if (solverDict.contains("smoother"))
    {
        solverDict.remove("smoother");
    }

    if (solverDict.isDict("preconditioner"))
    {
        NeoN::Dictionary& preconditionerDict = solverDict.subDict("preconditioner");
        if (preconditionerDict.isDict("type"))
        {
            throw std::runtime_error(
                "GAMG is not supported in FoamAdapter, please use a different preconditioner."
            );
            // std::string& preconditionerType = preconditionerDict.get<std::string>("type");
            // auto it = preconditionerMap.find(preconditionerType);
            // if (it != preconditionerMap.end())
            // {
            //     std::cout << __FILE__ << " replacing preconditioner " << preconditionerType << "
            //     by " << it->second << "\n"; preconditionerType = it->second;
            // }
        }
    }
    else
    {
        // If no preconditioner is specified, we can insert a default one
        std::string& preconditionerName = solverDict.get<std::string>("preconditioner");
        auto it = preconditionerMap.find(preconditionerName);
        if (it != preconditionerMap.end())
        {
            std::cout << __FILE__ << ":\n\treplacing preconditioner " << preconditionerName
                      << " by " << it->second << "\n";
            solverDict.insert("preconditioner", it->second);
        }
    }
}

void updateCriteria(NeoN::Dictionary& solverDict)
{
    // Ensure the criteria dictionary exists
    if (!solverDict.contains("criteria"))
    {
        solverDict.insert("criteria", NeoN::Dictionary());
    }

    // set default max iteration count
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("iteration", 1000);
    }

    // Set default values for relative residual norm and iteration count
    if (solverDict.contains("relTol"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("relative_residual_norm", solverDict.get<NeoN::scalar>("relTol"));
        solverDict.remove("relTol");
    }
    if (solverDict.contains("maxIter"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("iteration", solverDict.get<NeoN::label>("maxIter"));
        solverDict.remove("maxIter");
    }
    if (solverDict.contains("tolerance"))
    {
        NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
        criteriaDict.insert("absolute_residual_norm", solverDict.get<NeoN::scalar>("tolerance"));
        solverDict.remove("tolerance");
    }

    NeoN::Dictionary& criteriaDict = solverDict.subDict("criteria");
    std::cout << __FILE__ << ":\n\tStopping criteria: " << criteriaDict << "\n";
}


NeoN::Dictionary mapFvSolution(const NeoN::Dictionary& solverDict)
{
    NeoN::Dictionary modSolverDict = solverDict;

    if (solverDict.contains("configFile")) return solverDict;
    std::cout << __FILE__ << ":\n\tMapping OpenFOAM solver settings to NeoN settings\n"
              << "\tCurrently, it is advisable to specify configFile for fine grained Ginkgo "
                 "solver support\n";

    updateSolver(modSolverDict);
    updatePreconditioner(modSolverDict);
    updateCriteria(modSolverDict);

    return modSolverDict;
}

} // namespace FoamAdapter
