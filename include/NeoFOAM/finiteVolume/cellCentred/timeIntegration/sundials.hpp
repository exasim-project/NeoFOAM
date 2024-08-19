// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "arkode/arkode_arkstep.h" // access to ARKStep

#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::sundials
{

// TODO SEE SUNDIALS ARK STEP CONTROLLER TYPES
enum class ARKAdaptControllerType
{
    PID,     // ARK_ADAPT_PID
    PI,      // ARK_ADAPT_PI
    I,       // ARK_ADAPT_I
    EXP_GUS, // ARK_ADAPT_EXP_GUS
    IMP_GUS, // ARK_ADAPT_IMP_GUS
    IMEX_GUS // ARK_ADAPT_IMEX_GUS
};

SUNAdaptController createController(ARKAdaptControllerType controller, SUNContext context)
{
    switch (controller)
    {
    case (ARKAdaptControllerType::PID):
        return SUNAdaptController_PID(context);
    case (ARKAdaptControllerType::PI):
        return SUNAdaptController_PI(context);
    case (ARKAdaptControllerType::I):
        return SUNAdaptController_I(context);
    case (ARKAdaptControllerType::EXP_GUS):
        return SUNAdaptController_ExpGus(context);
    case (ARKAdaptControllerType::IMP_GUS):
        return SUNAdaptController_ImpGus(context);
    case (ARKAdaptControllerType::IMEX_GUS):
        return SUNAdaptController_ImExGus(context);
    default:
        NF_ERROR_EXIT("Invalid ARKAdaptControllerType");
    }
}

}
