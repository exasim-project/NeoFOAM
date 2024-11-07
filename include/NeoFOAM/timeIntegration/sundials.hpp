// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

// possibly useful headers.
#include <string>

#include "nvector/nvector_serial.h"
#include "nvector/nvector_kokkos.hpp"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_core.hpp"

#include "arkode/arkode_arkstep.h" // access to ARKStep
#include "arkode/arkode_erkstep.h"
#include "NeoFOAM/core/error.hpp"

namespace NeoFOAM::sundials
{

#if defined(USE_CUDA)
using ExecSpace = Kokkos::Cuda;
#elif defined(USE_HIP)
#if KOKKOS_VERSION / 10000 > 3
using ExecSpace = Kokkos::HIP;
#else
using ExecSpace = Kokkos::Experimental::HIP;
#endif
#elif defined(USE_OPENMP)
using ExecSpace = Kokkos::OpenMP;
#else
using ExecSpace = Kokkos::Serial;
#endif

// Sundials-Kokkos typing
using SKVectorType = ::sundials::kokkos::Vector<ExecSpace>;
using SKSizeType = SKVectorType::size_type;


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
        return nullptr; // avoids compiler warnings
    }
}

ARKODE_ERKTableID stringToERKTable(const std::string& key)
{
    if (key == "Forward Euler") return ARKODE_FORWARD_EULER_1_1;
    if (key == "Heun") return ARKODE_HEUN_EULER_2_1_2;
    if (key == "Midpoint") return ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2;
    NF_ERROR_EXIT("Unsupported Runge-Kutta time inteation method selectied: " + key);
    return ARKODE_ERK_NONE; // avoids compiler warnings.
}

}
