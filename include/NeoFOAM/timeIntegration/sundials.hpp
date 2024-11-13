// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <functional>
#include <memory>

#include "nvector/nvector_serial.h"
#include "nvector/nvector_kokkos.hpp"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_core.hpp"
#include "arkode/arkode_arkstep.h"
#include "arkode/arkode_erkstep.h"

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/parallelAlgorithms.hpp"
#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM::sundials
{

// Sundials-Kokkos typing
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
using SKVectorType = ::sundials::kokkos::Vector<ExecSpace>;
using SKSizeType = SKVectorType::size_type;

/**
 * Custom deleter for SUNContext_ shared pointers, frees the context if last context.
 * @param[in] ctx Pointer to the SUNContext_ to be freed, can be nullptr.
 */
auto SUNContextDeleter = [](SUNContext_* ctx)
{
    if (ctx != nullptr)
    {
        SUNContext_Free(&ctx);
    }
};

/**
 * @brief Function to map dictionary key-words to sundials RKButcher tableau.
 * @param key The key, name, of the explicit Runge-Kutta method to use.
 * @return The id of the assocaied Sundails RK Butcher tableau.
 */
ARKODE_ERKTableID stringToERKTable(const std::string& key)
{
    if (key == "Forward Euler") return ARKODE_FORWARD_EULER_1_1;
    if (key == "Heun")
    {
        NF_ERROR_EXIT("Currently unsupported until field time step-stage indexing resolved.");
        return ARKODE_HEUN_EULER_2_1_2;
    }
    if (key == "Midpoint")
    {
        NF_ERROR_EXIT("Currently unsupported until field time step-stage indexing resolved.");
        return ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2;
    }
    NF_ERROR_EXIT("Unsupported Runge-Kutta time inteation method selectied: " + key);
    return ARKODE_ERK_NONE; // avoids compiler warnings.
}

/**
 * @brief Converts a NeoFOAM Field to a SUNDIALS N_Vector
 * @tparam ValueType The data type of the field elements (e.g., double, float)
 * @param[in] field Source NeoFOAM Field containing the data to be copied
 * @param[out] vector Destination SUNDIALS N_Vector to receive the field data
 * @warning Assumes everything is correctly initialised, sized, correct executore etc.
 */
template<typename ValueType>
void fieldToNVector(const NeoFOAM::Field<ValueType>& field, N_Vector& vector)
{
    // Load the current solution for temporal integration
    sunrealtype* vectData = N_VGetArrayPointer(vector);
    NeoFOAM::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { vectData[i] = field[i]; }
    );
};

/**
 * @brief Converts a SUNDIALS N_Vector back to a NeoFOAM Field
 * @tparam ValueType The data type of the field elements (e.g., double, float)
 * @param[in] vector Source SUNDIALS N_Vector containing the data to be copied
 * @param[out] field Destination NeoFOAM Field to receive the vector data
 * @warning Assumes everything is correctly initialised, sized, correct executore etc.
 */
template<typename ValueType>
void NVectorToField(const N_Vector& vector, NeoFOAM::Field<ValueType>& field)
{
    sunrealtype* vectData = N_VGetArrayPointer(vector);
    ValueType* fieldData = field.data();
    NeoFOAM::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { fieldData[i] = vectData[i]; }
    );
};

/**
 * @brief Performs an iteration/stage of explicit Runge-Kutta with sundails and an expression.
 * @param t The current value of the independent variable
 * @param y The current value of the dependent variable vector
 * @param ydont The output vector that forms [a portion of] the ODE RHS, f(t, y).
 * @param user_data he user_data pointer that was passed to ARKodeSetUserData().
 *
 * @note https://sundials.readthedocs.io/en/latest/arkode/Usage/User_supplied.html#c.ARKRhsFn
 *
 * @details This is our implementation of the RHS of explicit spacital integration, to be integrated
 * in time. In our case user_data is a unique_ptr to an expression. In this function a 'working
 * source' vector is created and parsed to the explicitOperation, which should contain the field
 * variable at the start of the time step. Currently 'multi-stage RK' is not supported until y
 * can be copied to this field.
 */
template<typename SolutionFieldType>
int explicitRKSolve(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    (void)(t); // removes compiler warnings about unused.

    // Pointer wrangling
    NeoFOAM::dsl::Expression* PDEExpre = reinterpret_cast<NeoFOAM::dsl::Expression*>(user_data);
    sunrealtype* ydotarray = N_VGetArrayPointer(ydot);
    sunrealtype* yarray = N_VGetArrayPointer(y);

    if (ydotarray == nullptr || yarray == nullptr || PDEExpre == nullptr)
    {
        std::cerr << NF_ERROR_MESSAGE("Failed to dereference pointers in sundails.");
        return -1;
    }

    // Copy initial value from y to source.
    NeoFOAM::Field<NeoFOAM::scalar> source(PDEExpre->exec(), 1, 0.0);
    source = PDEExpre->explicitOperation(source); // compute spacial
    if (std::holds_alternative<NeoFOAM::GPUExecutor>(PDEExpre->exec()))
    {
        Kokkos::fence();
    }
    NeoFOAM::sundials::fieldToNVector(source, ydot); // assign rhs to ydot.
    return 0;
}

}
