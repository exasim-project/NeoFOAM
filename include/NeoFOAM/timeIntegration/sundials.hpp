// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <concepts>
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

/**
 * Custom deleter for SUNContext_ shared pointers, frees the context if last context.
 * @param[in] ctx Pointer to the SUNContext_ to be freed, can be nullptr.
 */
auto SUN_CONTEXT_DELETER = [](SUNContext_* ctx)
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
 * FIX ME
 * @tparam ValueType The data type of the field elements (e.g., double, float)
 * @param[in] field Source NeoFOAM Field containing the data to be copied
 * @param[out] vector Destination SUNDIALS N_Vector to receive the field data
 * @warning Assumes everything is correctly initialised, sized, correct executore etc.
 */
template<typename SKVectorType, typename ValueType>
void fieldToNVectorImpl(const NeoFOAM::Field<ValueType>& field, N_Vector& vector)
{
    // Load the current solution for temporal integration
    auto view = ::sundials::kokkos::GetVec<SKVectorType>(vector)->View();
    NeoFOAM::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { view(i) = field[i]; }
    );
};

// dispatcer For some reason we using DefaultExecutionSpace for both CPU and GPU
template<typename ValueType>
void fieldToNVector(const NeoFOAM::Field<ValueType>& field, N_Vector& vector)
{
    // CHECK FOR N_Vector on correct space in DEBUG


    if (std::holds_alternative<NeoFOAM::GPUExecutor>(field.exec()))
    {
        fieldToNVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>>(
            field, vector
        );
        return;
    }
    if (std::holds_alternative<NeoFOAM::CPUExecutor>(field.exec()))
    {
        fieldToNVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultHostExecutionSpace>>(
            field, vector
        );
        return;
    }
    if (std::holds_alternative<NeoFOAM::SerialExecutor>(field.exec()))
    {
        fieldToNVectorImpl<::sundials::kokkos::Vector<Kokkos::Serial>>(field, vector);
        return;
    }
    NF_ERROR_EXIT("Unsupported NeoFOAM executor for field.");
};

/**
 * @brief Converts a SUNDIALS N_Vector back to a NeoFOAM Field
 * * FIX ME
 * @tparam ValueType The data type of the field elements (e.g., double, float)
 * @param[in] vector Source SUNDIALS N_Vector containing the data to be copied
 * @param[out] field Destination NeoFOAM Field to receive the vector data
 * @warning Assumes everything is correctly initialised, sized, correct executore etc.
 */
template<typename SKVectorType, typename ValueType>
void NVectorToFieldImpl(const N_Vector& vector, NeoFOAM::Field<ValueType>& field)
{
    auto view = ::sundials::kokkos::GetVec<SKVectorType>(vector)->View();
    ValueType* fieldData = field.data();
    NeoFOAM::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { fieldData[i] = view(i); }
    );
};

// dispatcer For some reason we using DefaultExecutionSpace for both CPU and GPU
template<typename ValueType>
void NVectorToField(const N_Vector& vector, NeoFOAM::Field<ValueType>& field)
{
    // CHECK FOR N_Vector on correct space in DEBUG

    if (std::holds_alternative<NeoFOAM::GPUExecutor>(field.exec()))
    {
        NVectorToFieldImpl<::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>>(
            vector, field
        );
        return;
    }
    if (std::holds_alternative<NeoFOAM::CPUExecutor>(field.exec()))
    {
        NVectorToFieldImpl<::sundials::kokkos::Vector<Kokkos::DefaultHostExecutionSpace>>(
            vector, field
        );
        return;
    }
    if (std::holds_alternative<NeoFOAM::SerialExecutor>(field.exec()))
    {
        NVectorToFieldImpl<::sundials::kokkos::Vector<Kokkos::Serial>>(vector, field);
        return;
    }
    NF_ERROR_EXIT("Unsupported NeoFOAM executor for field.");
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
int explicitRKSolve([[maybe_unused]] sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    // Pointer wrangling
    NeoFOAM::dsl::Expression* pdeExpre = reinterpret_cast<NeoFOAM::dsl::Expression*>(user_data);
    sunrealtype* yDotArray = N_VGetArrayPointer(ydot);
    sunrealtype* yArray = N_VGetArrayPointer(y);

    NF_ASSERT(
        yDotArray != nullptr && yArray != nullptr && pdeExpre != nullptr,
        "Failed to dereference pointers in sundails."
    );

    // Copy initial value from y to source.
    NeoFOAM::Field<NeoFOAM::scalar> source(pdeExpre->exec(), 1, 0.0);
    source = pdeExpre->explicitOperation(source); // compute spacial
    if (std::holds_alternative<NeoFOAM::GPUExecutor>(pdeExpre->exec()))
    {
        Kokkos::fence();
    }
    NeoFOAM::sundials::fieldToNVector(source, ydot); // assign rhs to ydot.
    return 0;
}


namespace detail
{
template<typename Vector>
void initNVector(Vector& vec, size_t size, std::shared_ptr<SUNContext_> context)
{
    vec.initNVector(size, context);
}

template<typename Vector>
const N_Vector& NVector(const Vector& vec)
{
    return vec.NVector();
}

template<typename Vector>
N_Vector& NVector(Vector& vec)
{
    return vec.NVector();
}
}

template<typename ValueType>
class SKVectorDefault
{
public:

    using KVector = ::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>;

    SKVectorDefault() = default;
    ~SKVectorDefault() = default;
    SKVectorDefault(const SKVectorDefault& other) : kvector_(other.kvector_)
    {
        if (other.svector_ != nullptr)
        {
            svector_ = kvector_;
        }
    }

    SKVectorDefault& operator=(const SKVectorDefault& other)
    {
        if (this != &other)
        {
            kvector_ = other.kvector_;
            if (other.svector_ != nullptr)
            {
                svector_ = kvector_;
            }
            else
            {
                svector_ = nullptr;
            }
        }
        return *this;
    }
    SKVectorDefault(SKVectorDefault&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::exchange(other.svector_, nullptr))
    {}
    SKVectorDefault& operator=(SKVectorDefault&& other) noexcept
    {
        if (this != &other)
        {
            kvector_ = std::move(other.kvector_);
            svector_ = std::exchange(other.svector_, nullptr);
        }
        return *this;
    }

    void initNVector(size_t size, std::shared_ptr<SUNContext_> context)
    {
        kvector_ = KVector(size, context.get());
        svector_ = kvector_;
    };
    const N_Vector& NVector() const { return svector_; };
    N_Vector& NVector() { return svector_; };

private:

    KVector kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vector svector_ {nullptr};
};

template<typename ValueType>
class SKVectorSerial
{
public:

    SKVectorSerial() = default;
    ~SKVectorSerial() = default;
    SKVectorSerial(const SKVectorSerial& other) : kvector_(other.kvector_)
    {
        if (other.svector_ != nullptr)
        {
            svector_ = kvector_;
        }
    }

    SKVectorSerial& operator=(const SKVectorSerial& other)
    {
        if (this != &other)
        {
            kvector_ = other.kvector_;
            if (other.svector_ != nullptr)
            {
                svector_ = kvector_;
            }
            else
            {
                svector_ = nullptr;
            }
        }
        return *this;
    }
    SKVectorSerial(SKVectorSerial&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::exchange(other.svector_, nullptr))
    {}
    SKVectorSerial& operator=(SKVectorSerial&& other) noexcept
    {
        if (this != &other)
        {
            kvector_ = std::move(other.kvector_);
            svector_ = std::exchange(other.svector_, nullptr);
        }
        return *this;
    }

    using KVector = ::sundials::kokkos::Vector<Kokkos::Serial>;
    void initNVector(size_t size, std::shared_ptr<SUNContext_> context)
    {
        kvector_ = KVector(size, context.get());
        svector_ = kvector_;
    };
    const N_Vector& NVector() const { return svector_; };
    N_Vector& NVector() { return svector_; };

private:

    KVector kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vector svector_ {nullptr};
};

// base class
template<typename ValueType>
class SKVector
{
public:

    using SKDefaultVector = SKVectorDefault<ValueType>;
    using SKSerialVector = SKVectorSerial<ValueType>;

    using SKVectorVariant = std::variant<SKDefaultVector, SKSerialVector>;

    SKVector() : vector_(SKDefaultVector {}) {};
    ~SKVector() = default;
    SKVector(const SKVector&) = default;
    SKVector& operator=(const SKVector&) = default;
    SKVector(SKVector&&) noexcept = default;
    SKVector& operator=(SKVector&&) noexcept = default;


    explicit SKVector(const NeoFOAM::Executor& exec)
    {
        if (std::holds_alternative<NeoFOAM::GPUExecutor>(exec))
        {
            vector_.template emplace<SKDefaultVector>();
        }
        if (std::holds_alternative<NeoFOAM::CPUExecutor>(exec))
        {
            vector_.template emplace<SKDefaultVector>();
        }
        if (std::holds_alternative<NeoFOAM::SerialExecutor>(exec))
        {
            vector_.template emplace<SKSerialVector>();
        }

        NF_ERROR_EXIT(
            "Unsupported NeoFOAM executor "
            << std::visit([](const auto& e) { return e.name(); }, exec) << "."
        );
    }

    void initNVector(size_t size, std::shared_ptr<SUNContext_> context)
    {
        std::visit(
            [size, &context](auto& vec) { detail::initNVector(vec, size, context); }, vector_
        );
    }

    const N_Vector& NVector() const
    {
        return std::visit(
            [](const auto& vec) -> const N_Vector& { return detail::NVector(vec); }, vector_
        );
    }

    N_Vector& NVector()
    {
        return std::visit([](auto& vec) -> N_Vector& { return detail::NVector(vec); }, vector_);
    }

    const SKVectorVariant& variant() const { return vector_; }
    SKVectorVariant& variant() { return vector_; }

private:

    SKVectorVariant vector_;
};
}
