// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <cvode/cvode.h>               // prototypes for CVODE fcts., consts.
#include <nvector/nvector_serial.h>    // serial N_Vector types, fcts., macros
#include <sunmatrix/sunmatrix_dense.h> // access to dense SUNMatrix
#include <sunlinsol/sunlinsol_dense.h> // access to dense SUNLinearSolver
#include <sundials/sundials_types.h>   // definition of type sunrealtype


// Function to compute the right-hand sides of the ODE system
int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    sunrealtype k = *(sunrealtype*)user_data; // cast user_data to sunrealtype
    sunrealtype y_val = NV_Ith_S(y, 0);       // current value of y

    NV_Ith_S(ydot, 0) = -k * y_val; // dy/dt = -k * y
    return 0;                       // return with success
}


TEST_CASE("SUNDIALS CVODE solver")
{
    SUNContext sunctx;
    SUNContext_Create(SUN_COMM_NULL, &sunctx);

    sunrealtype T0 = 0.0;      // initial time
    sunrealtype T1 = 1.0;      // final time
    sunrealtype dt = 0.01;     // time step
    sunrealtype k = 100;       // decay constant
    sunrealtype reltol = 1e-8; // relative tolerance
    sunrealtype abstol = 1e-8; // absolute tolerance

    // Initial condition: y(0) = 1.0
    N_Vector y = N_VNew_Serial(1, sunctx);
    NV_Ith_S(y, 0) = 1.0;

    // Create CVODE object
    void* cvode_mem = CVodeCreate(CV_BDF, sunctx);
    CVodeInit(cvode_mem, f, T0, y);
    CVodeSStolerances(cvode_mem, reltol, abstol);
    CVodeSetUserData(cvode_mem, &k);

    // Use a dense SUNLinearSolver
    SUNMatrix A = SUNDenseMatrix(1, 1, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);
    CVodeSetLinearSolver(cvode_mem, LS, A);

    // Integrate over each time step
    sunrealtype t = T0;
    sunrealtype tout = T0 + dt;
    sunrealtype finalY = 0.0;
    while (tout <= T1)
    {
        if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) == CV_SUCCESS)
        {
            std::cout << "At time t = " << t << ", y = " << NV_Ith_S(y, 0) << std::endl;
            tout += dt;
            finalY = NV_Ith_S(y, 0);
        }
        else
        {
            std::cerr << "Solver failure." << std::endl;
            break;
        }
    }

    // Free resources
    N_VDestroy(y);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);
    CVodeFree(&cvode_mem);
    REQUIRE(finalY == Catch::Approx(0.0).margin(1e-8));
}
