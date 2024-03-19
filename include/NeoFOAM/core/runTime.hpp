// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once


#include "NeoFOAM/primitives/scalar.hpp"

#include "spdlog/spdlog.h"

#include <string>

namespace NeoFOAM
{

struct LoggerWrapper
{

    template<typename... T>
    void info(T&&... args)
    {
        spdlog::info(std::forward<T>(args)...);
    }

    template<typename... T>
    void error(T&&... args)
    {
        spdlog::error(std::forward<T>(args)...);
    }
};

/**
 * @class runTime
 * @brief A class representing the NeoFOAM runTime
 *
 * This class gives access to typical runTime functions, eg whether the main time
 * loop has finished, directory structure, comandline argument handing, object registry.
 *
 * @ingroup runTime
 */
class runTime
{

protected:

    runTime(int argc, char* argv[]) : deltaT_(1.0),
                                      timeCtr_(0),
                                      time_(0) {
                                      };

public:

    /* Checks whether the specified case is a valid Foam case */
    [[nodiscard]] bool checkRootCase() const { return true; };

    [[nodiscard]] static runTime initialize(
        int argc, char* argv[]
    )
    {
        return runTime(argc, argv);
    };

    bool loop()
    {

        if (timeCtr_ == 0)
        {
            this->getLogger().info("Starting time loop");
        }

        time_ += deltaT_;
        timeCtr_++;
        return 10.0 >= time_;
    };

    [[nodiscard]] std::string timeName() { return std::to_string(time_); }


    void printExecutionTime()
    {

        this->getLogger().info("ExecutionTime {} ClockTime {}", 0.0, 0.0);
    }

    void finalize() {}

    LoggerWrapper getLogger() { return LoggerWrapper {}; }

private:

    // counter for the current time step
    scalar deltaT_;

    // counter for the current time step
    label timeCtr_;

    // current simulation time
    scalar time_;
};
}
