// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <string>

const std::string nl = "\n";

namespace NeoFOAM {

using scalar = float;
using word = std::string;

class argList {

public:
  argList(int argc, char *argv[]){};

  [[nodiscard]] bool checkRootCase() const { return true; };
};

class Time {
public:
  const static word controlDictName;

  Time() : time_(0.){};

  Time(const word, const argList) : time_(0.){};

  [[nodiscard]] word timeName() { return std::to_string(time_); }

  [[nodiscard]] bool loop() {
    time_ += 1.;
    return 10.0 >= time_;
  };

  std::ostream &printExecutionTime(std::ostream &os) const { return os; };

private:
  scalar time_;
};

const word Time::controlDictName = "system/controlDict";

} // namespace NeoFOAM
