#pragma once

namespace NeoFOAM::Profiling {

struct Region {
  explicit Region(const char* name) noexcept;
  ~Region() noexcept;

  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
};

} // namespace NeoFOAM::Profiling