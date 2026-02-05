#include "NeoFOAM/auxiliary/profiling.hpp"
#include "NeoN/NeoN.hpp"
// #include <Kokkos_Profiling.hpp>

namespace NeoFOAM::Profiling {

Region::Region(const char* name) noexcept {
  Kokkos::Profiling::pushRegion(name);
}

Region::~Region() noexcept {
  Kokkos::Profiling::popRegion();
}

}