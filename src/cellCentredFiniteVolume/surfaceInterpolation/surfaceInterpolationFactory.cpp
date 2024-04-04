#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationFactory.hpp"

namespace NeoFOAM
{

inline std::unordered_map<std::string, surfaceInterpolationFactory::factoryFunction>  surfaceInterpolationFactory::classMap;

surfaceInterpolationFactory::surfaceInterpolationFactory()
{}



} // namespace NeoFOAM