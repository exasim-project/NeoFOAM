#include "NeoFOAM/cellCentredFiniteVolume/surfaceInterpolation/surfaceInterpolationSelector.hpp"
#include "NeoFOAM/core/Error.hpp"

namespace NeoFOAM
{

// inline std::unordered_map<std::string, surfaceInterpolationFactory::factoryFunction>  surfaceInterpolationFactory::classMap;

// surfaceInterpolationFactory::surfaceInterpolationFactory()
// {}
surfaceInterpolation surfaceInterpolationSelector(std::string interPolMethodName ,const executor& exec, const unstructuredMesh& mesh)
{
    if (interPolMethodName == "upwind")
    {
        return surfaceInterpolation(exec,mesh,std::make_unique<upwind>(exec,mesh));
    }
    else if (interPolMethodName == "linear")
    {
        return surfaceInterpolation(exec,mesh,std::make_unique<linear>(exec,mesh));
    }
    else
    {
        error("not found").exit();
        return surfaceInterpolation(exec,mesh,std::make_unique<upwind>(exec,mesh)); // for the compiler 
    }
}


} // namespace NeoFOAM