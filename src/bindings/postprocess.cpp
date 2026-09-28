// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <optional>
#include <type_traits>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

// NeoN headers
#include "NeoN/NeoN.hpp"

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vec3;

namespace NeoFOAM::bindings
{

// Named (not anonymous) so the device-kernel names of these post-processing kernels stay unique
// across translation units, following the convention of the turbulence-model kernels.
namespace postProcessDetail
{

//! The sentinel an aggregator writes for a group with no active element (``Foam::GREAT``).
constexpr scalar GREAT = 1.0e15;

// Host array shapes the numpy front doors accept. localIdx is the index type of NeoN's
// LabelVector, so a mask/group array from Python must have that dtype (nanobind converts a
// differently typed array in its second overload-resolution pass).
using HostScalars = nb::ndarray<const scalar, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using HostVec3s = nb::ndarray<const scalar, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using HostLabels = nb::ndarray<const localIdx, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template<typename ValueType>
ValueType uniformValue(scalar value)
{
    if constexpr (std::is_same_v<ValueType, Vec3>)
    {
        return Vec3(value);
    }
    else
    {
        return value;
    }
}

// nvcc rejects an extended __host__ __device__ lambda that first-captures a variable inside an
// `if constexpr` branch, so every scalar/Vec3 and max/min split lives in one of these device
// helpers and the kernels below run a single, component-agnostic loop.
template<typename ValueType>
struct ComponentAccess;

template<>
struct ComponentAccess<scalar>
{
    static constexpr NeoN::size_t count = 1;

    KOKKOS_INLINE_FUNCTION
    static scalar read(const scalar& value, NeoN::size_t) { return value; }

    KOKKOS_INLINE_FUNCTION
    static scalar* at(scalar* value, NeoN::size_t) { return value; }
};

template<>
struct ComponentAccess<Vec3>
{
    static constexpr NeoN::size_t count = 3;

    KOKKOS_INLINE_FUNCTION
    static scalar read(const Vec3& value, NeoN::size_t component) { return value[component]; }

    KOKKOS_INLINE_FUNCTION
    static scalar* at(Vec3* value, NeoN::size_t component) { return &(*value)[component]; }
};

template<bool IsMax>
struct AtomicExtremum
{
    KOKKOS_INLINE_FUNCTION
    static void apply(scalar* target, scalar value)
    {
        if constexpr (IsMax)
        {
            Kokkos::atomic_max(target, value);
        }
        else
        {
            Kokkos::atomic_min(target, value);
        }
    }
};

template<bool IsAnd>
struct MaskCombine
{
    KOKKOS_INLINE_FUNCTION
    static localIdx apply(localIdx a, localIdx b)
    {
        if constexpr (IsAnd)
        {
            return (a != 0 && b != 0) ? 1 : 0;
        }
        else
        {
            return (a != 0 || b != 0) ? 1 : 0;
        }
    }
};

// -------------------------------------------------------------------------------------------
// Kernels — one implementation per function, running on the executor of the inbound vector.
// -------------------------------------------------------------------------------------------

template<typename ValueType>
NeoN::Vector<ValueType> groupSum(
    const NeoN::Vector<ValueType>& values,
    localIdx nGroups,
    const std::optional<NeoN::Vector<localIdx>>& mask,
    const std::optional<NeoN::Vector<localIdx>>& group,
    const std::optional<NeoN::Vector<scalar>>& scaling
)
{
    NeoN::Vector<ValueType> result(values.exec(), nGroups, uniformValue<ValueType>(0.0));

    const ValueType* valuesPtr = values.data();
    ValueType* resultPtr = result.data();
    const localIdx* maskPtr = mask ? mask->data() : nullptr;
    const localIdx* groupPtr = group ? group->data() : nullptr;
    const scalar* scalingPtr = scaling ? scaling->data() : nullptr;

    using Access = ComponentAccess<ValueType>;

    NeoN::parallelFor(
        values.exec(),
        {0, values.size()},
        NEON_LAMBDA(const localIdx i) {
            const localIdx g = (groupPtr != nullptr) ? groupPtr[i] : 0;
            // A masked-out element contributes zero rather than being dropped, so the result
            // keeps its shape whatever the mask says.
            const scalar weight = ((maskPtr != nullptr) ? static_cast<scalar>(maskPtr[i]) : 1.0)
                                * ((scalingPtr != nullptr) ? scalingPtr[i] : 1.0);
            for (NeoN::size_t c = 0; c < Access::count; ++c)
            {
                Kokkos::atomic_add(
                    Access::at(resultPtr + g, c),
                    Access::read(valuesPtr[i], c) * weight
                );
            }
        },
        "postprocess::sum"
    );

    return result;
}

template<typename ValueType, bool IsMax>
NeoN::Vector<ValueType> groupExtremum(
    const NeoN::Vector<ValueType>& values,
    localIdx nGroups,
    const std::optional<NeoN::Vector<localIdx>>& mask,
    const std::optional<NeoN::Vector<localIdx>>& group
)
{
    NeoN::Vector<ValueType> result(
        values.exec(),
        nGroups,
        uniformValue<ValueType>(IsMax ? -GREAT : GREAT)
    );

    const ValueType* valuesPtr = values.data();
    ValueType* resultPtr = result.data();
    const localIdx* maskPtr = mask ? mask->data() : nullptr;
    const localIdx* groupPtr = group ? group->data() : nullptr;

    using Access = ComponentAccess<ValueType>;

    NeoN::parallelFor(
        values.exec(),
        {0, values.size()},
        NEON_LAMBDA(const localIdx i) {
            // Masked-out elements are skipped rather than counted as zero, so a selector
            // upstream never drags the extremum towards zero.
            if (maskPtr != nullptr && maskPtr[i] == 0)
            {
                return;
            }
            const localIdx g = (groupPtr != nullptr) ? groupPtr[i] : 0;
            for (NeoN::size_t c = 0; c < Access::count; ++c)
            {
                AtomicExtremum<IsMax>::apply(
                    Access::at(resultPtr + g, c),
                    Access::read(valuesPtr[i], c)
                );
            }
        },
        IsMax ? "postprocess::max" : "postprocess::min"
    );

    return result;
}

NeoN::Vector<scalar> magnitude(const NeoN::Vector<Vec3>& values)
{
    NeoN::Vector<scalar> result(values.exec(), values.size());

    const Vec3* valuesPtr = values.data();
    scalar* resultPtr = result.data();

    NeoN::parallelFor(
        values.exec(),
        {0, values.size()},
        NEON_LAMBDA(const localIdx i) {
            resultPtr[i] = Kokkos::sqrt(
                valuesPtr[i][0] * valuesPtr[i][0] + valuesPtr[i][1] * valuesPtr[i][1]
                + valuesPtr[i][2] * valuesPtr[i][2]
            );
        },
        "postprocess::mag"
    );

    return result;
}

NeoN::Vector<scalar> componentOf(const NeoN::Vector<Vec3>& values, localIdx index)
{
    NeoN::Vector<scalar> result(values.exec(), values.size());

    const Vec3* valuesPtr = values.data();
    scalar* resultPtr = result.data();
    const NeoN::size_t component = static_cast<NeoN::size_t>(index);

    NeoN::parallelFor(
        values.exec(),
        {0, values.size()},
        NEON_LAMBDA(const localIdx i) { resultPtr[i] = valuesPtr[i][component]; },
        "postprocess::component"
    );

    return result;
}

template<typename ValueType>
NeoN::Vector<ValueType> scaled(const NeoN::Vector<ValueType>& values, scalar factor)
{
    NeoN::Vector<ValueType> result(values.exec(), values.size());

    const ValueType* valuesPtr = values.data();
    ValueType* resultPtr = result.data();

    using Access = ComponentAccess<ValueType>;

    NeoN::parallelFor(
        values.exec(),
        {0, values.size()},
        NEON_LAMBDA(const localIdx i) {
            for (NeoN::size_t c = 0; c < Access::count; ++c)
            {
                *Access::at(resultPtr + i, c) = Access::read(valuesPtr[i], c) * factor;
            }
        },
        "postprocess::scale"
    );

    return result;
}

NeoN::Vector<localIdx>
boxMask(const NeoN::Vector<Vec3>& positions, std::array<scalar, 3> lo, std::array<scalar, 3> hi)
{
    NeoN::Vector<localIdx> result(positions.exec(), positions.size());

    const Vec3* positionsPtr = positions.data();
    localIdx* resultPtr = result.data();
    const scalar lo0 = lo[0], lo1 = lo[1], lo2 = lo[2];
    const scalar hi0 = hi[0], hi1 = hi[1], hi2 = hi[2];

    NeoN::parallelFor(
        positions.exec(),
        {0, positions.size()},
        NEON_LAMBDA(const localIdx i) {
            const bool inside = positionsPtr[i][0] >= lo0 && positionsPtr[i][0] <= hi0
                             && positionsPtr[i][1] >= lo1 && positionsPtr[i][1] <= hi1
                             && positionsPtr[i][2] >= lo2 && positionsPtr[i][2] <= hi2;
            resultPtr[i] = inside ? 1 : 0;
        },
        "postprocess::box_mask"
    );

    return result;
}

NeoN::Vector<localIdx>
sphereMask(const NeoN::Vector<Vec3>& positions, std::array<scalar, 3> centre, scalar radius)
{
    NeoN::Vector<localIdx> result(positions.exec(), positions.size());

    const Vec3* positionsPtr = positions.data();
    localIdx* resultPtr = result.data();
    const scalar c0 = centre[0], c1 = centre[1], c2 = centre[2];
    const scalar radiusSquared = radius * radius;

    NeoN::parallelFor(
        positions.exec(),
        {0, positions.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar d0 = positionsPtr[i][0] - c0;
            const scalar d1 = positionsPtr[i][1] - c1;
            const scalar d2 = positionsPtr[i][2] - c2;
            resultPtr[i] = (d0 * d0 + d1 * d1 + d2 * d2 <= radiusSquared) ? 1 : 0;
        },
        "postprocess::sphere_mask"
    );

    return result;
}

NeoN::Vector<localIdx> binIndex(
    const NeoN::Vector<Vec3>& positions,
    std::array<scalar, 3> direction,
    std::array<scalar, 3> origin,
    const std::vector<scalar>& edges
)
{
    NeoN::Vector<localIdx> result(positions.exec(), positions.size());
    // The edges are few, so they travel to the executor as a plain vector and the search below
    // is a linear count rather than a binary search.
    NeoN::Vector<scalar> edgeVector(positions.exec(), edges);

    const scalar length = Kokkos::sqrt(
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]
    );
    // The edges are distances in metres, so the direction is normalised first.
    const scalar n0 = direction[0] / length, n1 = direction[1] / length, n2 = direction[2] / length;
    const scalar o0 = origin[0], o1 = origin[1], o2 = origin[2];

    const Vec3* positionsPtr = positions.data();
    localIdx* resultPtr = result.data();
    const scalar* edgePtr = edgeVector.data();
    const localIdx nEdges = edgeVector.size();

    NeoN::parallelFor(
        positions.exec(),
        {0, positions.size()},
        NEON_LAMBDA(const localIdx i) {
            const scalar distance = (positionsPtr[i][0] - o0) * n0 + (positionsPtr[i][1] - o1) * n1
                                  + (positionsPtr[i][2] - o2) * n2;
            // np.digitize on strictly increasing edges: the index is the number of edges at or
            // below the distance, so the result runs from 0 to len(edges).
            localIdx bin = 0;
            for (localIdx e = 0; e < nEdges; ++e)
            {
                bin += (edgePtr[e] <= distance) ? 1 : 0;
            }
            resultPtr[i] = bin;
        },
        "postprocess::bin_index"
    );

    // parallelFor is asynchronous on a device executor, so the edges must outlive the kernel.
    NeoN::fence(positions.exec());

    return result;
}

template<bool IsAnd>
NeoN::Vector<localIdx>
combineMasks(const NeoN::Vector<localIdx>& a, const NeoN::Vector<localIdx>& b)
{
    NeoN::Vector<localIdx> result(a.exec(), a.size());

    const localIdx* aPtr = a.data();
    const localIdx* bPtr = b.data();
    localIdx* resultPtr = result.data();

    NeoN::parallelFor(
        a.exec(),
        {0, a.size()},
        NEON_LAMBDA(const localIdx i) {
            resultPtr[i] = MaskCombine<IsAnd>::apply(aPtr[i], bPtr[i]);
        },
        IsAnd ? "postprocess::mask_and" : "postprocess::mask_or"
    );

    return result;
}

NeoN::Vector<localIdx> invertMask(const NeoN::Vector<localIdx>& a)
{
    NeoN::Vector<localIdx> result(a.exec(), a.size());

    const localIdx* aPtr = a.data();
    localIdx* resultPtr = result.data();

    NeoN::parallelFor(
        a.exec(),
        {0, a.size()},
        NEON_LAMBDA(const localIdx i) { resultPtr[i] = (aPtr[i] != 0) ? 0 : 1; },
        "postprocess::mask_not"
    );

    return result;
}

// -------------------------------------------------------------------------------------------
// Host <-> device plumbing for the numpy front doors
// -------------------------------------------------------------------------------------------

template<typename ComponentType>
nb::ndarray<nb::numpy, ComponentType, nb::c_contig>
ownedArray(const ComponentType* source, NeoN::size_t count, NeoN::size_t nComponents)
{
    auto* buffer = new ComponentType[count * nComponents];
    std::memcpy(buffer, source, count * nComponents * sizeof(ComponentType));
    nb::capsule owner(
        buffer,
        [](void* data) noexcept { delete[] static_cast<ComponentType*>(data); }
    );
    NeoN::size_t shape[2] = {count, nComponents};
    return nb::ndarray<nb::numpy, ComponentType, nb::c_contig>(
        buffer,
        (nComponents > 1) ? 2 : 1,
        shape,
        owner
    );
}

template<typename ValueType>
auto toNumpy(const NeoN::Vector<ValueType>& values)
{
    const NeoN::Vector<ValueType> host = values.copyToHost();
    const auto count = static_cast<NeoN::size_t>(host.size());
    if constexpr (std::is_same_v<ValueType, Vec3>)
    {
        return ownedArray<scalar>(reinterpret_cast<const scalar*>(host.data()), count, 3);
    }
    else
    {
        return ownedArray<ValueType>(host.data(), count, 1);
    }
}

NeoN::Vector<scalar> toVector(const HostScalars& values)
{
    return NeoN::Vector<scalar>(
        NeoN::SerialExecutor(),
        values.data(),
        static_cast<localIdx>(values.shape(0))
    );
}

NeoN::Vector<Vec3> toVector(const HostVec3s& values)
{
    return NeoN::Vector<Vec3>(
        NeoN::SerialExecutor(),
        reinterpret_cast<const Vec3*>(values.data()),
        static_cast<localIdx>(values.shape(0))
    );
}

NeoN::Vector<localIdx> toVector(const HostLabels& values)
{
    return NeoN::Vector<localIdx>(
        NeoN::SerialExecutor(),
        values.data(),
        static_cast<localIdx>(values.shape(0))
    );
}

std::optional<NeoN::Vector<localIdx>> toVector(const std::optional<HostLabels>& values)
{
    if (!values)
    {
        return std::nullopt;
    }
    return toVector(*values);
}

std::optional<NeoN::Vector<scalar>> toVector(const std::optional<HostScalars>& values)
{
    if (!values)
    {
        return std::nullopt;
    }
    return toVector(*values);
}

} // namespace postProcessDetail

void registerPostProcess(nb::module_& m)
{
    using namespace postProcessDetail;

    nb::module_ post = m.def_submodule(
        "postprocess",
        "Post-processing kernels running on the executor of the inbound vector"
    );

    // -------------------------------------------------------------------
    // Aggregations — local to this rank, the Python caller reduces over MPI
    // -------------------------------------------------------------------
    const char* sumDoc = "Per-group sum of values * mask * scaling, as a host array";

    post.def(
        "sum",
        [](const NeoN::Vector<scalar>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group,
           const std::optional<NeoN::Vector<scalar>>& scaling)
        { return toNumpy(groupSum(values, nGroups, mask, group, scaling)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        nb::kw_only(),
        "scaling"_a = nb::none(),
        sumDoc
    );

    post.def(
        "sum",
        [](const NeoN::Vector<Vec3>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group,
           const std::optional<NeoN::Vector<scalar>>& scaling)
        { return toNumpy(groupSum(values, nGroups, mask, group, scaling)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        nb::kw_only(),
        "scaling"_a = nb::none(),
        sumDoc
    );

    post.def(
        "sum",
        [](const HostScalars& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group,
           const std::optional<HostScalars>& scaling)
        {
            return toNumpy(groupSum(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group),
                toVector(scaling)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        nb::kw_only(),
        "scaling"_a = nb::none(),
        sumDoc
    );

    post.def(
        "sum",
        [](const HostVec3s& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group,
           const std::optional<HostScalars>& scaling)
        {
            return toNumpy(groupSum(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group),
                toVector(scaling)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        nb::kw_only(),
        "scaling"_a = nb::none(),
        sumDoc
    );

    const char* maxDoc = "Per-group maximum over the unmasked elements, -1e15 where a group is "
                         "empty, as a host array";

    post.def(
        "max",
        [](const NeoN::Vector<scalar>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group)
        { return toNumpy(groupExtremum<scalar, true>(values, nGroups, mask, group)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        maxDoc
    );

    post.def(
        "max",
        [](const NeoN::Vector<Vec3>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group)
        { return toNumpy(groupExtremum<Vec3, true>(values, nGroups, mask, group)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        maxDoc
    );

    post.def(
        "max",
        [](const HostScalars& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group)
        {
            return toNumpy(groupExtremum<scalar, true>(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        maxDoc
    );

    post.def(
        "max",
        [](const HostVec3s& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group)
        {
            return toNumpy(groupExtremum<Vec3, true>(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        maxDoc
    );

    const char* minDoc = "Per-group minimum over the unmasked elements, 1e15 where a group is "
                         "empty, as a host array";

    post.def(
        "min",
        [](const NeoN::Vector<scalar>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group)
        { return toNumpy(groupExtremum<scalar, false>(values, nGroups, mask, group)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        minDoc
    );

    post.def(
        "min",
        [](const NeoN::Vector<Vec3>& values,
           localIdx nGroups,
           const std::optional<NeoN::Vector<localIdx>>& mask,
           const std::optional<NeoN::Vector<localIdx>>& group)
        { return toNumpy(groupExtremum<Vec3, false>(values, nGroups, mask, group)); },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        minDoc
    );

    post.def(
        "min",
        [](const HostScalars& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group)
        {
            return toNumpy(groupExtremum<scalar, false>(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        minDoc
    );

    post.def(
        "min",
        [](const HostVec3s& values,
           localIdx nGroups,
           const std::optional<HostLabels>& mask,
           const std::optional<HostLabels>& group)
        {
            return toNumpy(groupExtremum<Vec3, false>(
                toVector(values),
                nGroups,
                toVector(mask),
                toVector(group)
            ));
        },
        "values"_a,
        "n_groups"_a,
        "mask"_a = nb::none(),
        "group"_a = nb::none(),
        minDoc
    );

    // -------------------------------------------------------------------
    // Field functions
    // -------------------------------------------------------------------
    const char* magDoc = "Magnitude of a vector field, (n, 3) values to (n,)";

    post.def(
        "mag",
        [](const NeoN::Vector<Vec3>& values) { return magnitude(values); },
        "values"_a,
        magDoc
    );

    post.def(
        "mag",
        [](const HostVec3s& values) { return toNumpy(magnitude(toVector(values))); },
        "values"_a,
        magDoc
    );

    const char* componentDoc = "One component of a vector field, (n, 3) values to (n,)";

    post.def(
        "component",
        [](const NeoN::Vector<Vec3>& values, localIdx index) { return componentOf(values, index); },
        "values"_a,
        "i"_a,
        componentDoc
    );

    post.def(
        "component",
        [](const HostVec3s& values, localIdx index)
        { return toNumpy(componentOf(toVector(values), index)); },
        "values"_a,
        "i"_a,
        componentDoc
    );

    const char* scaleDoc = "Multiply every value by a factor";

    post.def(
        "scale",
        [](const NeoN::Vector<scalar>& values, scalar factor) { return scaled(values, factor); },
        "values"_a,
        "factor"_a,
        scaleDoc
    );

    post.def(
        "scale",
        [](const NeoN::Vector<Vec3>& values, scalar factor) { return scaled(values, factor); },
        "values"_a,
        "factor"_a,
        scaleDoc
    );

    post.def(
        "scale",
        [](const HostScalars& values, scalar factor)
        { return toNumpy(scaled(toVector(values), factor)); },
        "values"_a,
        "factor"_a,
        scaleDoc
    );

    post.def(
        "scale",
        [](const HostVec3s& values, scalar factor)
        { return toNumpy(scaled(toVector(values), factor)); },
        "values"_a,
        "factor"_a,
        scaleDoc
    );

    // -------------------------------------------------------------------
    // Selectors and binning
    // -------------------------------------------------------------------
    const char* boxDoc = "0/1 mask of the positions inside an axis-aligned box, bounds included";

    post.def(
        "box_mask",
        [](const NeoN::Vector<Vec3>& positions, std::array<scalar, 3> lo, std::array<scalar, 3> hi)
        { return boxMask(positions, lo, hi); },
        "positions"_a,
        "lo"_a,
        "hi"_a,
        boxDoc
    );

    post.def(
        "box_mask",
        [](const HostVec3s& positions, std::array<scalar, 3> lo, std::array<scalar, 3> hi)
        { return toNumpy(boxMask(toVector(positions), lo, hi)); },
        "positions"_a,
        "lo"_a,
        "hi"_a,
        boxDoc
    );

    const char* sphereDoc = "0/1 mask of the positions within radius of centre, boundary included";

    post.def(
        "sphere_mask",
        [](const NeoN::Vector<Vec3>& positions, std::array<scalar, 3> centre, scalar radius)
        { return sphereMask(positions, centre, radius); },
        "positions"_a,
        "centre"_a,
        "radius"_a,
        sphereDoc
    );

    post.def(
        "sphere_mask",
        [](const HostVec3s& positions, std::array<scalar, 3> centre, scalar radius)
        { return toNumpy(sphereMask(toVector(positions), centre, radius)); },
        "positions"_a,
        "centre"_a,
        "radius"_a,
        sphereDoc
    );

    const char* andDoc = "Element-wise AND of two 0/1 masks";

    post.def(
        "mask_and",
        [](const NeoN::Vector<localIdx>& a, const NeoN::Vector<localIdx>& b)
        { return combineMasks<true>(a, b); },
        "a"_a,
        "b"_a,
        andDoc
    );

    post.def(
        "mask_and",
        [](const HostLabels& a, const HostLabels& b)
        { return toNumpy(combineMasks<true>(toVector(a), toVector(b))); },
        "a"_a,
        "b"_a,
        andDoc
    );

    const char* orDoc = "Element-wise OR of two 0/1 masks";

    post.def(
        "mask_or",
        [](const NeoN::Vector<localIdx>& a, const NeoN::Vector<localIdx>& b)
        { return combineMasks<false>(a, b); },
        "a"_a,
        "b"_a,
        orDoc
    );

    post.def(
        "mask_or",
        [](const HostLabels& a, const HostLabels& b)
        { return toNumpy(combineMasks<false>(toVector(a), toVector(b))); },
        "a"_a,
        "b"_a,
        orDoc
    );

    const char* notDoc = "Element-wise negation of a 0/1 mask";

    post.def(
        "mask_not",
        [](const NeoN::Vector<localIdx>& a) { return invertMask(a); },
        "a"_a,
        notDoc
    );

    post.def(
        "mask_not",
        [](const HostLabels& a) { return toNumpy(invertMask(toVector(a))); },
        "a"_a,
        notDoc
    );

    const char* binDoc = "Bin index of every position by its signed distance along a normalised "
                         "direction, over strictly increasing edges (0 to len(edges))";

    post.def(
        "bin_index",
        [](const NeoN::Vector<Vec3>& positions,
           std::array<scalar, 3> direction,
           std::array<scalar, 3> origin,
           const std::vector<scalar>& edges)
        { return binIndex(positions, direction, origin, edges); },
        "positions"_a,
        "direction"_a,
        "origin"_a,
        "edges"_a,
        binDoc
    );

    post.def(
        "bin_index",
        [](const HostVec3s& positions,
           std::array<scalar, 3> direction,
           std::array<scalar, 3> origin,
           const std::vector<scalar>& edges)
        { return toNumpy(binIndex(toVector(positions), direction, origin, edges)); },
        "positions"_a,
        "direction"_a,
        "origin"_a,
        "edges"_a,
        binDoc
    );
}

} // namespace NeoFOAM::bindings
