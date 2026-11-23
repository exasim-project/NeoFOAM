// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// The private module neofoam._vtkhdf: one transient VTKHDF 2.1 UnstructuredGrid file, written by
// every MPI rank directly with collective HDF5 transfers. Its public face is the Python class
// neofoam.postprocess.writers.vtkhdf.VtkHdfWriter, which documents the file layout.

#include <hdf5.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

using Dims = std::vector<hsize_t>;
using Counts = std::array<std::int64_t, 4>; // points, cells, connectivity ids, offsets

constexpr std::size_t pointKind = 0;
constexpr std::size_t cellKind = 1;
constexpr std::array<const char*, 2> kindLabels {"point", "cell"};
constexpr std::array<const char*, 2> dataGroups {"PointData", "CellData"};
constexpr std::array<const char*, 2> offsetGroups {"PointDataOffsets", "CellDataOffsets"};

herr_t keepInnermostError(unsigned depth, const H5E_error2_t* error, void* message)
{
    if (depth == 0 && error->desc != nullptr)
    {
        *static_cast<std::string*>(message) = error->desc;
    }
    return 0;
}

template<class Result>
Result checked(Result result, const std::string& operation)
{
    if (result < 0)
    {
        std::string detail = "unknown HDF5 error";
        H5Ewalk2(H5E_DEFAULT, H5E_WALK_UPWARD, keepInnermostError, &detail);
        throw std::runtime_error("VTKHDF: " + operation + " failed: " + detail);
    }
    return result;
}

class Handle
{
public:

    Handle(hid_t id, herr_t (*closer)(hid_t), const std::string& operation)
        : id_(checked(id, operation))
        , closer_(closer)
    {}
    Handle(Handle&& other) noexcept
        : id_(std::exchange(other.id_, H5I_INVALID_HID))
        , closer_(other.closer_)
    {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle()
    {
        if (id_ >= 0)
        {
            closer_(id_);
        }
    }

    hid_t get() const { return id_; }

    // Unlike the destructor it reports a failure: closing a file is what flushes it.
    void close(const std::string& operation)
    {
        const hid_t id = std::exchange(id_, H5I_INVALID_HID);
        checked(closer_(id), operation);
    }

private:

    hid_t id_;
    herr_t (*closer_)(hid_t);
};

// The ranks of the writer; without a communicator, one rank and no MPI call at all.
class Ranks
{
public:

    explicit Ranks(std::optional<int> fortranComm)
    {
#ifdef H5_HAVE_PARALLEL
        if (fortranComm)
        {
            comm_ = MPI_Comm_f2c(*fortranComm);
            MPI_Comm_rank(*comm_, &rank_);
            MPI_Comm_size(*comm_, &size_);
        }
#else
        if (fortranComm)
        {
            throw std::runtime_error(
                "VtkHdfWriter got a communicator, but NeoFOAM was built without MPI "
                "(NEOFOAM_WITH_MPI=OFF)"
            );
        }
#endif
    }

    int rank() const { return rank_; }
    int size() const { return size_; }

    std::vector<std::string> allgather(const std::string& local) const
    {
#ifdef H5_HAVE_PARALLEL
        if (comm_)
        {
            int length = static_cast<int>(local.size());
            std::vector<int> lengths(static_cast<std::size_t>(size_));
            MPI_Allgather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, *comm_);
            std::vector<int> displacements(lengths.size(), 0);
            for (std::size_t i = 1; i < lengths.size(); ++i)
            {
                displacements[i] = displacements[i - 1] + lengths[i - 1];
            }
            std::string joined(
                static_cast<std::size_t>(displacements.back() + lengths.back()),
                '\0'
            );
            MPI_Allgatherv(
                local.data(),
                length,
                MPI_CHAR,
                joined.data(),
                lengths.data(),
                displacements.data(),
                MPI_CHAR,
                *comm_
            );
            std::vector<std::string> all;
            for (std::size_t i = 0; i < lengths.size(); ++i)
            {
                all.push_back(joined.substr(
                    static_cast<std::size_t>(displacements[i]),
                    static_cast<std::size_t>(lengths[i])
                ));
            }
            return all;
        }
#endif
        return {local};
    }

    // What the lower ranks hold in total, and what all ranks hold in total.
    std::pair<Counts, Counts> beforeAndTotal(const Counts& local) const
    {
        Counts before {};
        Counts total = local;
#ifdef H5_HAVE_PARALLEL
        if (comm_)
        {
            const int n = static_cast<int>(local.size());
            MPI_Exscan(local.data(), before.data(), n, MPI_INT64_T, MPI_SUM, *comm_);
            if (rank_ == 0)
            {
                before.fill(0); // MPI leaves rank 0's receive buffer undefined
            }
            MPI_Allreduce(local.data(), total.data(), n, MPI_INT64_T, MPI_SUM, *comm_);
        }
#endif
        return {before, total};
    }

    Handle fileAccess() const
    {
        Handle access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose, "file access properties");
#ifdef H5_HAVE_PARALLEL
        if (comm_)
        {
            checked(H5Pset_fapl_mpio(access.get(), *comm_, MPI_INFO_NULL), "MPI file access");
            checked(H5Pset_all_coll_metadata_ops(access.get(), true), "collective metadata reads");
            checked(H5Pset_coll_metadata_write(access.get(), true), "collective metadata writes");
        }
#endif
        return access;
    }

    Handle transfer() const
    {
        Handle properties(H5Pcreate(H5P_DATASET_XFER), H5Pclose, "transfer properties");
#ifdef H5_HAVE_PARALLEL
        if (comm_)
        {
            checked(H5Pset_dxpl_mpio(properties.get(), H5FD_MPIO_COLLECTIVE), "collective mode");
        }
#endif
        return properties;
    }

private:

#ifdef H5_HAVE_PARALLEL
    std::optional<MPI_Comm> comm_;
#endif
    int rank_ = 0;
    int size_ = 1;
};

// A block of rows of one dataset: this rank's `count` rows go to row `first` of the `total` rows.
struct Rows
{
    hid_t fileType;
    hid_t memoryType;
    Dims trailing;
    hsize_t total;
    hsize_t first;
    hsize_t count;
    const void* data;
};

Dims withRows(hsize_t rows, const Dims& trailing)
{
    Dims shape {rows};
    shape.insert(shape.end(), trailing.begin(), trailing.end());
    return shape;
}

// Every rank calls it, also one without rows: the transfer is collective.
void writeRows(hid_t dataset, hid_t transfer, hsize_t fileRow, const Rows& rows)
{
    static const char nothing = 0;
    Dims shape = withRows(rows.count, rows.trailing);
    const int rank = static_cast<int>(shape.size());
    Handle fileSpace(H5Dget_space(dataset), H5Sclose, "dataset space");
    if (rows.count == 0)
    {
        shape[0] = 1;
        Handle memorySpace(H5Screate_simple(rank, shape.data(), nullptr), H5Sclose, "memory space");
        checked(H5Sselect_none(fileSpace.get()), "empty file selection");
        checked(H5Sselect_none(memorySpace.get()), "empty memory selection");
        checked(
            H5Dwrite(
                dataset,
                rows.memoryType,
                memorySpace.get(),
                fileSpace.get(),
                transfer,
                &nothing
            ),
            "empty collective write"
        );
        return;
    }
    Dims start(shape.size(), 0);
    start[0] = fileRow;
    checked(
        H5Sselect_hyperslab(
            fileSpace.get(),
            H5S_SELECT_SET,
            start.data(),
            nullptr,
            shape.data(),
            nullptr
        ),
        "file selection"
    );
    Handle memorySpace(H5Screate_simple(rank, shape.data(), nullptr), H5Sclose, "memory space");
    checked(
        H5Dwrite(dataset, rows.memoryType, memorySpace.get(), fileSpace.get(), transfer, rows.data),
        "collective write"
    );
}

void writeFixed(hid_t group, const char* name, hid_t transfer, const Rows& rows)
{
    const Dims shape = withRows(rows.total, rows.trailing);
    Handle space(
        H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr),
        H5Sclose,
        "dataset space"
    );
    Handle dataset(
        H5Dcreate2(group, name, rows.fileType, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Dclose,
        std::string("create dataset ") + name
    );
    writeRows(dataset.get(), transfer, rows.first, rows);
}

Handle createAppendable(hid_t group, const std::string& name, const Rows& rows)
{
    constexpr hsize_t minChunkRows = 64;
    constexpr hsize_t maxChunkBytes = 1024 * 1024;
    hsize_t rowBytes = H5Tget_size(rows.fileType);
    Dims chunk = withRows(0, rows.trailing);
    for (std::size_t i = 1; i < chunk.size(); ++i)
    {
        rowBytes *= std::max<hsize_t>(chunk[i], 1);
    }
    const hsize_t chunkRowLimit = std::max<hsize_t>(maxChunkBytes / rowBytes, 1);
    chunk[0] = std::min(std::max(rows.total, minChunkRows), chunkRowLimit);

    const Dims shape = withRows(rows.total, rows.trailing);
    const Dims maxShape = withRows(H5S_UNLIMITED, rows.trailing);
    Handle space(
        H5Screate_simple(static_cast<int>(shape.size()), shape.data(), maxShape.data()),
        H5Sclose,
        "dataset space"
    );
    Handle creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "dataset properties");
    checked(H5Pset_chunk(creation.get(), static_cast<int>(chunk.size()), chunk.data()), "chunking");
    // Every appended row is written right away: filling it first would write it twice.
    checked(H5Pset_fill_time(creation.get(), H5D_FILL_TIME_NEVER), "fill time");
    return Handle(
        H5Dcreate2(
            group,
            name.c_str(),
            rows.fileType,
            space.get(),
            H5P_DEFAULT,
            creation.get(),
            H5P_DEFAULT
        ),
        H5Dclose,
        "create dataset " + name
    );
}

hsize_t rowsIn(hid_t dataset)
{
    Handle space(H5Dget_space(dataset), H5Sclose, "dataset space");
    const int rank = checked(H5Sget_simple_extent_ndims(space.get()), "dataset rank");
    Dims shape(static_cast<std::size_t>(rank));
    checked(H5Sget_simple_extent_dims(space.get(), shape.data(), nullptr), "dataset shape");
    return shape[0];
}

bool linkExists(hid_t group, const std::string& name)
{
    return checked(H5Lexists(group, name.c_str(), H5P_DEFAULT), "look up " + name) > 0;
}

// Append `rows` to the resizable dataset group/name, created on first use; its earlier row count.
hsize_t appendRows(hid_t group, const std::string& name, hid_t transfer, const Rows& rows)
{
    if (!linkExists(group, name))
    {
        Handle dataset = createAppendable(group, name, rows);
        writeRows(dataset.get(), transfer, rows.first, rows);
        return 0;
    }
    Handle dataset(H5Dopen2(group, name.c_str(), H5P_DEFAULT), H5Dclose, "open dataset " + name);
    const hsize_t start = rowsIn(dataset.get());
    const Dims extended = withRows(start + rows.total, rows.trailing);
    checked(H5Dset_extent(dataset.get(), extended.data()), "extend dataset " + name);
    writeRows(dataset.get(), transfer, start + rows.first, rows);
    return start;
}

void writeVersion(hid_t root)
{
    const std::array<std::int64_t, 2> version {2, 1};
    const hsize_t size = version.size();
    Handle space(H5Screate_simple(1, &size, nullptr), H5Sclose, "attribute space");
    Handle attribute(
        H5Acreate2(root, "Version", H5T_STD_I64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose,
        "create attribute Version"
    );
    checked(H5Awrite(attribute.get(), H5T_NATIVE_INT64, version.data()), "write Version");
}

void writeType(hid_t root)
{
    const std::string kind = "UnstructuredGrid";
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "string type");
    checked(H5Tset_size(type.get(), kind.size()), "string size");
    checked(H5Tset_strpad(type.get(), H5T_STR_NULLPAD), "string padding");
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "attribute space");
    Handle attribute(
        H5Acreate2(root, "Type", type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose,
        "create attribute Type"
    );
    checked(H5Awrite(attribute.get(), type.get(), kind.data()), "write Type");
}

void createNSteps(hid_t steps)
{
    const std::int64_t none = 0;
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "attribute space");
    Handle attribute(
        H5Acreate2(steps, "NSteps", H5T_STD_I64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT),
        H5Aclose,
        "create attribute NSteps"
    );
    checked(H5Awrite(attribute.get(), H5T_NATIVE_INT64, &none), "write NSteps");
}

void createEmptyGroup(hid_t parent, const char* name)
{
    Handle group(
        H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Gclose,
        std::string("create group ") + name
    );
}

template<class Scalar>
using Array = nb::ndarray<const Scalar, nb::c_contig>;

struct Field
{
    Array<double> values;
    Dims shape; // rows first
};

// What this rank passed for one step. A bad input is kept as text instead of raised: the other
// ranks are about to enter a collective, so every rank has to learn of it there and raise too.
struct Step
{
    double time = 0.0;
    std::array<std::map<std::string, Field>, 2> fields;
    std::string error;
};

Field toField(nb::handle values)
{
    nb::object array = nb::module_::import_("numpy")
                           .attr("asarray")(values, "dtype"_a = "float64", "order"_a = "C");
    Field field {nb::cast<Array<double>>(array), {}};
    if (field.values.ndim() == 0)
    {
        throw std::invalid_argument("a field needs one row per point or cell, got a scalar");
    }
    for (std::size_t axis = 0; axis < field.values.ndim(); ++axis)
    {
        field.shape.push_back(field.values.shape(axis));
    }
    return field;
}

void readFields(nb::handle mapping, std::map<std::string, Field>& fields)
{
    for (nb::handle item : mapping.attr("items")())
    {
        const auto name = nb::cast<std::string>(item[0]);
        // HDF5 would read these as a path or cut the name short.
        if (name.empty() || name == "." || name.find_first_of(std::string("/\0", 2)) != name.npos)
        {
            throw std::invalid_argument("invalid field name '" + name + "'");
        }
        fields.emplace(name, toField(item[1]));
    }
}

Step readStep(nb::handle time, nb::handle pointData, nb::handle cellData)
{
    Step step;
    try
    {
        step.time = nb::cast<double>(nb::float_(time));
        readFields(pointData, step.fields[pointKind]);
        readFields(cellData, step.fields[cellKind]);
    }
    catch (const nb::python_error& error)
    {
        step.error = nb::cast<std::string>(nb::str(error.value()));
    }
    catch (const std::exception& error)
    {
        step.error = error.what();
    }
    return step;
}

// Every rank's field names, which have to be equal byte for byte; the map keeps them sorted, so
// the order a caller's dicts were built in does not matter.
std::string fieldNames(const Step& step)
{
    std::string names;
    for (std::size_t kind = 0; kind < step.fields.size(); ++kind)
    {
        for (const auto& [name, field] : step.fields[kind])
        {
            names += kindLabels[kind];
            names += ' ';
            names += name;
            names += '\n';
        }
    }
    return names;
}

std::string printable(const std::vector<std::string>& perRank)
{
    std::string text;
    for (std::size_t rank = 0; rank < perRank.size(); ++rank)
    {
        text += "\nrank " + std::to_string(rank) + ":\n" + perRank[rank];
    }
    return text;
}

std::string listed(const std::vector<std::int64_t>& values)
{
    std::string text = "[";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        text += (i == 0 ? "" : ", ") + std::to_string(values[i]);
    }
    return text + "]";
}

// One rank's shape of every field in order: rows, expected rows, trailing rank, trailing dims.
std::string encodeShapes(const Step& step, const Counts& local)
{
    std::vector<std::int64_t> encoded;
    for (std::size_t kind = 0; kind < step.fields.size(); ++kind)
    {
        for (const auto& [name, field] : step.fields[kind])
        {
            encoded.push_back(static_cast<std::int64_t>(field.shape[0]));
            encoded.push_back(local[kind]);
            encoded.push_back(static_cast<std::int64_t>(field.shape.size()) - 1);
            encoded.insert(encoded.end(), field.shape.begin() + 1, field.shape.end());
        }
    }
    const auto* bytes = reinterpret_cast<const char*>(encoded.data());
    return std::string(bytes, encoded.size() * sizeof(std::int64_t));
}

std::vector<std::int64_t> decodeShapes(const std::string& bytes)
{
    std::vector<std::int64_t> decoded(bytes.size() / sizeof(std::int64_t));
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(decoded.data()));
    return decoded;
}

// A field's shape as all ranks passed it; `ValueError` for wrong rows or differing components.
Dims agreedTrailing(
    const std::string& label,
    std::vector<std::vector<std::int64_t>>& shapes,
    std::vector<std::size_t>& cursors
)
{
    std::vector<std::int64_t> rows, expected;
    std::optional<Dims> agreed;
    Dims firstRank;
    for (std::size_t rank = 0; rank < shapes.size(); ++rank)
    {
        std::size_t& at = cursors[rank];
        rows.push_back(shapes[rank][at]);
        expected.push_back(shapes[rank][at + 1]);
        const auto nTrailing = static_cast<std::size_t>(shapes[rank][at + 2]);
        const auto begin = shapes[rank].begin() + static_cast<std::ptrdiff_t>(at + 3);
        const Dims trailing(begin, begin + static_cast<std::ptrdiff_t>(nTrailing));
        at += 3 + nTrailing;
        if (rank == 0)
        {
            firstRank = trailing;
        }
        // A rank without rows may pass any shape, e.g. np.zeros(0) for a vector field.
        if (rows.back() > 0 && agreed && *agreed != trailing)
        {
            throw nb::value_error((label + ": the ranks' components differ").c_str());
        }
        if (rows.back() > 0)
        {
            agreed = trailing;
        }
    }
    if (rows != expected)
    {
        const std::string message =
            label + ": " + listed(rows) + " rows per part, " + listed(expected) + " expected";
        throw nb::value_error(message.c_str());
    }
    return agreed.value_or(firstRank);
}

class VtkHdfWriter
{
public:

    VtkHdfWriter(std::string path, const nb::dict& mesh, std::optional<int> fortranComm)
        : path_(std::move(path))
        , ranks_(fortranComm)
    {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); // errors become exceptions, not stderr
        createFile(mesh);
    }

    void write(nb::handle time, nb::handle pointData, nb::handle cellData)
    {
        const Step step = readStep(time, pointData, cellData);
        const auto trailing = validated(step);
        appendStep(step, trailing);
    }

private:

    using TrailingByName = std::array<std::map<std::string, Dims>, 2>;

    // Raises the same ValueError on every rank, before any HDF5 collective can wait for a rank
    // that dropped out.
    TrailingByName validated(const Step& step)
    {
        const std::string status = step.error.empty() ? "N" + fieldNames(step) : "E" + step.error;
        const auto statuses = ranks_.allgather(status);
        std::string errors;
        for (std::size_t rank = 0; rank < statuses.size(); ++rank)
        {
            if (statuses[rank][0] == 'E')
            {
                errors += "rank " + std::to_string(rank) + ": " + statuses[rank].substr(1) + "; ";
            }
        }
        if (!errors.empty())
        {
            throw nb::value_error(("VtkHdfWriter.write: " + errors).c_str());
        }
        if (std::any_of(
                statuses.begin(),
                statuses.end(),
                [&](const std::string& other) { return other != statuses[0]; }
            ))
        {
            std::vector<std::string> names;
            for (const auto& other : statuses)
            {
                names.push_back(other.substr(1));
            }
            const std::string message =
                "VtkHdfWriter.write: the ranks pass different fields:" + printable(names);
            throw nb::value_error(message.c_str());
        }
        return agreedShapes(step);
    }

    TrailingByName agreedShapes(const Step& step)
    {
        std::vector<std::vector<std::int64_t>> shapes;
        for (const auto& bytes : ranks_.allgather(encodeShapes(step, local_)))
        {
            shapes.push_back(decodeShapes(bytes));
        }
        std::vector<std::size_t> cursors(shapes.size(), 0);
        TrailingByName agreed;
        for (std::size_t kind = 0; kind < step.fields.size(); ++kind)
        {
            for (const auto& [name, field] : step.fields[kind])
            {
                const std::string label = std::string(kindLabels[kind]) + " data " + name;
                Dims trailing = agreedTrailing(label, shapes, cursors);
                const auto known = trailing_[kind].find(name);
                if (known != trailing_[kind].end() && known->second != trailing)
                {
                    throw nb::value_error((label + ": components changed between steps").c_str());
                }
                agreed[kind].emplace(name, std::move(trailing));
            }
        }
        for (std::size_t kind = 0; kind < agreed.size(); ++kind)
        {
            trailing_[kind].insert(agreed[kind].begin(), agreed[kind].end());
        }
        return agreed;
    }

    template<class Scalar>
    Rows
    meshRows(const Array<Scalar>& values, std::size_t column, std::pair<hid_t, hid_t> types) const
    {
        Dims trailing;
        for (std::size_t axis = 1; axis < values.ndim(); ++axis)
        {
            trailing.push_back(values.shape(axis));
        }
        return {
            types.first,
            types.second,
            trailing,
            static_cast<hsize_t>(total_[column]),
            static_cast<hsize_t>(before_[column]),
            static_cast<hsize_t>(local_[column]),
            values.data()
        };
    }

    void writeMesh(hid_t root, hid_t transfer, const nb::dict& mesh)
    {
        const auto points = nb::cast<Array<double>>(mesh["points"]);
        const auto connectivity = nb::cast<Array<std::int64_t>>(mesh["connectivity"]);
        const auto offsets = nb::cast<Array<std::int64_t>>(mesh["offsets"]);
        const auto types = nb::cast<Array<std::uint8_t>>(mesh["cell_types"]);
        const std::pair real {H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE};
        const std::pair integer {H5T_STD_I64LE, H5T_NATIVE_INT64};
        const std::pair byte {H5T_STD_U8LE, H5T_NATIVE_UINT8};
        writeFixed(root, "Points", transfer, meshRows(points, 0, real));
        writeFixed(root, "Connectivity", transfer, meshRows(connectivity, 2, integer));
        writeFixed(root, "Offsets", transfer, meshRows(offsets, 3, integer));
        writeFixed(root, "Types", transfer, meshRows(types, 1, byte));
    }

    void writePartSizes(hid_t root, hid_t transfer)
    {
        const std::array<const char*, 3> names {
            "NumberOfPoints",
            "NumberOfCells",
            "NumberOfConnectivityIds"
        };
        const std::array<std::size_t, 3> columns {0, 1, 2};
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            const Rows rows {
                H5T_STD_I64LE,
                H5T_NATIVE_INT64,
                {},
                static_cast<hsize_t>(ranks_.size()),
                static_cast<hsize_t>(ranks_.rank()),
                1,
                &local_[columns[i]]
            };
            writeFixed(root, names[i], transfer, rows);
        }
    }

    void createFile(const nb::dict& mesh)
    {
        const auto points = nb::cast<Array<double>>(mesh["points"]);
        const auto connectivity = nb::cast<Array<std::int64_t>>(mesh["connectivity"]);
        const auto offsets = nb::cast<Array<std::int64_t>>(mesh["offsets"]);
        const auto types = nb::cast<Array<std::uint8_t>>(mesh["cell_types"]);
        local_ = {
            static_cast<std::int64_t>(points.shape(0)),
            static_cast<std::int64_t>(types.shape(0)),
            static_cast<std::int64_t>(connectivity.shape(0)),
            static_cast<std::int64_t>(offsets.shape(0))
        };
        std::tie(before_, total_) = ranks_.beforeAndTotal(local_);

        Handle access = ranks_.fileAccess();
        Handle transfer = ranks_.transfer();
        Handle file(
            H5Fcreate(path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access.get()),
            H5Fclose,
            "create " + path_
        );
        {
            Handle root(
                H5Gcreate2(file.get(), "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Gclose,
                "create group VTKHDF"
            );
            writeVersion(root.get());
            writeType(root.get());
            writePartSizes(root.get(), transfer.get());
            writeMesh(root.get(), transfer.get(), mesh);
            createEmptyGroup(root.get(), "PointData");
            createEmptyGroup(root.get(), "CellData");
            Handle steps(
                H5Gcreate2(root.get(), "Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Gclose,
                "create group Steps"
            );
            createNSteps(steps.get());
            createEmptyGroup(steps.get(), "PointDataOffsets");
            createEmptyGroup(steps.get(), "CellDataOffsets");
        }
        file.close("close " + path_);
    }

    // One row every rank agrees on, written by rank 0.
    template<class Scalar>
    void appendShared(
        hid_t group,
        const std::string& name,
        hid_t transfer,
        std::pair<hid_t, hid_t> types,
        const Scalar& value,
        const Dims& trailing = {}
    ) const
    {
        const hsize_t mine = ranks_.rank() == 0 ? 1 : 0;
        const Rows rows {types.first, types.second, trailing, 1, 0, mine, &value};
        appendRows(group, name, transfer, rows);
    }

    void appendField(
        hid_t root,
        hid_t steps,
        hid_t transfer,
        std::size_t kind,
        const std::pair<const std::string, Field>& entry,
        const Dims& trailing
    ) const
    {
        const auto& [name, field] = entry;
        Handle data(H5Gopen2(root, dataGroups[kind], H5P_DEFAULT), H5Gclose, "open data group");
        Handle offsets(
            H5Gopen2(steps, offsetGroups[kind], H5P_DEFAULT),
            H5Gclose,
            "open offset group"
        );
        std::int64_t start = 0;
        if (linkExists(data.get(), name))
        {
            Handle dataset(
                H5Dopen2(data.get(), name.c_str(), H5P_DEFAULT),
                H5Dclose,
                "open dataset " + name
            );
            start = static_cast<std::int64_t>(rowsIn(dataset.get()));
        }
        appendShared(offsets.get(), name, transfer, {H5T_STD_I64LE, H5T_NATIVE_INT64}, start);
        const Rows rows {
            H5T_IEEE_F64LE,
            H5T_NATIVE_DOUBLE,
            trailing,
            static_cast<hsize_t>(total_[kind]),
            static_cast<hsize_t>(before_[kind]),
            field.shape[0],
            field.values.data()
        };
        appendRows(data.get(), name, transfer, rows);
    }

    void updateNSteps(hid_t steps) const
    {
        Handle values(H5Dopen2(steps, "Values", H5P_DEFAULT), H5Dclose, "open Steps/Values");
        const auto nSteps = static_cast<std::int64_t>(rowsIn(values.get()));
        Handle attribute(H5Aopen(steps, "NSteps", H5P_DEFAULT), H5Aclose, "open NSteps");
        checked(H5Awrite(attribute.get(), H5T_NATIVE_INT64, &nSteps), "write NSteps");
    }

    void appendStep(const Step& step, const TrailingByName& trailing)
    {
        Handle access = ranks_.fileAccess();
        Handle transfer = ranks_.transfer();
        Handle file(H5Fopen(path_.c_str(), H5F_ACC_RDWR, access.get()), H5Fclose, "open " + path_);
        {
            Handle root(H5Gopen2(file.get(), "VTKHDF", H5P_DEFAULT), H5Gclose, "open VTKHDF");
            Handle steps(H5Gopen2(root.get(), "Steps", H5P_DEFAULT), H5Gclose, "open Steps");
            const std::pair real {H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE};
            const std::pair integer {H5T_STD_I64LE, H5T_NATIVE_INT64};
            const std::int64_t parts = ranks_.size();
            const std::int64_t zero = 0; // one static mesh: every step reads the same parts
            appendShared(steps.get(), "Values", transfer.get(), real, step.time);
            appendShared(steps.get(), "NumberOfParts", transfer.get(), integer, parts);
            appendShared(steps.get(), "PartOffsets", transfer.get(), integer, zero);
            appendShared(steps.get(), "PointOffsets", transfer.get(), integer, zero);
            appendShared(steps.get(), "CellOffsets", transfer.get(), integer, zero);
            appendShared(
                steps.get(),
                "ConnectivityIdOffsets",
                transfer.get(),
                integer,
                zero,
                Dims {1}
            );
            for (std::size_t kind = 0; kind < step.fields.size(); ++kind)
            {
                for (const auto& entry : step.fields[kind])
                {
                    appendField(
                        root.get(),
                        steps.get(),
                        transfer.get(),
                        kind,
                        entry,
                        trailing[kind].at(entry.first)
                    );
                }
            }
            updateNSteps(steps.get());
        }
        file.close("close " + path_);
    }

    std::string path_;
    Ranks ranks_;
    Counts local_ {};
    Counts before_ {};
    Counts total_ {};
    TrailingByName trailing_;
};

} // namespace

NB_MODULE(_vtkhdf, m)
{
    nb::class_<VtkHdfWriter>(m, "VtkHdfWriter")
        .def(
            nb::init<std::string, const nb::dict&, std::optional<int>>(),
            "path"_a,
            "mesh"_a,
            "comm"_a.none()
        )
        .def("write", &VtkHdfWriter::write, "time"_a, "point_data"_a, "cell_data"_a);
}
