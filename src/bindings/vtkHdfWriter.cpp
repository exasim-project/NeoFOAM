// SPDX-FileCopyrightText: 2026 NeoFOAM authors
// SPDX-License-Identifier: GPL-3.0-or-later

// The private module neofoam._vtkhdf: one transient VTKHDF 2.1 UnstructuredGrid file, written by
// every MPI rank directly with collective HDF5 transfers. Its public face, which validates all
// input on every rank first and documents the file layout, is the Python class
// neofoam.postprocess.writers.vtkhdf.VtkHdfWriter.

#include <hdf5.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

template<class Scalar>
using Array = nb::ndarray<const Scalar, nb::c_contig>;

using Field = std::tuple<std::string, Array<double>, Dims>; // name, rows, agreed trailing shape

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

Handle createGroup(hid_t parent, const char* name)
{
    return Handle(
        H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Gclose,
        std::string("create group ") + name
    );
}

Handle openGroup(hid_t parent, const char* name)
{
    return Handle(H5Gopen2(parent, name, H5P_DEFAULT), H5Gclose, std::string("open group ") + name);
}

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

struct DataType
{
    hid_t file;
    hid_t memory;
};

DataType realType() { return {H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE}; }
DataType integerType() { return {H5T_STD_I64LE, H5T_NATIVE_INT64}; }
DataType byteType() { return {H5T_STD_U8LE, H5T_NATIVE_UINT8}; }

// A block of rows of one dataset: this rank's `count` rows go to row `first` of the `total` rows.
struct Rows
{
    DataType type;
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

Handle simpleSpace(const Dims& shape, const hsize_t* maxShape = nullptr)
{
    return Handle(
        H5Screate_simple(static_cast<int>(shape.size()), shape.data(), maxShape),
        H5Sclose,
        "dataspace"
    );
}

// Every rank calls it, also one without rows: the transfer is collective.
void writeRows(hid_t dataset, hid_t transfer, hsize_t fileRow, const Rows& rows)
{
    static const char nothing = 0;
    const Dims shape = withRows(std::max<hsize_t>(rows.count, 1), rows.trailing);
    Handle memorySpace = simpleSpace(shape);
    Handle fileSpace(H5Dget_space(dataset), H5Sclose, "dataset space");
    if (rows.count == 0)
    {
        checked(H5Sselect_none(memorySpace.get()), "empty memory selection");
        checked(H5Sselect_none(fileSpace.get()), "empty file selection");
    }
    else
    {
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
    }
    const void* data = rows.count == 0 ? &nothing : rows.data;
    checked(
        H5Dwrite(dataset, rows.type.memory, memorySpace.get(), fileSpace.get(), transfer, data),
        "collective write"
    );
}

void writeFixed(hid_t group, const char* name, hid_t transfer, const Rows& rows)
{
    Handle space = simpleSpace(withRows(rows.total, rows.trailing));
    Handle dataset(
        H5Dcreate2(group, name, rows.type.file, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Dclose,
        std::string("create dataset ") + name
    );
    writeRows(dataset.get(), transfer, rows.first, rows);
}

Handle createAppendable(hid_t group, const std::string& name, const Rows& rows)
{
    constexpr hsize_t minChunkRows = 64;
    constexpr hsize_t maxChunkBytes = 1024 * 1024;
    hsize_t rowBytes = H5Tget_size(rows.type.file);
    for (const hsize_t components : rows.trailing)
    {
        rowBytes *= components; // never 0: the Python writer rejects such a field
    }
    const hsize_t chunkRowLimit = std::max<hsize_t>(maxChunkBytes / rowBytes, 1);
    const Dims chunk =
        withRows(std::min(std::max(rows.total, minChunkRows), chunkRowLimit), rows.trailing);
    const Dims maxShape = withRows(H5S_UNLIMITED, rows.trailing);
    Handle space = simpleSpace(withRows(rows.total, rows.trailing), maxShape.data());
    Handle creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "dataset properties");
    checked(H5Pset_chunk(creation.get(), static_cast<int>(chunk.size()), chunk.data()), "chunking");
    // Every appended row is written right away: filling it first would write it twice.
    checked(H5Pset_fill_time(creation.get(), H5D_FILL_TIME_NEVER), "fill time");
    return Handle(
        H5Dcreate2(
            group,
            name.c_str(),
            rows.type.file,
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

// Append `rows` to the resizable dataset group/name, created on first use; its earlier row count.
hsize_t appendRows(hid_t group, const std::string& name, hid_t transfer, const Rows& rows)
{
    if (checked(H5Lexists(group, name.c_str(), H5P_DEFAULT), "look up " + name) == 0)
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

// One value is a scalar attribute, several a 1-D one; an existing attribute is overwritten.
void setIntAttribute(hid_t object, const char* name, const std::vector<std::int64_t>& values)
{
    const std::string label = std::string("attribute ") + name;
    if (checked(H5Aexists(object, name), label) == 0)
    {
        const hsize_t size = values.size();
        Handle space(
            values.size() == 1 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &size, nullptr),
            H5Sclose,
            label
        );
        Handle created(
            H5Acreate2(object, name, H5T_STD_I64LE, space.get(), H5P_DEFAULT, H5P_DEFAULT),
            H5Aclose,
            label
        );
    }
    Handle attribute(H5Aopen(object, name, H5P_DEFAULT), H5Aclose, label);
    checked(H5Awrite(attribute.get(), H5T_NATIVE_INT64, values.data()), label);
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

class VtkHdfWriter
{
public:

    VtkHdfWriter(
        std::string path,
        const Array<double>& points,
        const Array<std::int64_t>& connectivity,
        const Array<std::int64_t>& offsets,
        const Array<std::uint8_t>& cellTypes,
        std::optional<int> fortranComm
    )
        : path_(std::move(path))
        , ranks_(fortranComm)
    {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); // errors become exceptions, not stderr
        local_ = {
            static_cast<std::int64_t>(points.shape(0)),
            static_cast<std::int64_t>(cellTypes.shape(0)),
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
            Handle root = createGroup(file.get(), "VTKHDF");
            setIntAttribute(root.get(), "Version", {2, 1});
            writeType(root.get());
            writePartSizes(root.get(), transfer.get());
            writeFixed(root.get(), "Points", transfer.get(), meshRows(points, 0, realType()));
            writeFixed(
                root.get(),
                "Connectivity",
                transfer.get(),
                meshRows(connectivity, 2, integerType())
            );
            writeFixed(root.get(), "Offsets", transfer.get(), meshRows(offsets, 3, integerType()));
            writeFixed(root.get(), "Types", transfer.get(), meshRows(cellTypes, 1, byteType()));
            createGroup(root.get(), "PointData");
            createGroup(root.get(), "CellData");
            Handle steps = createGroup(root.get(), "Steps");
            setIntAttribute(steps.get(), "NSteps", {0});
            createGroup(steps.get(), "PointDataOffsets");
            createGroup(steps.get(), "CellDataOffsets");
        }
        file.close("close " + path_);
    }

    // The fields come validated and sorted by name, the same on every rank.
    void write(double time, const std::vector<Field>& pointData, const std::vector<Field>& cellData)
    {
        Handle access = ranks_.fileAccess();
        Handle transfer = ranks_.transfer();
        Handle file(H5Fopen(path_.c_str(), H5F_ACC_RDWR, access.get()), H5Fclose, "open " + path_);
        {
            Handle root = openGroup(file.get(), "VTKHDF");
            Handle steps = openGroup(root.get(), "Steps");
            const std::int64_t parts = ranks_.size();
            const std::int64_t zero = 0; // one static mesh: every step reads the same parts
            appendShared(steps.get(), "Values", transfer.get(), realType(), time);
            appendShared(steps.get(), "NumberOfParts", transfer.get(), integerType(), parts);
            for (const char* name : {"PartOffsets", "PointOffsets", "CellOffsets"})
            {
                appendShared(steps.get(), name, transfer.get(), integerType(), zero);
            }
            appendShared(
                steps.get(),
                "ConnectivityIdOffsets",
                transfer.get(),
                integerType(),
                zero,
                Dims {1}
            );
            for (const Field& field : pointData)
            {
                appendField(root.get(), steps.get(), transfer.get(), 0, field);
            }
            for (const Field& field : cellData)
            {
                appendField(root.get(), steps.get(), transfer.get(), 1, field);
            }
            Handle values(H5Dopen2(steps.get(), "Values", H5P_DEFAULT), H5Dclose, "open Values");
            const auto nSteps = static_cast<std::int64_t>(rowsIn(values.get()));
            setIntAttribute(steps.get(), "NSteps", {nSteps});
        }
        file.close("close " + path_);
    }

private:

    template<class Scalar>
    Rows meshRows(const Array<Scalar>& values, std::size_t column, DataType type) const
    {
        Dims trailing;
        for (std::size_t axis = 1; axis < values.ndim(); ++axis)
        {
            trailing.push_back(values.shape(axis));
        }
        return {
            type,
            trailing,
            static_cast<hsize_t>(total_[column]),
            static_cast<hsize_t>(before_[column]),
            static_cast<hsize_t>(local_[column]),
            values.data()
        };
    }

    void writePartSizes(hid_t root, hid_t transfer) const
    {
        const std::array<const char*, 3> names {
            "NumberOfPoints",
            "NumberOfCells",
            "NumberOfConnectivityIds"
        };
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            const auto nParts = static_cast<hsize_t>(ranks_.size());
            const auto rank = static_cast<hsize_t>(ranks_.rank());
            writeFixed(root, names[i], transfer, {integerType(), {}, nParts, rank, 1, &local_[i]});
        }
    }

    // One row every rank agrees on, written by rank 0.
    template<class Scalar>
    void appendShared(
        hid_t group,
        const std::string& name,
        hid_t transfer,
        DataType type,
        const Scalar& value,
        const Dims& trailing = {}
    ) const
    {
        const hsize_t mine = ranks_.rank() == 0 ? 1 : 0;
        appendRows(group, name, transfer, {type, trailing, 1, 0, mine, &value});
    }

    void
    appendField(hid_t root, hid_t steps, hid_t transfer, std::size_t kind, const Field& field) const
    {
        const auto& [name, values, trailing] = field;
        Handle data = openGroup(root, dataGroups[kind]);
        Handle offsets = openGroup(steps, offsetGroups[kind]);
        const Rows rows {
            realType(),
            trailing,
            static_cast<hsize_t>(total_[kind]),
            static_cast<hsize_t>(before_[kind]),
            values.shape(0),
            values.data()
        };
        const auto start = static_cast<std::int64_t>(appendRows(data.get(), name, transfer, rows));
        appendShared(offsets.get(), name, transfer, integerType(), start);
    }

    std::string path_;
    Ranks ranks_;
    Counts local_ {};
    Counts before_ {};
    Counts total_ {};
};

} // namespace

NB_MODULE(_vtkhdf, m)
{
    nb::class_<VtkHdfWriter>(m, "VtkHdfWriter")
        .def(
            nb::init<
                std::string,
                const Array<double>&,
                const Array<std::int64_t>&,
                const Array<std::int64_t>&,
                const Array<std::uint8_t>&,
                std::optional<int>>(),
            "path"_a,
            "points"_a,
            "connectivity"_a,
            "offsets"_a,
            "cell_types"_a,
            "comm"_a.none()
        )
        .def("write", &VtkHdfWriter::write, "time"_a, "point_data"_a, "cell_data"_a);
}
