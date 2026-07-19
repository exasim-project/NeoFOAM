// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 NeoFOAM authors

#pragma once

// Lightweight, header-only GPU/host memory-footprint probe used to build a *timeline* of
// Umpire-pool consumption across a solver run. The idea: sprinkle NF_MEM_PROBE()/NF_MEM_SCOPE()
// annotations at the interesting points (phase boundaries in the time loop, temporary allocations
// inside the PDE assemble/solve, the kOmegaSST kernels) and, at the end of the run, dump the whole
// timeline to a CSV that can be plotted (seq on the x-axis, current/actual bytes on the y-axis).
// The tag column tells you WHICH annotation each sample came from, so a spike in reserved memory
// is attributable to a specific code region — that is the "bottleneck".
//
// Three numbers are recorded per sample, straight from the Umpire QuickPool:
//   * current   — live bytes currently handed out from the pool (true working-set footprint)
//   * actual    — bytes the pool has reserved from the driver (>= current; grows on fragmentation)
//   * highWater — peak `current` ever seen (monotone; the OOM-relevant number)
//
// Zero overhead when disabled: every entry point short-circuits on a cached env-var check, so the
// annotations can stay in the hot path. Enable at runtime with
//     NEOFOAM_MEM_TIMELINE=1            # turn CSV probing on
//     NEOFOAM_MEM_TIMELINE_FILE=mem.csv # optional; default "memoryTimeline.csv"
//     NEOFOAM_MEM_NVTX=1                # ALSO mirror every probe as an NVTX range/mark (see below)
// The CSV path is a no-op (records zeros) unless NeoN was built with Umpire and CUDA.
//
// NVTX bridge: with NEOFOAM_MEM_NVTX=1, each NF_MEM_SCOPE enter/exit becomes an nvtxRangePush/Pop
// and each NF_MEM_PROBE point becomes an nvtxMark, so the SAME annotations line up against the
// kernels on an Nsight Systems timeline (`nsys profile ...`). The live device-pool bytes at the
// probe are attached as a uint64 NVTX payload, so hovering a region in Nsight shows its footprint.
// This is independent of the CSV timeline -- enable either, both, or neither. The bridge is only
// compiled in when the CUDA-toolkit NVTX header is on the include path (checked via __has_include);
// otherwise NEOFOAM_MEM_NVTX is silently a no-op. No extra link is needed -- the NVTX v3 API is
// header-only and injected by the profiler at runtime.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if NF_WITH_UMPIRE
#include "NeoN/core/memory/umpire.hpp"
#include "umpire/Umpire.hpp"   // get_allocator_records / get_backtrace (allocation-records dump hook)
#endif

// Optional NVTX bridge (header-only; no link). Guarded by __has_include so the probe header stays
// usable in translation units / builds where the CUDA toolkit's NVTX header is not reachable.
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NF_WITH_NVTX 1
#endif
#endif

namespace NeoFOAM
{

/** @brief True iff an Umpire DEVICE memory pool is active (the "DEVICE_POOL" allocator exists).
 *  NeoN only registers it when the run is configured with a pool (allocator=UmpirePool + memPoolSize);
 *  raw Umpire / Kokkos allocators never do. Cached after the first call.
 *
 *  Used to gate the between-/within-phase scratch releases (resize-to-0) added across the solver: with
 *  a pool, resize(0) returns memory to the pool and a later regrow reuses it (cheap); without one it
 *  would be a per-step raw cudaMalloc/cudaFree pair — far costlier than keeping the buffer resident —
 *  so every such release becomes a no-op. */
inline bool umpireDevicePoolActive()
{
    static const bool active = []
    {
#if NF_WITH_UMPIRE && defined(KOKKOS_ENABLE_CUDA)
        try
        {
            (void)NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
            return true;
        }
        catch (...)
        {
            return false;
        }
#else
        return false;
#endif
    }();
    return active;
}

/** @brief One timeline entry: where (tag/phase) and how much (current/actual/highWater) at a
 *  monotonically increasing sequence index. step/time carry the caller's solver context so the
 *  CSV can be correlated with the iteration log. */
struct MemorySample
{
    std::uint64_t seq;            ///< monotonic probe index (timeline x-axis)
    long step;                    ///< solver/timestep index (caller-provided; -1 if unknown)
    double time;                  ///< simulation time (caller-provided)
    const char* tag;             ///< static string naming the probe site
    const char* phase;           ///< "point" | "enter" | "exit"
    std::size_t currentBytes;     ///< live bytes handed out by the device pool
    std::size_t actualBytes;      ///< bytes the pool reserved from the driver
    std::size_t highWaterBytes;   ///< peak live bytes so far
};

/** @brief Static registry + sampler for the memory timeline. All members are static; there is no
 *  instance state beyond the RAII helper MemoryProbeScope below. Header-only and safe across
 *  translation units (C++17 inline function-local statics collapse to a single definition). */
class MemoryProbe
{
public:

    /** @brief True iff NEOFOAM_MEM_TIMELINE is set to a truthy value. Cached after first call so
     *  probe sites cost one predictable branch when disabled. */
    static bool enabled()
    {
        static const bool on = [] {
            const char* e = std::getenv("NEOFOAM_MEM_TIMELINE");
            if (e == nullptr || e[0] == '\0') return false;
            // treat "0"/"off"/"false"/"no" as disabled
            const std::string v(e);
            return !(v == "0" || v == "off" || v == "false" || v == "no");
        }();
        return on;
    }

    /** @brief True iff NEOFOAM_MEM_NVTX is set to a truthy value AND the NVTX header was found at
     *  build time. Cached like enabled(). When on, every probe also emits an NVTX range/mark so the
     *  annotations appear on an Nsight Systems timeline. */
    static bool nvtxEnabled()
    {
        static const bool on = [] {
#if defined(NF_WITH_NVTX)
            const char* e = std::getenv("NEOFOAM_MEM_NVTX");
            if (e == nullptr || e[0] == '\0') return false;
            const std::string v(e);
            return !(v == "0" || v == "off" || v == "false" || v == "no");
#else
            return false;
#endif
        }();
        return on;
    }

    /** @brief True iff NEOFOAM_MEM_ALLOC_RECORDS is truthy AND the build has Umpire. When on, the
     *  probe snapshots the pool's live allocation records at (near-)peak footprint and, at dump time,
     *  writes a ranking of the peak's composition attributed to the ALLOCATING CALL STACK — i.e. the
     *  named fields. Requires an Umpire built with UMPIRE_ENABLE_BACKTRACE (see
     *  scripts/coma/build-nvidia-h200-gcc.sh) and runtime UMPIRE_BACKTRACE=On for the stacks to carry
     *  symbols; without backtrace it still ranks by size but the call-site key is a placeholder. */
    static bool allocRecordsEnabled()
    {
        static const bool on = [] {
#if NF_WITH_UMPIRE
            const char* e = std::getenv("NEOFOAM_MEM_ALLOC_RECORDS");
            if (e == nullptr || e[0] == '\0') return false;
            const std::string v(e);
            return !(v == "0" || v == "off" || v == "false" || v == "no");
#else
            return false;
#endif
        }();
        return on;
    }

    /** @brief Record one point sample. `tag` MUST be a string literal / static-lifetime string —
     *  it is stored by pointer, not copied, to keep the hot path allocation-free. When the NVTX
     *  bridge is enabled the same sample is mirrored as an NVTX range (enter/exit) or mark (point),
     *  carrying the live pool bytes as a payload. */
    static void
    sample(const char* tag, long step = -1, double time = 0.0, const char* phase = "point")
    {
        const bool csv = enabled();
        const bool nvtx = nvtxEnabled();
        const bool records = allocRecordsEnabled();
        if (!csv && !nvtx && !records) return;

        std::size_t cur = 0, act = 0, high = 0;
        queryPool(cur, act, high);

        if (nvtx) emitNvtx(tag, phase, cur);

        // Snapshot the peak's allocation-record composition. Re-captured only when a new footprint
        // exceeds the last snapshot by >5%, so warmup growth triggers a handful of captures and the
        // steady per-timestep sawtooth triggers none — keeping the (symbolizing) cost bounded.
        if (records && cur > 0 && cur > peakRecordBytes() + peakRecordBytes() / 20)
            captureAllocatorRecords(cur, tag);

        if (csv)
        {
            MemorySample s {};
            s.seq = nextSeq();
            s.step = step;
            s.time = time;
            s.tag = tag;
            s.phase = phase;
            s.currentBytes = cur;
            s.actualBytes = act;
            s.highWaterBytes = high;
            timeline().push_back(s);
        }
    }

    /** @brief Write the timeline to `path` as CSV. Called once at end of run. Bytes are emitted
     *  raw (not MB) so the plotting script controls the unit. No-op if nothing was recorded. */
    static void dump(const std::string& path = defaultFile())
    {
        // Independent of the CSV timeline: write the peak allocation-record composition (if captured)
        // to a sibling file, so it is produced even when only NEOFOAM_MEM_ALLOC_RECORDS is enabled.
        dumpAllocatorRecords(path);

        if (timeline().empty()) return;
        std::ofstream os(path);
        os << "seq,step,time,tag,phase,current_bytes,actual_bytes,highwater_bytes\n";
        for (const auto& s : timeline())
        {
            os << s.seq << ',' << s.step << ',' << s.time << ',' << s.tag << ',' << s.phase << ','
               << s.currentBytes << ',' << s.actualBytes << ',' << s.highWaterBytes << '\n';
        }
    }

    /** @brief Path the run dumps to: $NEOFOAM_MEM_TIMELINE_FILE or "memoryTimeline.csv". */
    static std::string defaultFile()
    {
        const char* f = std::getenv("NEOFOAM_MEM_TIMELINE_FILE");
        return (f != nullptr && f[0] != '\0') ? std::string(f) : std::string("memoryTimeline.csv");
    }

    static const std::vector<MemorySample>& samples() { return timeline(); }

private:

    /** @brief Mirror a probe onto the NVTX timeline. "enter" -> nvtxRangePush, "exit" ->
     *  nvtxRangePop, "point" -> nvtxMark. The live pool bytes ride along as a uint64 payload so the
     *  footprint is visible on the range/mark in Nsight. Balanced push/pop per thread because it is
     *  driven by MemoryProbeScope's ctor/dtor. No-op when the NVTX header was not found. */
    static void emitNvtx([[maybe_unused]] const char* tag,
                         [[maybe_unused]] const char* phase,
                         [[maybe_unused]] std::size_t currentBytes)
    {
#if defined(NF_WITH_NVTX)
        if (std::strcmp(phase, "exit") == 0)
        {
            nvtxRangePop();
            return;
        }
        nvtxEventAttributes_t a {};
        a.version = NVTX_VERSION;
        a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.messageType = NVTX_MESSAGE_TYPE_ASCII;
        a.message.ascii = tag;
        a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        a.payload.ullValue = static_cast<std::uint64_t>(currentBytes);
        if (std::strcmp(phase, "enter") == 0)
            nvtxRangePushEx(&a);
        else
            nvtxMarkEx(&a);
#endif
    }

    static void queryPool(std::size_t& cur, std::size_t& act, std::size_t& high)
    {
        cur = act = high = 0;
#if NF_WITH_UMPIRE && defined(KOKKOS_ENABLE_CUDA)
        try
        {
            auto pool = NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
            cur = pool.getCurrentSize();
            act = pool.getActualSize();
            high = pool.getHighWatermark();
        }
        catch (...)
        {
            // Pool not created yet (no device allocation has happened). Leave zeros.
        }
#endif
    }

    // ---- Allocation-record (peak composition) snapshot state -------------------------------------
    // Holds the ranking captured at the highest footprint seen so far; overwritten as the peak grows,
    // written to a sibling file at dump(). Function-local statics so the header stays single-definition.
    static std::string& peakRecordText()
    {
        static std::string t;
        return t;
    }
    static std::size_t& peakRecordBytes()
    {
        static std::size_t b = 0;
        return b;
    }

    /** @brief Snapshot the live allocation records of the device pool and fold them into a ranking
     *  keyed by allocating call stack (or by Umpire allocation name when present), so the blocks that
     *  make up the peak footprint are attributed to the code that created them. Only the largest K
     *  records are symbolized (they dominate the bytes), bounding the cost of stack resolution. */
    static void captureAllocatorRecords([[maybe_unused]] std::size_t curBytes,
                                        [[maybe_unused]] const char* tag)
    {
#if NF_WITH_UMPIRE && defined(KOKKOS_ENABLE_CUDA)
        try
        {
            auto pool = NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
            auto recs = umpire::get_allocator_records(pool);
            if (recs.empty()) return;

            std::size_t total = 0;
            for (const auto& r : recs) total += r.size;
            // Largest-first: the big blocks are what we want named, and symbolizing only the top K
            // keeps stack resolution cheap even with thousands of live allocations.
            std::sort(recs.begin(), recs.end(),
                      [](const auto& a, const auto& b) { return a.size > b.size; });
            constexpr std::size_t K = 256;

            struct Agg
            {
                std::size_t bytes = 0, count = 0, maxsz = 0;
            };
            std::map<std::string, Agg> agg;
            std::size_t covered = 0, n = 0;
            for (const auto& r : recs)
            {
                if (n++ >= K) break;
                // Prefer an Umpire allocation name (from named_allocate) if NeoN ever sets one;
                // otherwise attribute to the allocating call stack (needs UMPIRE_ENABLE_BACKTRACE).
                std::string key =
                    !r.name.empty() ? ("name=" + r.name) : umpire::get_backtrace(r.ptr);
                auto& a = agg[key];
                a.bytes += r.size;
                a.count += 1;
                a.maxsz = std::max(a.maxsz, r.size);
                covered += r.size;
            }

            std::vector<std::pair<std::string, Agg>> ranked(agg.begin(), agg.end());
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });

            std::ostringstream os;
            os << "# NeoFOAM peak device-pool composition (Umpire allocation records)\n"
               << "# trigger tag     : " << (tag != nullptr ? tag : "?") << '\n'
               << "# live at snapshot: " << (curBytes >> 20) << " MB across " << recs.size()
               << " allocations\n"
               << "# top-" << K << " records cover " << (covered >> 20) << " / " << (total >> 20)
               << " MB; ranked by call-site (needs UMPIRE_ENABLE_BACKTRACE + UMPIRE_BACKTRACE=On):\n\n";
            std::size_t shown = 0;
            for (const auto& [key, a] : ranked)
            {
                os << (a.bytes >> 20) << " MB   x" << a.count << "   (largest " << (a.maxsz >> 20)
                   << " MB)\n"
                   << key << "\n\n";
                if (++shown >= 50) break;
            }

            peakRecordText() = os.str();
            peakRecordBytes() = curBytes;
        }
        catch (...)
        {
            // Pool not created yet, or introspection unavailable: leave the last snapshot as-is.
        }
#endif
    }

    /** @brief Write the captured peak composition next to the CSV: <csv-stem>.allocRecords.txt (or
     *  <path>.allocRecords.txt when the path has no .csv suffix). No-op when nothing was captured. */
    static void dumpAllocatorRecords(const std::string& csvPath)
    {
        if (!allocRecordsEnabled() || peakRecordText().empty()) return;
        std::string p = csvPath;
        const std::string suffix = ".csv";
        if (p.size() >= suffix.size() && p.compare(p.size() - suffix.size(), suffix.size(), suffix) == 0)
            p.replace(p.size() - suffix.size(), suffix.size(), ".allocRecords.txt");
        else
            p += ".allocRecords.txt";
        std::ofstream os(p);
        os << peakRecordText();
    }

    static std::vector<MemorySample>& timeline()
    {
        static std::vector<MemorySample> t = [] {
            std::vector<MemorySample> v;
            v.reserve(4096);
            return v;
        }();
        return t;
    }

    static std::uint64_t nextSeq()
    {
        static std::uint64_t s = 0;
        return s++;
    }
};

/** @brief RAII helper that records an "enter" sample on construction and an "exit" sample on
 *  destruction with the SAME tag. The difference in `current_bytes` between the two rows is the
 *  net footprint of the scoped region — i.e. how much a temporary allocation cost. Use via
 *  NF_MEM_SCOPE(...) so it compiles to nothing measurable when probing is disabled. */
class MemoryProbeScope
{
public:

    MemoryProbeScope(const char* tag, long step = -1, double time = 0.0)
        : tag_(tag), step_(step), time_(time),
          active_(MemoryProbe::enabled() || MemoryProbe::nvtxEnabled()
                  || MemoryProbe::allocRecordsEnabled())
    {
        if (active_) MemoryProbe::sample(tag_, step_, time_, "enter");
    }

    ~MemoryProbeScope()
    {
        if (active_) MemoryProbe::sample(tag_, step_, time_, "exit");
    }

    MemoryProbeScope(const MemoryProbeScope&) = delete;
    MemoryProbeScope& operator=(const MemoryProbeScope&) = delete;

private:

    const char* tag_;
    long step_;
    double time_;
    bool active_;
};

} // namespace NeoFOAM

// Point probe: single timeline sample at this line.
#define NF_MEM_PROBE(tag, step, time) ::NeoFOAM::MemoryProbe::sample((tag), (step), (time))

// Scoped probe: enter/exit pair bracketing the enclosing block; the current-bytes delta is the
// region's footprint. Token-pastes a unique variable name so multiple scopes can nest in one block.
#define NF_MEM_SCOPE_CAT2(a, b) a##b
#define NF_MEM_SCOPE_CAT(a, b) NF_MEM_SCOPE_CAT2(a, b)
#define NF_MEM_SCOPE(tag, step, time)                                                              \
    ::NeoFOAM::MemoryProbeScope NF_MEM_SCOPE_CAT(nfMemScope_, __LINE__)((tag), (step), (time))
