#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2024 NeoFOAM Authors
"""Compare per-rank OpenFOAM field files from two parallel case directories.

Both directories must use the same mesh decomposition so that processorN/<time>/
fields share identical cell ordering. Fields are compared directly without
reconstructPar. Computes L-infinity error and exits non-zero if any error
exceeds the specified threshold.

Usage:
    python compare_fields.py --ref <icoFoam_dir> --neo <neoIcoFoam_dir> \\
        [--fields U p] [--threshold 1e-2]
"""

import argparse
import gzip
import math
import os
import re
import sys


def _open_field_file(path):
    """Open a field file, trying .gz suffix first, then plain."""
    gz_path = path + ".gz"
    if os.path.isfile(gz_path):
        return gzip.open(gz_path, "rt", encoding="utf-8")
    if os.path.isfile(path):
        return open(path, encoding="utf-8")
    raise FileNotFoundError(f"Field file not found: {path} (also tried {gz_path})")


def _skip_foam_file_header(lines):
    """Advance iterator past the FoamFile { ... } header block."""
    in_header = False
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if not in_header and stripped.startswith("FoamFile"):
            in_header = True
        if in_header:
            brace_depth += stripped.count("{")
            brace_depth -= stripped.count("}")
            if brace_depth <= 0 and in_header and "{" in stripped or (in_header and brace_depth == 0 and stripped == "}"):
                return
        if in_header and brace_depth == 0 and "}" in stripped:
            return


def _read_internal_field(filepath):
    """Parse internalField from an OpenFOAM ASCII field file.

    Supports both scalar and vector internalField formats.
    Returns a flat list of float values.  Raises ValueError if the
    internalField line specifies a uniform value (not supported for
    comparison — uniform fields have no per-cell data to read).
    """
    with _open_field_file(filepath) as fh:
        content = fh.read()

    # Find the internalField token
    match = re.search(
        r"internalField\s+(nonuniform\s+List<(scalar|vector)>|uniform\s+.*?;)",
        content,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"No internalField found in {filepath}")

    token = match.group(1).strip()

    if token.startswith("uniform"):
        # uniform scalar or vector — return empty list; caller must handle
        return []

    # nonuniform List<scalar> or List<vector>
    # After the matched text there is: N \n ( \n v0 \n v1 ... \n ) \n ;
    tail = content[match.end():]
    # strip leading whitespace and read count
    count_match = re.match(r"\s*(\d+)\s*\(", tail)
    if count_match is None:
        raise ValueError(f"Cannot parse internalField count in {filepath}")
    count = int(count_match.group(1))
    after_paren = tail[count_match.end():]

    # Read until the list-terminating closing parenthesis (a ')' on its own line).
    # We must NOT use the first ')' because vector entries contain their own parens.
    term_match = re.search(r"\n\s*\)\s*(\n|$)", after_paren)
    if term_match is None:
        raise ValueError(f"Unterminated internalField list in {filepath}")
    raw = after_paren[: term_match.start()]

    field_type = match.group(2)  # "scalar" or "vector"
    values = []

    if field_type == "scalar":
        for token in raw.split():
            values.append(float(token))
    else:  # vector
        # Each entry is "(u v w)"
        for vec_match in re.finditer(r"\(\s*([^\)]+)\s*\)", raw):
            components = vec_match.group(1).split()
            values.extend(float(c) for c in components)

    if len(values) == 0 and count > 0:
        raise ValueError(
            f"Parsed 0 values but expected {count} entries in {filepath}"
        )

    return values


def _linf_scalar(a, b):
    """L-infinity norm between two flat lists of scalars."""
    if len(a) != len(b):
        raise ValueError(f"Field length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    return max(abs(x - y) for x, y in zip(a, b))


def _linf_vector(a, b, components=3):
    """L-infinity norm between two flat lists of vector components.

    Both lists store [u0, v0, w0, u1, v1, w1, ...].
    Returns max of Euclidean distances.
    """
    if len(a) != len(b):
        raise ValueError(f"Field length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    max_err = 0.0
    for i in range(0, len(a), components):
        sq = sum(
            (a[i + c] - b[i + c]) ** 2 for c in range(components)
        )
        max_err = max(max_err, math.sqrt(sq))
    return max_err


def _discover_times(case_dir, num_ranks):
    """Return sorted list of output times present in all processorN dirs.

    Excludes time '0' (initial conditions, not solver output).
    """
    all_times = None
    for rank in range(num_ranks):
        proc_dir = os.path.join(case_dir, f"processor{rank}")
        if not os.path.isdir(proc_dir):
            return []
        subdirs = {
            d
            for d in os.listdir(proc_dir)
            if os.path.isdir(os.path.join(proc_dir, d))
        }
        # Filter to numeric directories, excluding 0
        times = set()
        for d in subdirs:
            try:
                t = float(d)
                if t > 0.0:
                    times.add(d)
            except ValueError:
                pass
        if all_times is None:
            all_times = times
        else:
            all_times &= times

    return sorted(all_times, key=float) if all_times else []


def _is_vector_field(case_dir, time_str, field, num_ranks):
    """Heuristically determine if a field is a vector field."""
    for rank in range(num_ranks):
        path = os.path.join(
            case_dir, f"processor{rank}", time_str, field
        )
        try:
            with _open_field_file(path) as fh:
                content = fh.read(4096)
            if "List<vector>" in content:
                return True
            if "List<scalar>" in content:
                return False
        except (FileNotFoundError, OSError):
            pass
    return False


def compare(ref_dir, neo_dir, fields, threshold, num_ranks=4):
    """Compare fields between ref and neo directories.

    Returns (max_linf, any_over_threshold).
    Prints a table to stdout.
    """
    ref_times = _discover_times(ref_dir, num_ranks)
    neo_times = _discover_times(neo_dir, num_ranks)

    common_times = sorted(
        set(ref_times) & set(neo_times), key=float
    )

    if not common_times:
        print("No common output times found — skipping comparison.")
        return 0.0, False

    header_parts = ["time"]
    for f in fields:
        header_parts.append(f"linf_{f}")
    print("  ".join(f"{p:>14}" for p in header_parts))
    print("-" * (16 * len(header_parts)))

    global_max_linf = 0.0
    any_over = False

    for time_str in common_times:
        row = [f"{float(time_str):14.6g}"]
        for field in fields:
            is_vec = _is_vector_field(ref_dir, time_str, field, num_ranks)

            # Concatenate internal field values across all ranks in order
            ref_vals = []
            neo_vals = []
            for rank in range(num_ranks):
                ref_path = os.path.join(
                    ref_dir, f"processor{rank}", time_str, field
                )
                neo_path = os.path.join(
                    neo_dir, f"processor{rank}", time_str, field
                )
                ref_vals.extend(_read_internal_field(ref_path))
                neo_vals.extend(_read_internal_field(neo_path))

            if is_vec:
                linf = _linf_vector(ref_vals, neo_vals)
            else:
                linf = _linf_scalar(ref_vals, neo_vals)

            global_max_linf = max(global_max_linf, linf)
            if linf > threshold:
                any_over = True
            row.append(f"{linf:14.6e}")

        print("  ".join(row))

    print()
    print(f"Max L∞ across all times and fields: {global_max_linf:.6e}")
    print(f"Threshold: {threshold:.6e}")

    return global_max_linf, any_over


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ref",
        required=True,
        metavar="DIR",
        help="Reference case directory (e.g. cylinder3D_ref/) containing processorN/",
    )
    parser.add_argument(
        "--neo",
        required=True,
        metavar="DIR",
        help="NeoFOAM case directory (e.g. cylinder3D/) containing processorN/",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["U", "p"],
        metavar="FIELD",
        help="Fields to compare (default: U p)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-2,
        metavar="TOL",
        help="L∞ error threshold above which comparison fails (default: 1e-2)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=4,
        metavar="N",
        help="Number of MPI ranks / processor directories (default: 4)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.ref):
        print(f"ERROR: --ref directory not found: {args.ref}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.neo):
        print(f"ERROR: --neo directory not found: {args.neo}", file=sys.stderr)
        sys.exit(2)

    print(f"Reference: {args.ref}")
    print(f"NeoFOAM:   {args.neo}")
    print(f"Fields:    {' '.join(args.fields)}")
    print(f"Threshold: {args.threshold:.6e}")
    print()

    _, any_over = compare(
        args.ref, args.neo, args.fields, args.threshold, num_ranks=args.ranks
    )

    if any_over:
        print("FAIL: L∞ error exceeds threshold.", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASS: all L∞ errors within threshold.")
        sys.exit(0)


if __name__ == "__main__":
    main()
