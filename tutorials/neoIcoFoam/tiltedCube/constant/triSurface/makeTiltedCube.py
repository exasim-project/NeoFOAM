#!/usr/bin/env python3
# Generate an ASCII STL of a cube rotated relative to the background grid axes.
# A tilted cube forces snappyHexMesh to snap onto non-axis-aligned faces, producing
# skewed / non-orthogonal cells -- a deliberately "complex" surface for distributed tests.
import math

# --- cube geometry (domain is x[0,3] y[0,1.2] z[0,1.2]; see blockMeshDict) ---
center = (1.10, 0.60, 0.60)
half = 0.26  # half side length -> side 0.52

# --- tilt: rotate about z then about y by deliberately non-trivial angles ---
az = math.radians(27.0)  # yaw   (about z)
ay = math.radians(18.0)  # pitch (about y)


def rot(p):
    x, y, z = p
    # Rz
    x1 = x * math.cos(az) - y * math.sin(az)
    y1 = x * math.sin(az) + y * math.cos(az)
    z1 = z
    # Ry
    x2 = x1 * math.cos(ay) + z1 * math.sin(ay)
    y2 = y1
    z2 = -x1 * math.sin(ay) + z1 * math.cos(ay)
    return (x2, y2, z2)


# 8 corners in local frame
locals_ = [
    (-half, -half, -half),
    (half, -half, -half),
    (half, half, -half),
    (-half, half, -half),
    (-half, -half, half),
    (half, -half, half),
    (half, half, half),
    (-half, half, half),
]
verts = []
for p in locals_:
    rx, ry, rz = rot(p)
    verts.append((rx + center[0], ry + center[1], rz + center[2]))

# 6 faces (quads, CCW seen from outside) -> 12 triangles
faces = [
    (0, 3, 2, 1),  # bottom (-z)
    (4, 5, 6, 7),  # top    (+z)
    (0, 1, 5, 4),  # -y
    (2, 3, 7, 6),  # +y
    (1, 2, 6, 5),  # +x
    (0, 4, 7, 3),  # -x
]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    m = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1.0
    return (a[0] / m, a[1] / m, a[2] / m)


def tri(out, i, j, k):
    p0, p1, p2 = verts[i], verts[j], verts[k]
    n = norm(cross(sub(p1, p0), sub(p2, p0)))
    out.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
    out.append("    outer loop\n")
    for p in (p0, p1, p2):
        out.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
    out.append("    endloop\n")
    out.append("  endfacet\n")


lines = ["solid tiltedCube\n"]
for a, b, c, d in faces:
    tri(lines, a, b, c)
    tri(lines, a, c, d)
lines.append("endsolid tiltedCube\n")

with open("tiltedCube.stl", "w") as f:
    f.writelines(lines)
print("wrote tiltedCube.stl with", (len(lines) - 2) // 7, "facets")
