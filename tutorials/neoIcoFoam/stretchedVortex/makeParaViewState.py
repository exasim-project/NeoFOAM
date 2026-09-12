# SPDX-FileCopyrightText: 2023 - 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense
"""Build the stretchedVortex ParaView state.

Run with ParaView's python:

    pvpython make_pvsm.py <case.foam> <out.pvsm>
"""
import sys
from paraview.simple import *

foam, out = sys.argv[1], sys.argv[2]
paraview.simple._DisableFirstRenderCameraReset()

# --- case -------------------------------------------------------------------
case = OpenFOAMReader(registrationName="case.foam", FileName=foam)
case.MeshRegions = ["internalMesh"]
case.CellArrays = ["U", "p"]
case.Createcelltopointfiltereddata = 1
case.UpdatePipeline()
times = case.TimestepValues or [0.0]

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1280, 860]

# --- azimuthal velocity, the quantity the case is about ---------------------
swirl = Calculator(registrationName="uTheta", Input=case)
swirl.AttributeType = "Point Data"
swirl.ResultArrayName = "uTheta"
swirl.Function = "(-coordsY*U_X + coordsX*U_Y)/sqrt(coordsX^2 + coordsY^2 + 1e-12)"

# --- core surface: an iso-surface of vorticity magnitude --------------------
# Vorticity decays monotonically outwards, so one level gives one tube. (An
# iso-surface of uTheta has two branches -- inside and outside the swirl peak --
# and the outer shell hides the core.) On the axis |omega| = Gamma/(pi delta^2),
# so it grows 0.88 -> 8 as the core contracts: the level below is crossed at
# t ~ 0.25 and the tube tightens from there.
grad = Gradient(registrationName="vorticity", Input=swirl)
grad.ScalarArray = ["POINTS", "U"]
grad.ComputeGradient = 0
grad.ComputeVorticity = 1
grad.VorticityArrayName = "Vorticity"
vmag = Calculator(registrationName="vortMag", Input=grad)
vmag.AttributeType = "Point Data"
vmag.ResultArrayName = "vortMag"
vmag.Function = "mag(Vorticity)"
core = Contour(registrationName="core", Input=vmag)
core.ContourBy = ["POINTS", "vortMag"]
core.Isosurfaces = [1.0]
coreDisplay = Show(core, view, "GeometryRepresentation")
ColorBy(coreDisplay, ("POINTS", "uTheta"))
coreDisplay.Opacity = 0.55
lut = GetColorTransferFunction("uTheta")
lut.ApplyPreset("Plasma (matplotlib)", True)
lut.RescaleTransferFunction(0.15, 0.50)
coreDisplay.SetScalarBarVisibility(view, True)

# --- streamlines through the 3D field ---------------------------------------
lines = StreamTracer(registrationName="streamlines", Input=swirl,
                     SeedType="Line")
lines.Vectors = ["POINTS", "U"]
lines.SeedType.Point1 = [-0.92, 0.0, 0.004]
lines.SeedType.Point2 = [0.92, 0.0, 0.004]
lines.SeedType.Resolution = 24
lines.MaximumStreamlineLength = 12.0
lines.IntegrationDirection = "FORWARD"
tubes = Tube(registrationName="streamtubes", Input=lines)
tubes.Scalars = ["POINTS", "uTheta"]
tubes.Vectors = ["POINTS", "U"]
tubes.Radius = 0.004
tubeDisplay = Show(tubes, view, "GeometryRepresentation")
ColorBy(tubeDisplay, None)
tubeDisplay.AmbientColor = [0.2, 0.2, 0.2]
tubeDisplay.DiffuseColor = [0.2, 0.2, 0.2]

# --- mid-plane context ------------------------------------------------------
slice_ = Slice(registrationName="midplane", Input=swirl)
slice_.SliceType.Origin = [0.0, 0.0, 0.0]
slice_.SliceType.Normal = [0.0, 0.0, 1.0]
sliceDisplay = Show(slice_, view, "GeometryRepresentation")
ColorBy(sliceDisplay, ("POINTS", "uTheta"))
sliceDisplay.Opacity = 0.25
sliceDisplay.SetScalarBarVisibility(view, False)

# --- outline + camera -------------------------------------------------------
# The reader's own outline representation, not FeatureEdges: that filter
# segfaults ParaView 5.13 on this reader's multiblock output.
outline = Show(case, view, "OutlineRepresentation")
outline.AmbientColor = [0.6, 0.6, 0.6]
outline.DiffuseColor = [0.6, 0.6, 0.6]

view.CameraPosition = [3.2, -3.6, 2.2]
view.CameraFocalPoint = [0.0, 0.0, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.Background = [1.0, 1.0, 1.0]
view.UseColorPaletteForBackground = 0
scene.AnimationTime = times[-1]
Render()

SaveState(out)
print(f"wrote {out} ({len(times)} time steps, t = {times[0]} .. {times[-1]})")
