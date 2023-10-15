#!/bin/sh
# Run from OPENFOAM src/ directory only
cd "${0%/*}" || exit
wmake -check-dir "$WM_PROJECT_DIR/src" 2>/dev/null || {
    echo "Error (${0##*/}) : not located in \$WM_PROJECT_DIR/src"
    echo "    Check your OpenFOAM environment and installation"
    exit 1
}
if [ -f "$WM_PROJECT_DIR"/wmake/scripts/AllwmakeParseArguments ]
then  . "$WM_PROJECT_DIR"/wmake/scripts/AllwmakeParseArguments || \
    echo "Argument parse error"
else
    echo "Error (${0##*/}) : WM_PROJECT_DIR appears to be incorrect"
    echo "    Check your OpenFOAM environment and installation"
    exit 1
fi

echo ========================================
echo Compile OpenFOAM libraries
echo ========================================

#------------------------------------------------------------------------------

wmakeLnInclude -u OpenFOAM
wmakeLnInclude -u OSspecific/"${WM_OSTYPE:-POSIX}"

# Update version info (as required)
OpenFOAM/Alltouch -check 2>/dev/null

OSspecific/"${WM_OSTYPE:-POSIX}"/Allwmake $targetType $*

case "$WM_COMPILER" in
(Mingw*)
    # Pstream/OpenFOAM cyclic dependency
    # 1st pass: link as Pstream as single .o object
    WM_MPLIB=dummy Pstream/Allwmake libo
    FOAM_LINK_DUMMY_PSTREAM=libo wmake $targetType OpenFOAM

    # 2nd pass: link Pstream.{dll,so} against libOpenFOAM.{dll,so}
    Pstream/Allwmake $targetType $*

    # Force relink libOpenFOAM.{dll,so} against libPstream.{dll,so}
    OpenFOAM/Alltouch 2>/dev/null
    ;;
(*)
    Pstream/Allwmake $targetType $*
    ;;
esac

wmake $targetType OpenFOAM

wmake $targetType fileFormats
wmake $targetType surfMesh
wmake $targetType meshTools

wmake $targetType finiteArea
wmake $targetType finiteVolume
