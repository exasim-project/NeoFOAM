project('OpenFOAM', 'c', 'cpp',
        meson_version : '>=0.56.0',
)
of_path = get_option('of_path')
of_platform = get_option('of_platform')

cppc = meson.get_compiler('cpp')

compiler_args = ''
if cppc.get_id() == 'gcc'
    compiler_args = '-DWM_COMPILER="Gcc"'
elif cppc.get_id() == 'clang'
    compiler_args = '-DWM_COMPILER="Clang"'
else
    error('Unknown Compiler. I do not know what to fill in here for the dots: -DWM_COMPILER="..."')
endif

# run_command('./build.sh', install_prefix, install_libdir, install_includedir, capture: true, check: true)
# not enough ram with -j
# run_command('./Allwmake','--prefix=false','-q', '-j 6', '-l', capture: true, check: true)


# Fetch current project arguments using the utility function
comp_args = ['-DWM_LABEL_SIZE=' + get_option('WM_LABEL_SIZE'),
             '-DWM_ARCH=' + get_option('WM_ARCH'),
            '-DWM_' + get_option('WM_PRECISION_OPTION'),
            '-DNoRepository',
            '-DOPENFOAM=2306',
            '-DOMPI_SKIP_MPICXX',
            '-ftemplate-depth-100',
            '-m64',
            '-Wl,--add-needed',
            '-Wl,--no-as-needed',
            compiler_args
            ]
pstream = cppc.find_library('Pstream', required : true, dirs : of_path + of_platform + '/sys-openmpi')
openfoam = cppc.find_library('OpenFOAM', required : true, dirs : of_path + of_platform)
finiteVolume = cppc.find_library('finiteVolume', required : true, dirs : of_path + of_platform)
meshTools = cppc.find_library('meshTools', required : true, dirs : of_path + of_platform)
openfoam_dep = declare_dependency(
  compile_args: comp_args,
  dependencies : [openfoam, pstream],
  include_directories : [of_path + 'src/OSspecific/POSIX/lnInclude', of_path + 'src/OpenFOAM/lnInclude' ]
)
finiteVolume_dep = declare_dependency(
  dependencies : [finiteVolume],
  include_directories : [of_path + 'src/finiteVolume/lnInclude' ]
)
meshTools_dep = declare_dependency(
  dependencies : [meshTools],
  include_directories : [of_path + 'src/meshTools/lnInclude' ]
)
