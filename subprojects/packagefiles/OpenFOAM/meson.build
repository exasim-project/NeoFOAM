project('petsc', 'c', 'cpp',
        meson_version : '>=0.56.0',
)
install_prefix = get_option('prefix')
install_libdir = get_option('libdir')
install_includedir = get_option('includedir')
# run_command('./build.sh', install_prefix, install_libdir, install_includedir, capture: true, check: true)
run_command('./build.sh', capture: true, check: true)

# petsc_dep = p.dependency('petsc')
# meson.override_dependency('petsc', petsc_dep)
