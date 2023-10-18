project('NeoFOAM', 'cpp',
  version: '0.0.0',
  license: 'GPL-3.0-or-later'
  )

of_proj = subproject('OpenFOAM-v2306')
openfoam_dep = of_proj.get_variable('openfoam_dep')
meshTools_dep = of_proj.get_variable('meshTools_dep')
finiteVolume_dep = of_proj.get_variable('finiteVolume_dep')

inc = include_directories('include')

subdir('include')
subdir('src')
subdir('applications')
subdir('tests')

