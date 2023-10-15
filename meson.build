project('NeoFOAM', 'cpp',
  version: '0.0.0',
  license: 'GPL-3.0-or-later'
  )

of_thirdparty_proj = subproject('openfoam-thirdparty')
openfoam_proj = subproject('openfoam')

inc = include_directories('include')

subdir('include')
subdir('src')
subdir('tests')

