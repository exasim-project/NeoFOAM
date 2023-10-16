project('NeoFOAM', 'cpp',
  version: '0.0.0',
  license: 'GPL-3.0-or-later',
  default_options : ['cpp_std=c++20']
  )

inc = include_directories('include')

subdir('include')
subdir('src')
subdir('tests')

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')
gmock_dep = gtest_proj.get_variable('gmock_dep')

app_src = files('src/hello_world.cpp')

test_src = files('tests/test_hello_world.cpp')

e = executable(
  'gtest',
  test_src + app_src,
  include_directories: inc,
  dependencies : [
    gtest_dep,
    gmock_dep,
    ]
)


test('gtest', e)
