cmake_minimum_required (VERSION 3.24)

# Set PF_USE_CUDA=OFF to build against the library's OpenMP backend, which
# needs no GPU and no nvcc, only the CCCL headers (thrust / cub / libcu++).
option (PF_USE_CUDA "Build the filters for CUDA rather than OpenMP" ON)

# -march=native is worth 15-20% single threaded on the Kalman heavy filters.
# Turn it OFF when building a wheel on a machine other than the one that will
# run it, since the binary will use whatever ISA the build host had.
option (PF_MARCH_NATIVE "Compile for the build host's own CPU" ON)

if (PF_USE_CUDA)
  project (robomaster_particle_filters LANGUAGES CXX CUDA)
else ()
  project (robomaster_particle_filters LANGUAGES CXX)
endif ()

include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wpedantic -Wall -Wextra -flto -fopenmp -lgomp")

if (PF_MARCH_NATIVE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

add_subdirectory (deps/pybind11)

pybind11_add_module (
  robomaster_particle_filters
  src/robomaster_particle_filters/robomaster_particle_filters.cc

  src/plate_orbit/init.cc
  src/plate_orbit/particle_filter.cu

  src/fast_plate_orbit/init.cc
  src/fast_plate_orbit/particle_filter.cu

  src/fast_plate_orbit_with_z_offset/init.cc
  src/fast_plate_orbit_with_z_offset/particle_filter.cu

  src/plate_orbit_v2/init.cc
  src/plate_orbit_v2/particle_filter.cu

  src/plate_orbit_v3/init.cc
  src/plate_orbit_v3/particle_filter.cu
)

# The .cu translation units contain no CUDA-specific syntax; the backend is
# selected entirely by PF_TARGET_CUDA / PF_TARGET_OMP in target_config.h. On
# the OpenMP path they are simply compiled as C++.
if (NOT PF_USE_CUDA)
  set_source_files_properties (
    src/plate_orbit/particle_filter.cu
    src/fast_plate_orbit/particle_filter.cu
    src/fast_plate_orbit_with_z_offset/particle_filter.cu
    src/plate_orbit_v2/particle_filter.cu
    src/plate_orbit_v3/particle_filter.cu
    PROPERTIES LANGUAGE CXX
  )
endif ()

target_compile_features (
  robomaster_particle_filters PRIVATE
  cxx_std_20
)

target_include_directories (
  robomaster_particle_filters PRIVATE
  include
  deps/eigen
  deps/particle-filter
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${PF_CCCL_INCLUDE_DIRECTORIES}
)

target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: PF_TARGET_OMP>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP>)
target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CXX>: THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP>)

if (PF_USE_CUDA)
  target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: PF_TARGET_CUDA>)
  target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP>)
  target_compile_definitions (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA>)

  set_target_properties (robomaster_particle_filters PROPERTIES CUDA_ARCHITECTURES native)
  target_compile_options (robomaster_particle_filters PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                         --expt-relaxed-constexpr
                         --extended-lambda
                         --use_fast_math
                         >)
endif ()

install (TARGETS robomaster_particle_filters DESTINATION .)
