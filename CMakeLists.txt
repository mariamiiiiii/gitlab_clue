cmake_minimum_required(VERSION 3.8)

project(
  CLUE
  VERSION 1.1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

# Location of the ALPAKA
set(ALPAKA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/alpaka CACHE PATH "Path to ALPAKA (by default: submodule)")

# Location of TBB
#set(TBB_DIR "/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/tbb/2020_U2-ghbfee/cmake/TBB/")

# Activate VERBOSE output
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CXX_STANDARD 17)
  #set(CMAKE_CUDA_ARCHITECTURES "60;70;75")
else()
  message(STATUS "No CUDA compiler found. Still, you can run the C++ version!")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

# Find Boost
set(Boost_DEBUG 1)
find_package(Boost REQUIRED)

# Include Boost headers
include_directories(${Boost_INCLUDE_DIRS})

if(Boost_FOUND)
  message(STATUS "Boost package found!")
endif()

find_package(TBB REQUIRED)
if(TBB_FOUND)
  message(STATUS "TBB package found!")
endif()

add_subdirectory(clueLib)
add_subdirectory(src)

# SOME TESTS
enable_testing()
function(run_test_cpu target label repeats)
  FILE(GLOB inputs "${CMAKE_CURRENT_SOURCE_DIR}/data/input/*.csv")
  foreach(input_file IN LISTS inputs)
    add_test(NAME ${input_file}_${label}_CPU COMMAND ${target} -i ${input_file} -d 7.0 -r10.0 -o 2 -e ${repeats} -v)
  endforeach()
endfunction()

function(run_test_gpu target label repeats)
  FILE(GLOB inputs "${CMAKE_CURRENT_SOURCE_DIR}/data/input/*.csv")
  foreach(input_file IN LISTS inputs)
    add_test(NAME ${input_file}_${label}_GPU COMMAND ${target} -i ${input_file} -d 7.0 -r10.0 -o 2 -e ${repeats} -v -u)
  endforeach()
endfunction()

if(CMAKE_CUDA_COMPILER)
  run_test_cpu(./src/clue/main main 4)
  run_test_cpu(./src/clue_cuda_alpaka/mainAlpakaCUDA Alpaka 4)
  run_test_gpu(./src/clue/main CUDA 100)
  run_test_gpu(./src/clue_cuda_alpaka/mainAlpakaCUDA AlpakaCUDA 100)
else()
  run_test_cpu(./src/clue_tbb_alpaka/mainAlpakaTBB TBB 4)
endif()

#--- add version files ---------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLUEVersion.h
               ${CMAKE_CURRENT_BINARY_DIR}/CLUEVersion.h )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CLUEVersion.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CLUE )
include(cmake/CLUECreateConfig.cmake)

install(EXPORT CLUETargets
  NAMESPACE ${PROJECT_NAME}::
  FILE "${PROJECT_NAME}Targets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/"
  )

message(STATUS "Finished!")

