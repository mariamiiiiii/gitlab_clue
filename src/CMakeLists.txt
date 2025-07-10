if(CMAKE_CUDA_COMPILER)
  # Native C++ CLUE and native CUDA CLUE
  add_subdirectory(clue)

  # Native C++ CLUE and CUDA CLUE using ALPAKA
  add_subdirectory(clue_cuda_alpaka)

endif(CMAKE_CUDA_COMPILER)

# Native C++ CLUE and TBB CLUE using ALPAKA
#add_subdirectory(clue_tbb_alpaka)
