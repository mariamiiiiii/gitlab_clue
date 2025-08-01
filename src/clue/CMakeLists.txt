# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

# if(CMAKE_CUDA_COMPILER)
#   set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
#   add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)

#   target_compile_definitions(main PRIVATE)
#   target_compile_options(main PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:
#                         -Xcompiler;
#                         -m64;
#                         -expt-relaxed-constexpr;
#                         -w;
#   >
#   )

#   target_link_libraries(main PRIVATE CLUELib)

#   target_include_directories(main PUBLIC
#   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
#   )
# endif(CMAKE_CUDA_COMPILER)

# ------------------------------------------------------------
# LUMI IMPLEMENTATION BEGIN
# ------------------------------------------------------------
# ------------------------------------------------------------
# Native HIP executable
# ------------------------------------------------------------
if(GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc
                                PROPERTIES LANGUAGE HIP)

    add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)

    # Link against the ROCm device runtime
    target_link_libraries(main PRIVATE CLUELib hip::device)

    # Make sure we compile for the right GPU
    target_compile_options(main PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${CMAKE_HIP_ARCHITECTURES}>
    )
endif()
# ------------------------------------------------------------
# LUMI IMPLEMENTATION END
# ------------------------------------------------------------