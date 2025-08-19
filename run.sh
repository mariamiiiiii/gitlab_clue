#!/usr/bin/env bash
set -e
#set -x
rm -rf build


CUDA_BASE=/usr/local/cuda-12.8
ROCM_BASE=/opt/rocm-6.4.1



export PATH="${ROCM_BASE}/llvm/bin:${ROCM_BASE}/bin:$PATH"
export CC=clang
export CXX=clang++
export HSA_XNACK=1        # enables recoverable GPU page‑faults


AMDDEVICE_DIR="${ROCM_BASE}/lib/cmake/AMDDeviceLibs"

# --- choose back-end ---------------------------------------------------------
BACKEND=${1:-HIP}          # pass CUDA or NONE to override
if [ "${BACKEND}" = "CUDA" ]; then
  PREFIX_PATH="${CUDA_BASE}"
else                          # HIP
  PREFIX_PATH="${ROCM_BASE}"
# ${ROCM_BASE}/lib/cmake;$(pwd)/cmake/stubs/hsa-runtime64;$(pwd)/cmake/stubs/AMDDeviceLibs"
fi

cmake -B build -S . -L \
  -DGPU_BACKEND=${BACKEND} \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
  # -DALPAKA_DIR="${ALPAKA_BASE}" \
  # -DCHECK_CUDA_VERSION=OFF

cmake --build build -j$(nproc)

  for i in {0..10}
  do
    ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u $i
  done