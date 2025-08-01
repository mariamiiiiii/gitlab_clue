#!/usr/bin/env bash
set -e
#set -x
rm -rf build

# --- CMS environment ---------------------------------------------------------
cd /data/cmssw/el8_amd64_gcc12/cms/cmssw/CMSSW_15_0_6
eval `scram runtime -sh`
CUDA_BASE=$(scram tool tag cuda  CUDA_BASE)
ROCM_BASE=$(scram tool tag rocm  ROCM_BASE)
TBB_BASE=$(scram tool tag tbb   TBB_BASE)
BOOST_BASE=$(scram tool tag boost BOOST_BASE)
# ALPAKA_BASE=$(scram tool tag alpaka ALPAKA_BASE)
cd -


export PATH="${ROCM_BASE}/llvm/bin:${ROCM_BASE}/bin:$PATH"
export CC=clang
export CXX=clang++
export HSA_XNACK=1        # enables recoverable GPU page‑faults


AMDDEVICE_DIR="${ROCM_BASE}/lib/cmake/AMDDeviceLibs"

# --- choose back-end ---------------------------------------------------------
BACKEND=${1:-HIP}          # pass CUDA or NONE to override
if [ "${BACKEND}" = "CUDA" ]; then
  PREFIX_PATH="${CUDA_BASE};${BOOST_BASE};"
else                          # HIP
  PREFIX_PATH="${ROCM_BASE};${BOOST_BASE};"
# ${ROCM_BASE}/lib/cmake;$(pwd)/cmake/stubs/hsa-runtime64;$(pwd)/cmake/stubs/AMDDeviceLibs"
fi

cmake -B build -S . -L \
  -DGPU_BACKEND=${BACKEND} \
  -DCMAKE_PREFIX_PATH="${PREFIX_PATH}" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
  # -DALPAKA_DIR="${ALPAKA_BASE}" \
  # -DCHECK_CUDA_VERSION=OFF

cmake --build build -j$(nproc)

# CPU
#   ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v
# GPU
  rm -f ../Results/results_unified*.csv

  for i in {0..10}
  do
    ./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u $i
  done